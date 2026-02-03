"""
Strategy Parameters Module
==========================

Addresses issues:
- #Q20: Magic numbers in RSI calculation (70/30 thresholds)
- #Q21: No parameter sensitivity analysis

Features:
- Centralized strategy parameter definitions
- Configurable thresholds with validation
- Parameter sensitivity analysis framework
- Parameter optimization support
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Iterator
from collections import defaultdict

logger = logging.getLogger(__name__)


# =========================================================================
# STRATEGY PARAMETER DEFINITIONS (#Q20)
# =========================================================================

@dataclass
class ParameterBounds:
    """
    Defines valid bounds for a strategy parameter.

    Eliminates magic numbers by providing named, validated parameters.
    """
    min_value: float
    max_value: float
    default_value: float
    step: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        """Validate bounds configuration."""
        if self.min_value > self.max_value:
            raise ValueError(f"min_value ({self.min_value}) > max_value ({self.max_value})")
        if not self.min_value <= self.default_value <= self.max_value:
            raise ValueError(
                f"default_value ({self.default_value}) not in "
                f"[{self.min_value}, {self.max_value}]"
            )

    def validate(self, value: float) -> bool:
        """Check if value is within bounds."""
        return self.min_value <= value <= self.max_value

    def generate_range(self) -> list[float]:
        """Generate all valid values within bounds."""
        values = []
        current = self.min_value
        while current <= self.max_value:
            values.append(current)
            current += self.step
        return values


class RSIParameters:
    """
    RSI indicator parameters with named constants (#Q20).

    Replaces magic numbers 70/30 with configurable, documented thresholds.
    """

    # Standard thresholds with academic/industry backing
    OVERBOUGHT_DEFAULT = 70.0
    OVERSOLD_DEFAULT = 30.0

    # Alternative thresholds for different market conditions
    OVERBOUGHT_AGGRESSIVE = 80.0  # Strong trending markets
    OVERSOLD_AGGRESSIVE = 20.0

    OVERBOUGHT_CONSERVATIVE = 65.0  # Mean-reverting markets
    OVERSOLD_CONSERVATIVE = 35.0

    # Parameter bounds for optimization
    OVERBOUGHT_BOUNDS = ParameterBounds(
        min_value=60.0,
        max_value=90.0,
        default_value=70.0,
        step=5.0,
        description="RSI level above which asset is considered overbought"
    )

    OVERSOLD_BOUNDS = ParameterBounds(
        min_value=10.0,
        max_value=40.0,
        default_value=30.0,
        step=5.0,
        description="RSI level below which asset is considered oversold"
    )

    PERIOD_BOUNDS = ParameterBounds(
        min_value=5,
        max_value=30,
        default_value=14,
        step=1.0,
        description="Lookback period for RSI calculation (Wilder's default: 14)"
    )

    def __init__(
        self,
        overbought: float | None = None,
        oversold: float | None = None,
        period: int | None = None,
    ):
        """
        Initialize RSI parameters.

        Args:
            overbought: Overbought threshold (default 70)
            oversold: Oversold threshold (default 30)
            period: Lookback period (default 14)
        """
        self.overbought = overbought or self.OVERBOUGHT_DEFAULT
        self.oversold = oversold or self.OVERSOLD_DEFAULT
        self.period = period or int(self.PERIOD_BOUNDS.default_value)

        self._validate()

    def _validate(self) -> None:
        """Validate parameter consistency."""
        if self.oversold >= self.overbought:
            raise ValueError(
                f"Oversold ({self.oversold}) must be less than "
                f"overbought ({self.overbought})"
            )
        if not self.OVERBOUGHT_BOUNDS.validate(self.overbought):
            logger.warning(
                f"Overbought threshold {self.overbought} outside recommended range "
                f"[{self.OVERBOUGHT_BOUNDS.min_value}, {self.OVERBOUGHT_BOUNDS.max_value}]"
            )
        if not self.OVERSOLD_BOUNDS.validate(self.oversold):
            logger.warning(
                f"Oversold threshold {self.oversold} outside recommended range "
                f"[{self.OVERSOLD_BOUNDS.min_value}, {self.OVERSOLD_BOUNDS.max_value}]"
            )

    def is_overbought(self, rsi_value: float) -> bool:
        """Check if RSI indicates overbought condition."""
        return rsi_value >= self.overbought

    def is_oversold(self, rsi_value: float) -> bool:
        """Check if RSI indicates oversold condition."""
        return rsi_value <= self.oversold

    def get_signal_strength(self, rsi_value: float) -> float:
        """
        Get signal strength from RSI value.

        Returns:
            -1 to 1 where:
            - Negative = oversold (buy signal)
            - Positive = overbought (sell signal)
            - Near zero = neutral
        """
        midpoint = (self.overbought + self.oversold) / 2

        if rsi_value >= self.overbought:
            # Overbought region (sell signal)
            excess = (rsi_value - self.overbought) / (100 - self.overbought)
            return min(1.0, 0.5 + 0.5 * excess)
        elif rsi_value <= self.oversold:
            # Oversold region (buy signal)
            excess = (self.oversold - rsi_value) / self.oversold
            return max(-1.0, -0.5 - 0.5 * excess)
        else:
            # Neutral region
            return (rsi_value - midpoint) / (self.overbought - midpoint) * 0.5

    @classmethod
    def for_trending_market(cls) -> "RSIParameters":
        """Create parameters optimized for trending markets."""
        return cls(
            overbought=cls.OVERBOUGHT_AGGRESSIVE,
            oversold=cls.OVERSOLD_AGGRESSIVE,
        )

    @classmethod
    def for_ranging_market(cls) -> "RSIParameters":
        """Create parameters optimized for ranging/mean-reverting markets."""
        return cls(
            overbought=cls.OVERBOUGHT_CONSERVATIVE,
            oversold=cls.OVERSOLD_CONSERVATIVE,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "overbought": self.overbought,
            "oversold": self.oversold,
            "period": self.period,
        }


class MACDParameters:
    """MACD indicator parameters."""

    # Standard MACD parameters (Appel's original)
    FAST_PERIOD_DEFAULT = 12
    SLOW_PERIOD_DEFAULT = 26
    SIGNAL_PERIOD_DEFAULT = 9

    FAST_PERIOD_BOUNDS = ParameterBounds(
        min_value=5,
        max_value=20,
        default_value=12,
        step=1.0,
        description="Fast EMA period for MACD"
    )

    SLOW_PERIOD_BOUNDS = ParameterBounds(
        min_value=15,
        max_value=50,
        default_value=26,
        step=1.0,
        description="Slow EMA period for MACD"
    )

    SIGNAL_PERIOD_BOUNDS = ParameterBounds(
        min_value=5,
        max_value=15,
        default_value=9,
        step=1.0,
        description="Signal line EMA period"
    )

    def __init__(
        self,
        fast_period: int | None = None,
        slow_period: int | None = None,
        signal_period: int | None = None,
    ):
        self.fast_period = fast_period or self.FAST_PERIOD_DEFAULT
        self.slow_period = slow_period or self.SLOW_PERIOD_DEFAULT
        self.signal_period = signal_period or self.SIGNAL_PERIOD_DEFAULT

        self._validate()

    def _validate(self) -> None:
        """Validate parameter consistency."""
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"Fast period ({self.fast_period}) must be less than "
                f"slow period ({self.slow_period})"
            )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
        }


class BollingerBandsParameters:
    """Bollinger Bands indicator parameters."""

    PERIOD_DEFAULT = 20
    STD_DEV_DEFAULT = 2.0

    PERIOD_BOUNDS = ParameterBounds(
        min_value=10,
        max_value=50,
        default_value=20,
        step=5.0,
        description="Lookback period for SMA and standard deviation"
    )

    STD_DEV_BOUNDS = ParameterBounds(
        min_value=1.0,
        max_value=3.0,
        default_value=2.0,
        step=0.5,
        description="Number of standard deviations for bands"
    )

    def __init__(
        self,
        period: int | None = None,
        std_dev: float | None = None,
    ):
        self.period = period or self.PERIOD_DEFAULT
        self.std_dev = std_dev or self.STD_DEV_DEFAULT

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "period": self.period,
            "std_dev": self.std_dev,
        }


# =========================================================================
# PARAMETER SENSITIVITY ANALYSIS (#Q21)
# =========================================================================

@dataclass
class SensitivityResult:
    """Result of a single sensitivity test."""
    parameter_name: str
    parameter_value: float
    metric_name: str
    metric_value: float
    baseline_value: float
    change_pct: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "parameter_name": self.parameter_name,
            "parameter_value": self.parameter_value,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "baseline_value": self.baseline_value,
            "change_pct": self.change_pct,
        }


@dataclass
class SensitivityReport:
    """Complete sensitivity analysis report."""
    strategy_name: str
    generated_at: datetime
    baseline_params: dict[str, Any]
    baseline_metrics: dict[str, float]
    results: list[SensitivityResult]
    stability_scores: dict[str, float] = field(default_factory=dict)

    def get_most_sensitive_parameter(self, metric: str = "sharpe_ratio") -> str | None:
        """Find parameter with highest sensitivity to given metric."""
        param_impacts = defaultdict(list)

        for result in self.results:
            if result.metric_name == metric:
                param_impacts[result.parameter_name].append(abs(result.change_pct))

        if not param_impacts:
            return None

        avg_impacts = {
            param: statistics.mean(impacts)
            for param, impacts in param_impacts.items()
        }

        return max(avg_impacts.items(), key=lambda x: x[1])[0]

    def get_stability_score(self, parameter: str) -> float:
        """
        Get stability score for a parameter.

        Higher score = more stable (less sensitive)
        Range: 0-100
        """
        return self.stability_scores.get(parameter, 50.0)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "generated_at": self.generated_at.isoformat(),
            "baseline_params": self.baseline_params,
            "baseline_metrics": self.baseline_metrics,
            "results": [r.to_dict() for r in self.results],
            "stability_scores": self.stability_scores,
        }


class ParameterSensitivityAnalyzer:
    """
    Parameter sensitivity analysis framework (#Q21).

    Analyzes how strategy performance changes with parameter variations.
    Helps identify:
    - Overfitted parameters
    - Stable vs unstable parameter regions
    - Optimal parameter ranges
    """

    def __init__(
        self,
        strategy_evaluator: Callable[[dict], dict[str, float]],
    ):
        """
        Initialize analyzer.

        Args:
            strategy_evaluator: Function that takes parameter dict and returns
                               metrics dict (e.g., {"sharpe_ratio": 1.5, "return": 0.1})
        """
        self.evaluator = strategy_evaluator
        self._results: list[SensitivityResult] = []

    def analyze_parameter(
        self,
        baseline_params: dict[str, Any],
        parameter_name: str,
        test_values: list[float],
        metrics: list[str] | None = None,
    ) -> list[SensitivityResult]:
        """
        Analyze sensitivity to a single parameter.

        Args:
            baseline_params: Baseline parameter values
            parameter_name: Parameter to vary
            test_values: Values to test
            metrics: Which metrics to track (default: all)

        Returns:
            List of sensitivity results
        """
        # Get baseline metrics
        baseline_metrics = self.evaluator(baseline_params)

        results = []
        for value in test_values:
            # Create test params
            test_params = baseline_params.copy()
            test_params[parameter_name] = value

            try:
                test_metrics = self.evaluator(test_params)
            except Exception as e:
                logger.warning(f"Failed to evaluate {parameter_name}={value}: {e}")
                continue

            # Calculate sensitivity for each metric
            for metric_name, metric_value in test_metrics.items():
                if metrics and metric_name not in metrics:
                    continue

                baseline_value = baseline_metrics.get(metric_name, 0)
                if baseline_value != 0:
                    change_pct = (metric_value - baseline_value) / abs(baseline_value) * 100
                else:
                    change_pct = 0 if metric_value == 0 else 100

                result = SensitivityResult(
                    parameter_name=parameter_name,
                    parameter_value=value,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    baseline_value=baseline_value,
                    change_pct=change_pct,
                )
                results.append(result)

        self._results.extend(results)
        return results

    def analyze_all_parameters(
        self,
        baseline_params: dict[str, Any],
        parameter_bounds: dict[str, ParameterBounds],
        num_samples: int = 5,
        metrics: list[str] | None = None,
    ) -> SensitivityReport:
        """
        Analyze sensitivity to all parameters.

        Args:
            baseline_params: Baseline parameter values
            parameter_bounds: Bounds for each parameter
            num_samples: Number of values to test per parameter
            metrics: Which metrics to track

        Returns:
            Complete sensitivity report
        """
        baseline_metrics = self.evaluator(baseline_params)
        all_results = []

        for param_name, bounds in parameter_bounds.items():
            # Generate test values
            step = (bounds.max_value - bounds.min_value) / (num_samples - 1)
            test_values = [bounds.min_value + i * step for i in range(num_samples)]

            results = self.analyze_parameter(
                baseline_params, param_name, test_values, metrics
            )
            all_results.extend(results)

        # Calculate stability scores
        stability_scores = self._calculate_stability_scores(all_results, baseline_metrics)

        return SensitivityReport(
            strategy_name="Strategy",
            generated_at=datetime.now(timezone.utc),
            baseline_params=baseline_params,
            baseline_metrics=baseline_metrics,
            results=all_results,
            stability_scores=stability_scores,
        )

    def _calculate_stability_scores(
        self,
        results: list[SensitivityResult],
        baseline_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Calculate stability score for each parameter."""
        param_changes = defaultdict(list)

        # Focus on Sharpe ratio for stability
        for result in results:
            if result.metric_name == "sharpe_ratio":
                param_changes[result.parameter_name].append(abs(result.change_pct))

        stability_scores = {}
        for param, changes in param_changes.items():
            if changes:
                avg_change = statistics.mean(changes)
                # Convert to 0-100 scale (less change = higher stability)
                stability_scores[param] = max(0, 100 - avg_change)
            else:
                stability_scores[param] = 50.0

        return stability_scores

    def find_optimal_parameters(
        self,
        baseline_params: dict[str, Any],
        parameter_bounds: dict[str, ParameterBounds],
        objective_metric: str = "sharpe_ratio",
        num_iterations: int = 100,
    ) -> dict[str, Any]:
        """
        Find optimal parameters using grid search.

        Args:
            baseline_params: Starting parameters
            parameter_bounds: Search bounds
            objective_metric: Metric to optimize
            num_iterations: Max iterations

        Returns:
            Optimal parameter set
        """
        best_params = baseline_params.copy()
        best_metric = float("-inf")

        # Grid search
        param_grids = {
            name: bounds.generate_range()
            for name, bounds in parameter_bounds.items()
        }

        # Generate combinations (limited by num_iterations)
        from itertools import product

        param_names = list(param_grids.keys())
        param_values = [param_grids[name] for name in param_names]

        iteration = 0
        for values in product(*param_values):
            if iteration >= num_iterations:
                break

            test_params = baseline_params.copy()
            for name, value in zip(param_names, values):
                test_params[name] = value

            try:
                metrics = self.evaluator(test_params)
                metric_value = metrics.get(objective_metric, float("-inf"))

                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = test_params.copy()
            except Exception as e:
                logger.warning(f"Parameter evaluation failed for {test_params}: {e}")
                continue  # More explicit than pass

            iteration += 1

        logger.info(
            f"Found optimal parameters with {objective_metric}={best_metric:.4f}"
        )
        return best_params


class StrategyParameterSet:
    """
    Complete parameter set for a trading strategy.

    Centralizes all parameters with validation and defaults.
    """

    def __init__(
        self,
        strategy_name: str,
        rsi: RSIParameters | None = None,
        macd: MACDParameters | None = None,
        bollinger: BollingerBandsParameters | None = None,
        custom_params: dict[str, Any] | None = None,
    ):
        self.strategy_name = strategy_name
        self.rsi = rsi or RSIParameters()
        self.macd = macd or MACDParameters()
        self.bollinger = bollinger or BollingerBandsParameters()
        self.custom_params = custom_params or {}

    def to_dict(self) -> dict:
        """Convert all parameters to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "rsi": self.rsi.to_dict(),
            "macd": self.macd.to_dict(),
            "bollinger": self.bollinger.to_dict(),
            "custom": self.custom_params,
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for optimization."""
        flat = {
            "rsi_overbought": self.rsi.overbought,
            "rsi_oversold": self.rsi.oversold,
            "rsi_period": self.rsi.period,
            "macd_fast": self.macd.fast_period,
            "macd_slow": self.macd.slow_period,
            "macd_signal": self.macd.signal_period,
            "bb_period": self.bollinger.period,
            "bb_std_dev": self.bollinger.std_dev,
        }
        flat.update(self.custom_params)
        return flat

    @classmethod
    def from_flat_dict(
        cls,
        strategy_name: str,
        params: dict[str, Any],
    ) -> "StrategyParameterSet":
        """Create from flat dictionary."""
        return cls(
            strategy_name=strategy_name,
            rsi=RSIParameters(
                overbought=params.get("rsi_overbought"),
                oversold=params.get("rsi_oversold"),
                period=params.get("rsi_period"),
            ),
            macd=MACDParameters(
                fast_period=params.get("macd_fast"),
                slow_period=params.get("macd_slow"),
                signal_period=params.get("macd_signal"),
            ),
            bollinger=BollingerBandsParameters(
                period=params.get("bb_period"),
                std_dev=params.get("bb_std_dev"),
            ),
        )

    @classmethod
    def get_default_bounds(cls) -> dict[str, ParameterBounds]:
        """Get default parameter bounds for optimization."""
        return {
            "rsi_overbought": RSIParameters.OVERBOUGHT_BOUNDS,
            "rsi_oversold": RSIParameters.OVERSOLD_BOUNDS,
            "rsi_period": RSIParameters.PERIOD_BOUNDS,
            "macd_fast": MACDParameters.FAST_PERIOD_BOUNDS,
            "macd_slow": MACDParameters.SLOW_PERIOD_BOUNDS,
            "macd_signal": MACDParameters.SIGNAL_PERIOD_BOUNDS,
            "bb_period": BollingerBandsParameters.PERIOD_BOUNDS,
            "bb_std_dev": BollingerBandsParameters.STD_DEV_BOUNDS,
        }


# =============================================================================
# PARAMETER SENSITIVITY ANALYSIS EXTENSIONS (P3 Enhancement)
# =============================================================================

@dataclass
class ParameterInteraction:
    """Measures interaction between two parameters."""
    param1: str
    param2: str
    interaction_score: float  # -1 to 1, magnitude indicates strength
    joint_optimal: tuple[float, float] | None
    synergy: bool  # True if parameters work better together

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "param1": self.param1,
            "param2": self.param2,
            "interaction_score": self.interaction_score,
            "joint_optimal": self.joint_optimal,
            "synergy": self.synergy,
        }


class ExtendedSensitivityAnalyzer:
    """
    Extended parameter sensitivity analysis (P3 Enhancement).

    Adds interaction analysis and advanced sensitivity metrics.
    """

    def __init__(
        self,
        strategy_evaluator: Callable[[dict], dict[str, float]],
    ):
        """
        Initialize extended analyzer.

        Args:
            strategy_evaluator: Function that evaluates strategy performance
        """
        self.evaluator = strategy_evaluator
        self._interaction_cache: dict[tuple[str, str], ParameterInteraction] = {}

    def analyze_parameter_interaction(
        self,
        baseline_params: dict[str, Any],
        param1: str,
        param2: str,
        bounds1: ParameterBounds,
        bounds2: ParameterBounds,
        num_samples: int = 5,
        metric: str = "sharpe_ratio",
    ) -> ParameterInteraction:
        """
        Analyze interaction between two parameters.

        Args:
            baseline_params: Baseline parameter values
            param1, param2: Parameters to analyze
            bounds1, bounds2: Bounds for each parameter
            num_samples: Grid samples per parameter
            metric: Performance metric to use

        Returns:
            ParameterInteraction describing the relationship
        """
        # Generate test values
        step1 = (bounds1.max_value - bounds1.min_value) / (num_samples - 1)
        step2 = (bounds2.max_value - bounds2.min_value) / (num_samples - 1)

        values1 = [bounds1.min_value + i * step1 for i in range(num_samples)]
        values2 = [bounds2.min_value + i * step2 for i in range(num_samples)]

        # Evaluate grid
        results = {}
        for v1 in values1:
            for v2 in values2:
                test_params = baseline_params.copy()
                test_params[param1] = v1
                test_params[param2] = v2

                try:
                    metrics = self.evaluator(test_params)
                    results[(v1, v2)] = metrics.get(metric, 0)
                except Exception:
                    results[(v1, v2)] = float('-inf')

        # Find joint optimal
        joint_optimal = max(results.items(), key=lambda x: x[1])[0]

        # Calculate interaction score
        # Compare: f(a,b) vs f(a,baseline_b) + f(baseline_a,b) - f(baseline_a,baseline_b)
        baseline_v1 = baseline_params.get(param1, bounds1.default_value)
        baseline_v2 = baseline_params.get(param2, bounds2.default_value)

        interaction_scores = []
        for v1 in values1:
            for v2 in values2:
                if v1 == baseline_v1 or v2 == baseline_v2:
                    continue

                actual = results.get((v1, v2), 0)
                p1_effect = results.get((v1, baseline_v2), 0) - results.get((baseline_v1, baseline_v2), 0)
                p2_effect = results.get((baseline_v1, v2), 0) - results.get((baseline_v1, baseline_v2), 0)
                expected = results.get((baseline_v1, baseline_v2), 0) + p1_effect + p2_effect

                if expected != 0:
                    interaction = (actual - expected) / abs(expected)
                    interaction_scores.append(interaction)

        avg_interaction = statistics.mean(interaction_scores) if interaction_scores else 0.0

        # Determine synergy
        synergy = avg_interaction > 0.1  # Positive interaction indicates synergy

        result = ParameterInteraction(
            param1=param1,
            param2=param2,
            interaction_score=max(-1, min(1, avg_interaction)),
            joint_optimal=joint_optimal,
            synergy=synergy,
        )

        self._interaction_cache[(param1, param2)] = result
        return result

    def analyze_all_interactions(
        self,
        baseline_params: dict[str, Any],
        parameter_bounds: dict[str, ParameterBounds],
        metric: str = "sharpe_ratio",
    ) -> list[ParameterInteraction]:
        """
        Analyze all pairwise parameter interactions.

        Args:
            baseline_params: Baseline parameter values
            parameter_bounds: Bounds for each parameter
            metric: Performance metric to use

        Returns:
            List of ParameterInteraction for all pairs
        """
        interactions = []
        params = list(parameter_bounds.keys())

        for i, param1 in enumerate(params):
            for param2 in params[i + 1:]:
                interaction = self.analyze_parameter_interaction(
                    baseline_params,
                    param1,
                    param2,
                    parameter_bounds[param1],
                    parameter_bounds[param2],
                    metric=metric,
                )
                interactions.append(interaction)

        return interactions

    def get_parameter_importance_ranking(
        self,
        baseline_params: dict[str, Any],
        parameter_bounds: dict[str, ParameterBounds],
        metric: str = "sharpe_ratio",
    ) -> list[tuple[str, float]]:
        """
        Rank parameters by their impact on performance.

        Args:
            baseline_params: Baseline parameter values
            parameter_bounds: Bounds for each parameter
            metric: Performance metric to use

        Returns:
            List of (parameter_name, importance_score) sorted by importance
        """
        importance = {}
        baseline_metrics = self.evaluator(baseline_params)
        baseline_value = baseline_metrics.get(metric, 0)

        for param_name, bounds in parameter_bounds.items():
            # Test at min and max
            test_params = baseline_params.copy()

            test_params[param_name] = bounds.min_value
            try:
                min_metrics = self.evaluator(test_params)
                min_value = min_metrics.get(metric, 0)
            except Exception:
                min_value = baseline_value

            test_params[param_name] = bounds.max_value
            try:
                max_metrics = self.evaluator(test_params)
                max_value = max_metrics.get(metric, 0)
            except Exception:
                max_value = baseline_value

            # Importance = range of metric values
            importance[param_name] = abs(max_value - min_value)

        # Sort by importance (descending)
        ranked = sorted(importance.items(), key=lambda x: -x[1])
        return ranked


# =============================================================================
# PARAMETER OPTIMIZATION BOUNDS (P3 Enhancement)
# =============================================================================

class OptimizationConstraint(str, Enum):
    """Types of optimization constraints."""
    INDEPENDENT = "independent"  # Parameter optimized independently
    LINKED = "linked"  # Parameter linked to another
    BOUNDED_BY = "bounded_by"  # Upper bound determined by another param
    RATIO = "ratio"  # Maintain ratio with another param


@dataclass
class AdvancedParameterBounds:
    """
    Advanced parameter bounds with constraints (P3 Enhancement).

    Extends ParameterBounds with inter-parameter constraints.
    """
    name: str
    min_value: float
    max_value: float
    default_value: float
    step: float = 1.0
    description: str = ""
    constraint_type: OptimizationConstraint = OptimizationConstraint.INDEPENDENT
    constraint_reference: str | None = None  # Reference parameter for constraint
    constraint_factor: float = 1.0  # Factor for ratio/bounded constraints

    def __post_init__(self) -> None:
        """Validate bounds configuration."""
        if self.min_value > self.max_value:
            raise ValueError(f"min_value ({self.min_value}) > max_value ({self.max_value})")
        if not self.min_value <= self.default_value <= self.max_value:
            raise ValueError(
                f"default_value ({self.default_value}) not in "
                f"[{self.min_value}, {self.max_value}]"
            )

    def get_effective_bounds(
        self,
        reference_value: float | None = None,
    ) -> tuple[float, float]:
        """
        Get effective bounds considering constraints.

        Args:
            reference_value: Value of the reference parameter if constrained

        Returns:
            Tuple of (effective_min, effective_max)
        """
        if reference_value is None or self.constraint_type == OptimizationConstraint.INDEPENDENT:
            return (self.min_value, self.max_value)

        if self.constraint_type == OptimizationConstraint.BOUNDED_BY:
            # Max is bounded by reference * factor
            effective_max = min(self.max_value, reference_value * self.constraint_factor)
            return (self.min_value, effective_max)

        if self.constraint_type == OptimizationConstraint.RATIO:
            # Value must be reference * factor (within tolerance)
            target = reference_value * self.constraint_factor
            tolerance = (self.max_value - self.min_value) * 0.1
            return (max(self.min_value, target - tolerance),
                    min(self.max_value, target + tolerance))

        return (self.min_value, self.max_value)

    def to_parameter_bounds(self) -> ParameterBounds:
        """Convert to basic ParameterBounds."""
        return ParameterBounds(
            min_value=self.min_value,
            max_value=self.max_value,
            default_value=self.default_value,
            step=self.step,
            description=self.description,
        )


class ConstrainedOptimizer:
    """
    Parameter optimizer with constraint support (P3 Enhancement).

    Handles inter-parameter constraints during optimization.
    """

    def __init__(
        self,
        strategy_evaluator: Callable[[dict], dict[str, float]],
        bounds: dict[str, AdvancedParameterBounds],
    ):
        """
        Initialize constrained optimizer.

        Args:
            strategy_evaluator: Function that evaluates strategy performance
            bounds: Advanced bounds for each parameter
        """
        self.evaluator = strategy_evaluator
        self.bounds = bounds
        self._dependency_order = self._compute_dependency_order()

    def _compute_dependency_order(self) -> list[str]:
        """Compute order to optimize parameters based on dependencies."""
        # Independent parameters first, then dependent ones
        independent = []
        dependent = []

        for name, bound in self.bounds.items():
            if bound.constraint_type == OptimizationConstraint.INDEPENDENT:
                independent.append(name)
            else:
                dependent.append(name)

        return independent + dependent

    def optimize(
        self,
        baseline_params: dict[str, Any],
        objective_metric: str = "sharpe_ratio",
        num_iterations: int = 100,
    ) -> dict[str, Any]:
        """
        Find optimal parameters respecting constraints.

        Args:
            baseline_params: Starting parameter values
            objective_metric: Metric to optimize
            num_iterations: Max iterations

        Returns:
            Optimal parameter set
        """
        current_params = baseline_params.copy()
        best_params = current_params.copy()
        best_metric = float('-inf')

        for _ in range(num_iterations):
            # Optimize each parameter in dependency order
            for param_name in self._dependency_order:
                bound = self.bounds.get(param_name)
                if not bound:
                    continue

                # Get reference value for constraints
                ref_value = None
                if bound.constraint_reference:
                    ref_value = current_params.get(bound.constraint_reference)

                # Get effective bounds
                min_val, max_val = bound.get_effective_bounds(ref_value)

                # Test a few values within bounds
                test_values = [
                    min_val,
                    (min_val + max_val) / 2,
                    max_val,
                ]

                for test_val in test_values:
                    test_params = current_params.copy()
                    test_params[param_name] = test_val

                    try:
                        metrics = self.evaluator(test_params)
                        metric_value = metrics.get(objective_metric, float('-inf'))

                        if metric_value > best_metric:
                            best_metric = metric_value
                            best_params = test_params.copy()
                            current_params = test_params.copy()
                    except Exception:
                        continue

        return best_params


# =============================================================================
# PARAMETER CHANGE HISTORY (P3 Enhancement)
# =============================================================================

@dataclass
class ParameterChange:
    """Record of a parameter change."""
    parameter_name: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    reason: str
    changed_by: str  # User or system component
    performance_before: dict[str, float] | None = None
    performance_after: dict[str, float] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "parameter_name": self.parameter_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "changed_by": self.changed_by,
            "performance_before": self.performance_before,
            "performance_after": self.performance_after,
        }


class ParameterChangeHistory:
    """
    Track parameter change history (P3 Enhancement).

    Maintains audit trail of all parameter changes for analysis and rollback.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize change history.

        Args:
            max_history: Maximum number of changes to store
        """
        self.max_history = max_history
        self._history: deque = deque(maxlen=max_history)
        self._current_params: dict[str, Any] = {}

    def set_initial_params(self, params: dict[str, Any]) -> None:
        """Set initial parameter values."""
        self._current_params = params.copy()

    def record_change(
        self,
        parameter_name: str,
        new_value: Any,
        reason: str,
        changed_by: str = "system",
        performance_before: dict[str, float] | None = None,
        performance_after: dict[str, float] | None = None,
    ) -> ParameterChange:
        """
        Record a parameter change.

        Args:
            parameter_name: Name of changed parameter
            new_value: New parameter value
            reason: Reason for change
            changed_by: Who/what made the change
            performance_before: Performance metrics before change
            performance_after: Performance metrics after change

        Returns:
            ParameterChange record
        """
        old_value = self._current_params.get(parameter_name)

        change = ParameterChange(
            parameter_name=parameter_name,
            old_value=old_value,
            new_value=new_value,
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            changed_by=changed_by,
            performance_before=performance_before,
            performance_after=performance_after,
        )

        self._history.append(change)
        self._current_params[parameter_name] = new_value

        logger.info(
            f"Parameter change recorded: {parameter_name} "
            f"{old_value} -> {new_value} ({reason})"
        )

        return change

    def get_history(
        self,
        parameter_name: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[ParameterChange]:
        """
        Get parameter change history.

        Args:
            parameter_name: Filter by parameter (None for all)
            since: Only changes after this time
            limit: Maximum number of changes to return

        Returns:
            List of ParameterChange records
        """
        history = list(self._history)

        if parameter_name:
            history = [c for c in history if c.parameter_name == parameter_name]

        if since:
            history = [c for c in history if c.timestamp >= since]

        if limit:
            history = history[-limit:]

        return history

    def get_rollback_point(self, parameter_name: str, n_changes_back: int = 1) -> Any:
        """
        Get value from n changes ago for potential rollback.

        Args:
            parameter_name: Parameter to check
            n_changes_back: How many changes to look back

        Returns:
            Previous value or None if not found
        """
        changes = self.get_history(parameter_name=parameter_name)

        if len(changes) >= n_changes_back:
            return changes[-(n_changes_back + 1)].old_value if len(changes) > n_changes_back else changes[0].old_value

        return None

    def rollback(
        self,
        parameter_name: str,
        n_changes_back: int = 1,
        reason: str = "manual_rollback",
    ) -> ParameterChange | None:
        """
        Roll back a parameter to a previous value.

        Args:
            parameter_name: Parameter to roll back
            n_changes_back: How many changes to undo
            reason: Reason for rollback

        Returns:
            New ParameterChange record or None if rollback not possible
        """
        rollback_value = self.get_rollback_point(parameter_name, n_changes_back)

        if rollback_value is not None:
            return self.record_change(
                parameter_name=parameter_name,
                new_value=rollback_value,
                reason=f"{reason} (n={n_changes_back})",
                changed_by="rollback_system",
            )

        return None

    def analyze_change_impact(
        self,
        parameter_name: str,
    ) -> dict:
        """
        Analyze the impact of changes to a parameter.

        Args:
            parameter_name: Parameter to analyze

        Returns:
            Dict with impact analysis
        """
        changes = self.get_history(parameter_name=parameter_name)

        if not changes:
            return {"parameter": parameter_name, "total_changes": 0}

        # Calculate statistics
        performance_deltas = []
        for change in changes:
            if change.performance_before and change.performance_after:
                for metric in change.performance_before:
                    if metric in change.performance_after:
                        before = change.performance_before[metric]
                        after = change.performance_after[metric]
                        if before != 0:
                            delta = (after - before) / abs(before) * 100
                            performance_deltas.append({
                                "metric": metric,
                                "delta_pct": delta,
                                "timestamp": change.timestamp,
                            })

        return {
            "parameter": parameter_name,
            "total_changes": len(changes),
            "first_change": changes[0].timestamp.isoformat() if changes else None,
            "last_change": changes[-1].timestamp.isoformat() if changes else None,
            "current_value": self._current_params.get(parameter_name),
            "performance_impacts": performance_deltas[:10],  # Last 10 impacts
        }

    def export_history(self) -> list[dict]:
        """Export complete history as list of dicts."""
        return [c.to_dict() for c in self._history]

    def get_change_frequency(
        self,
        parameter_name: str | None = None,
        days: int = 30,
    ) -> dict:
        """
        Get change frequency statistics.

        Args:
            parameter_name: Filter by parameter
            days: Lookback period in days

        Returns:
            Dict with frequency statistics
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        changes = self.get_history(parameter_name=parameter_name, since=cutoff)

        if not changes:
            return {
                "parameter": parameter_name or "all",
                "days": days,
                "total_changes": 0,
                "changes_per_day": 0.0,
            }

        # Group by day
        by_day: dict[str, int] = defaultdict(int)
        for change in changes:
            day_key = change.timestamp.date().isoformat()
            by_day[day_key] += 1

        return {
            "parameter": parameter_name or "all",
            "days": days,
            "total_changes": len(changes),
            "changes_per_day": len(changes) / days,
            "days_with_changes": len(by_day),
            "max_changes_per_day": max(by_day.values()) if by_day else 0,
        }

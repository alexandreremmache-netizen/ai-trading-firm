"""
Yield Curve Analysis
====================

Comprehensive yield curve analysis for macro trading signals.

Features:
- Yield curve shape detection (normal, flat, inverted, steep)
- 2s10s spread calculation and monitoring
- Real rate computation
- Recession probability estimation
- Curve steepness metrics

The yield curve is a powerful leading indicator:
- Inverted curves (short > long) historically precede recessions
- Steep curves indicate expected growth and rising rates
- Flat curves suggest uncertainty or policy transition

References:
- Estrella & Mishkin (1996) "The Yield Curve as a Predictor of US Recessions"
- Adrian et al. (2019) "Vulnerable Growth" - NY Fed model
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# NumPy for numerical operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

try:
    from scipy import stats as scipy_stats
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_stats = None  # type: ignore
    interp1d = None  # type: ignore


logger = logging.getLogger(__name__)


class YieldCurveState(str, Enum):
    """
    Yield curve shape classification.

    Based on term spread and curve characteristics.
    """
    NORMAL = "normal"       # Positive slope, 2s10s > 50bps
    FLAT = "flat"           # Near-zero slope, |2s10s| < 50bps
    INVERTED = "inverted"   # Negative slope, 2s10s < -10bps
    STEEP = "steep"         # Very positive slope, 2s10s > 150bps
    KINKED = "kinked"       # Non-monotonic curve (belly inversion)


@dataclass
class YieldCurvePoint:
    """
    A single point on the yield curve.

    Represents the yield for a specific tenor.
    """
    tenor: str              # e.g., "3M", "2Y", "10Y", "30Y"
    tenor_years: float      # Tenor in years (e.g., 0.25, 2.0, 10.0)
    yield_pct: float        # Yield in percentage (e.g., 4.5 for 4.5%)

    def __post_init__(self):
        """Validate inputs."""
        if self.yield_pct < -5 or self.yield_pct > 30:
            logger.warning(f"Unusual yield value: {self.yield_pct}% for {self.tenor}")

    @classmethod
    def from_tenor(cls, tenor: str, yield_pct: float) -> "YieldCurvePoint":
        """
        Create point from tenor string.

        Supports formats: 3M, 6M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y
        """
        tenor_map = {
            "1M": 1/12, "3M": 0.25, "6M": 0.5,
            "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0,
            "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
        }
        tenor_years = tenor_map.get(tenor.upper(), 1.0)
        return cls(tenor=tenor, tenor_years=tenor_years, yield_pct=yield_pct)


@dataclass
class YieldCurveAnalysisResult:
    """Result of yield curve analysis."""
    state: YieldCurveState
    spread_2s10s_bps: float         # 2s10s spread in basis points
    spread_3m10y_bps: float         # 3m10y spread (Fed's preferred)
    curve_steepness: float          # Overall slope metric
    real_rate_10y: float | None     # 10Y real rate (if inflation available)
    recession_probability: float    # Model-based recession probability
    front_end_rate: float           # Short-term rate (3M or 2Y)
    long_end_rate: float            # Long-term rate (10Y)
    curve_concavity: float          # Butterfly/concavity measure
    is_warning: bool                # True if curve signals caution
    warning_message: str | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "spread_2s10s_bps": self.spread_2s10s_bps,
            "spread_3m10y_bps": self.spread_3m10y_bps,
            "curve_steepness": self.curve_steepness,
            "real_rate_10y": self.real_rate_10y,
            "recession_probability": self.recession_probability,
            "front_end_rate": self.front_end_rate,
            "long_end_rate": self.long_end_rate,
            "curve_concavity": self.curve_concavity,
            "is_warning": self.is_warning,
            "warning_message": self.warning_message,
            "timestamp": self.timestamp.isoformat(),
        }


def calculate_2s10s_spread(y2: float, y10: float) -> float:
    """
    Calculate 2-year / 10-year spread in basis points.

    The 2s10s spread is the most commonly watched yield curve metric.

    Args:
        y2: 2-year Treasury yield (percentage, e.g., 4.5)
        y10: 10-year Treasury yield (percentage, e.g., 4.2)

    Returns:
        Spread in basis points (negative = inverted)
    """
    spread_pct = y10 - y2
    spread_bps = spread_pct * 100  # Convert to basis points

    return spread_bps


def calculate_3m10y_spread(y3m: float, y10: float) -> float:
    """
    Calculate 3-month / 10-year spread in basis points.

    This is the Federal Reserve's preferred recession indicator.
    Research shows 3m10y has better predictive power than 2s10s.

    Args:
        y3m: 3-month Treasury yield (percentage)
        y10: 10-year Treasury yield (percentage)

    Returns:
        Spread in basis points
    """
    spread_pct = y10 - y3m
    spread_bps = spread_pct * 100

    return spread_bps


def calculate_real_rate(
    nominal_rate: float,
    inflation_expectation: float,
) -> float:
    """
    Calculate real interest rate using Fisher equation.

    Real Rate = Nominal Rate - Inflation Expectation

    For more precision, use:
    Real Rate = (1 + Nominal) / (1 + Inflation) - 1

    Args:
        nominal_rate: Nominal interest rate (percentage)
        inflation_expectation: Expected inflation (percentage)

    Returns:
        Real interest rate (percentage)
    """
    # Simple approximation (accurate for low rates)
    real_rate = nominal_rate - inflation_expectation

    # More precise calculation
    # real_rate = ((1 + nominal_rate/100) / (1 + inflation_expectation/100) - 1) * 100

    return real_rate


def detect_curve_state(curve_points: list[YieldCurvePoint]) -> YieldCurveState:
    """
    Detect the shape/state of the yield curve.

    Args:
        curve_points: List of YieldCurvePoint objects (sorted by tenor)

    Returns:
        YieldCurveState classification
    """
    if len(curve_points) < 2:
        return YieldCurveState.FLAT

    # Sort by tenor
    sorted_points = sorted(curve_points, key=lambda p: p.tenor_years)

    # Extract tenors and yields
    tenors = [p.tenor_years for p in sorted_points]
    yields = [p.yield_pct for p in sorted_points]

    # Find 2Y and 10Y yields (or closest proxies)
    y2 = _interpolate_yield(tenors, yields, 2.0)
    y10 = _interpolate_yield(tenors, yields, 10.0)

    spread_2s10s = (y10 - y2) * 100  # bps

    # Check for inversions
    inversions = sum(1 for i in range(len(yields) - 1) if yields[i] > yields[i + 1])

    # Check for kinks (non-monotonic)
    if inversions > 0 and inversions < len(yields) - 1:
        # Partial inversion - could be kinked
        if abs(spread_2s10s) < 50:
            return YieldCurveState.KINKED

    # Classify based on 2s10s spread
    if spread_2s10s < -10:
        return YieldCurveState.INVERTED
    elif spread_2s10s < 50:
        return YieldCurveState.FLAT
    elif spread_2s10s > 150:
        return YieldCurveState.STEEP
    else:
        return YieldCurveState.NORMAL


def _interpolate_yield(
    tenors: list[float],
    yields: list[float],
    target_tenor: float,
) -> float:
    """
    Interpolate yield for a specific tenor.

    Uses linear interpolation; cubic spline could be used for more accuracy.
    """
    if len(tenors) < 2:
        return yields[0] if yields else 0.0

    # Check if target is outside range
    if target_tenor <= tenors[0]:
        return yields[0]
    if target_tenor >= tenors[-1]:
        return yields[-1]

    # Linear interpolation
    for i in range(len(tenors) - 1):
        if tenors[i] <= target_tenor <= tenors[i + 1]:
            t1, t2 = tenors[i], tenors[i + 1]
            y1, y2 = yields[i], yields[i + 1]
            weight = (target_tenor - t1) / (t2 - t1)
            return y1 + weight * (y2 - y1)

    return yields[-1]


class YieldCurveAnalyzer:
    """
    Comprehensive yield curve analyzer for trading signals.

    Provides yield curve analysis, recession probability estimation,
    and trading signal generation.
    """

    def __init__(
        self,
        recession_model: str = "probit",
        inflation_expectation: float | None = None,
    ):
        """
        Initialize yield curve analyzer.

        Args:
            recession_model: Model for recession probability ("probit" or "logistic")
            inflation_expectation: Expected inflation for real rate calculation
        """
        self.recession_model = recession_model
        self.inflation_expectation = inflation_expectation

        # Historical spreads for tracking
        self._spread_history: list[tuple[datetime, float]] = []

        # Key tenors
        self._current_curve: dict[str, float] = {}

    def set_inflation_expectation(self, inflation_pct: float) -> None:
        """
        Set inflation expectation for real rate calculations.

        Args:
            inflation_pct: Expected inflation in percentage
        """
        self.inflation_expectation = inflation_pct

    def update_yield(self, tenor: str, yield_pct: float) -> None:
        """
        Update yield for a specific tenor.

        Args:
            tenor: Tenor string (e.g., "2Y", "10Y")
            yield_pct: Yield in percentage
        """
        self._current_curve[tenor.upper()] = yield_pct

    def update_curve(self, curve_data: dict[str, float]) -> None:
        """
        Update entire curve at once.

        Args:
            curve_data: Dictionary of tenor -> yield mappings
        """
        for tenor, yield_pct in curve_data.items():
            self._current_curve[tenor.upper()] = yield_pct

    def calculate_curve_steepness(self) -> float:
        """
        Calculate overall curve steepness metric.

        Uses linear regression slope of yield vs log(tenor).

        Returns:
            Steepness metric (higher = steeper)
        """
        if len(self._current_curve) < 3:
            return 0.0

        points = [
            YieldCurvePoint.from_tenor(tenor, yld)
            for tenor, yld in self._current_curve.items()
        ]
        points = sorted(points, key=lambda p: p.tenor_years)

        if not HAS_NUMPY:
            # Simple calculation without numpy
            return (points[-1].yield_pct - points[0].yield_pct) / (
                math.log(points[-1].tenor_years) - math.log(points[0].tenor_years + 0.01)
            )

        # Log-linear regression
        log_tenors = np.log(np.array([p.tenor_years for p in points]) + 0.01)
        yields = np.array([p.yield_pct for p in points])

        # Simple linear regression
        n = len(log_tenors)
        sum_x = np.sum(log_tenors)
        sum_y = np.sum(yields)
        sum_xy = np.sum(log_tenors * yields)
        sum_xx = np.sum(log_tenors ** 2)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2 + 1e-10)

        return float(slope)

    def get_recession_probability(
        self,
        spread_bps: float | None = None,
        horizon_months: int = 12,
    ) -> float:
        """
        Estimate recession probability from yield curve.

        Based on NY Fed model using probit regression.
        The model relates the 3m10y spread to recession probability.

        Args:
            spread_bps: Spread in basis points (uses 3m10y if not provided)
            horizon_months: Forecast horizon in months (12 is standard)

        Returns:
            Probability of recession (0 to 1)
        """
        if spread_bps is None:
            # Calculate from current curve
            y3m = self._current_curve.get("3M", self._current_curve.get("6M", 5.0))
            y10 = self._current_curve.get("10Y", 4.0)
            spread_bps = calculate_3m10y_spread(y3m, y10)

        spread_pct = spread_bps / 100  # Convert to percentage

        # NY Fed probit model coefficients (approximate)
        # Based on historical regression of spread on recession probability
        # P(Recession) = F(-0.5333 - 0.6330 * spread)
        # where F is the standard normal CDF

        if self.recession_model == "probit":
            # Probit model (NY Fed approach)
            z = -0.5333 - 0.6330 * spread_pct

            if HAS_SCIPY:
                prob = float(scipy_stats.norm.cdf(z))
            else:
                # Approximation of normal CDF
                prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        else:
            # Logistic model (simpler alternative)
            z = -1.0 - 0.5 * spread_pct
            prob = 1 / (1 + math.exp(-z))

        # Clamp to [0, 1]
        prob = max(0.0, min(1.0, prob))

        return prob

    def calculate_concavity(self) -> float:
        """
        Calculate butterfly/concavity of the curve.

        Measures how much the belly (5Y) deviates from
        a straight line between 2Y and 10Y.

        Positive = belly rich (yields lower than interpolation)
        Negative = belly cheap (yields higher than interpolation)

        Returns:
            Concavity in basis points
        """
        y2 = self._current_curve.get("2Y", None)
        y5 = self._current_curve.get("5Y", None)
        y10 = self._current_curve.get("10Y", None)

        if y2 is None or y5 is None or y10 is None:
            return 0.0

        # Interpolated 5Y from 2Y-10Y line
        interpolated_5y = y2 + (y10 - y2) * (5 - 2) / (10 - 2)

        # Concavity = interpolated - actual
        # Positive means actual is below interpolation (belly rich)
        concavity_pct = interpolated_5y - y5
        concavity_bps = concavity_pct * 100

        return concavity_bps

    def analyze(
        self,
        curve_data: dict[str, float] | None = None,
    ) -> YieldCurveAnalysisResult:
        """
        Perform full yield curve analysis.

        Args:
            curve_data: Optional curve data override

        Returns:
            YieldCurveAnalysisResult with complete analysis
        """
        if curve_data:
            self.update_curve(curve_data)

        # Get key rates
        y3m = self._current_curve.get("3M", self._current_curve.get("6M"))
        y2 = self._current_curve.get("2Y")
        y10 = self._current_curve.get("10Y")

        # Calculate spreads
        spread_2s10s = 0.0
        spread_3m10y = 0.0

        if y2 is not None and y10 is not None:
            spread_2s10s = calculate_2s10s_spread(y2, y10)

        if y3m is not None and y10 is not None:
            spread_3m10y = calculate_3m10y_spread(y3m, y10)

        # Detect curve state
        curve_points = [
            YieldCurvePoint.from_tenor(tenor, yld)
            for tenor, yld in self._current_curve.items()
        ]
        state = detect_curve_state(curve_points)

        # Calculate steepness
        steepness = self.calculate_curve_steepness()

        # Calculate real rate
        real_rate = None
        if y10 is not None and self.inflation_expectation is not None:
            real_rate = calculate_real_rate(y10, self.inflation_expectation)

        # Calculate recession probability
        recession_prob = self.get_recession_probability(spread_3m10y)

        # Calculate concavity
        concavity = self.calculate_concavity()

        # Determine warnings
        is_warning = False
        warning_message = None

        if state == YieldCurveState.INVERTED:
            is_warning = True
            warning_message = f"Yield curve inverted (2s10s: {spread_2s10s:.0f}bps)"
        elif recession_prob > 0.5:
            is_warning = True
            warning_message = f"Elevated recession probability: {recession_prob:.0%}"
        elif state == YieldCurveState.FLAT:
            is_warning = True
            warning_message = "Flat yield curve - policy transition risk"

        # Track spread history
        self._spread_history.append((datetime.now(timezone.utc), spread_2s10s))
        if len(self._spread_history) > 252:  # Keep ~1 year
            self._spread_history = self._spread_history[-252:]

        return YieldCurveAnalysisResult(
            state=state,
            spread_2s10s_bps=spread_2s10s,
            spread_3m10y_bps=spread_3m10y,
            curve_steepness=steepness,
            real_rate_10y=real_rate,
            recession_probability=recession_prob,
            front_end_rate=y2 or y3m or 0.0,
            long_end_rate=y10 or 0.0,
            curve_concavity=concavity,
            is_warning=is_warning,
            warning_message=warning_message,
        )

    def get_trading_signal(
        self,
        curve_data: dict[str, float] | None = None,
    ) -> tuple[float, str]:
        """
        Generate trading signal from yield curve analysis.

        Args:
            curve_data: Optional curve data

        Returns:
            (signal_strength, rationale) tuple
            signal_strength: -1 to +1 (positive = risk-on)
        """
        result = self.analyze(curve_data)

        # Signal mapping based on curve state
        state_signals = {
            YieldCurveState.STEEP: (0.5, "Steep curve suggests growth expectations"),
            YieldCurveState.NORMAL: (0.2, "Normal curve - standard risk appetite"),
            YieldCurveState.FLAT: (-0.2, "Flat curve - policy uncertainty"),
            YieldCurveState.INVERTED: (-0.7, "Inverted curve - recession warning"),
            YieldCurveState.KINKED: (-0.3, "Kinked curve - market stress"),
        }

        signal, rationale = state_signals.get(
            result.state, (0.0, "Unknown curve state")
        )

        # Adjust for recession probability
        if result.recession_probability > 0.6:
            signal = min(signal, -0.5)
            rationale += f" (recession prob: {result.recession_probability:.0%})"

        # Add spread info
        rationale += f" [2s10s: {result.spread_2s10s_bps:.0f}bps]"

        return signal, rationale

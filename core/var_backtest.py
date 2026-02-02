"""
VaR Backtesting Module
======================

Addresses issue #R28: Historical VaR backtest missing.

Features:
- VaR model validation through backtesting
- Kupiec POF test for exception analysis
- Christoffersen independence test
- Traffic light framework (Basel)
- Backtesting reports and visualization data
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from collections import deque

logger = logging.getLogger(__name__)


class VaRBacktestZone(str, Enum):
    """Basel traffic light zones for VaR backtesting."""
    GREEN = "green"  # Model acceptable
    YELLOW = "yellow"  # Model may need review
    RED = "red"  # Model needs remediation


@dataclass
class VaRException:
    """A single VaR exception (loss exceeded VaR)."""
    date: datetime
    var_estimate: float  # VaR estimate (positive)
    actual_loss: float  # Actual P&L (negative for loss)
    excess: float  # Amount by which loss exceeded VaR

    @property
    def exception_severity(self) -> float:
        """Severity as ratio of loss to VaR."""
        if self.var_estimate > 0:
            return abs(self.actual_loss) / self.var_estimate
        return 0.0


@dataclass
class KupiecTestResult:
    """Result of Kupiec POF (Proportion of Failures) test."""
    test_statistic: float
    p_value: float
    is_rejected: bool  # True if model rejected at significance level
    observed_exceptions: int
    expected_exceptions: float
    exception_rate: float
    confidence_level: float
    significance_level: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "is_rejected": self.is_rejected,
            "observed_exceptions": self.observed_exceptions,
            "expected_exceptions": self.expected_exceptions,
            "exception_rate_pct": self.exception_rate * 100,
            "confidence_level": self.confidence_level,
            "significance_level": self.significance_level,
        }


@dataclass
class ChristoffersenTestResult:
    """Result of Christoffersen independence test."""
    test_statistic: float
    p_value: float
    is_rejected: bool
    cluster_ratio: float  # Ratio of clustered to non-clustered exceptions

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "is_rejected": self.is_rejected,
            "cluster_ratio": self.cluster_ratio,
        }


@dataclass
class VaRBacktestResult:
    """Complete VaR backtest result."""
    # Basic stats
    start_date: datetime
    end_date: datetime
    total_observations: int
    var_confidence_level: float

    # Exception analysis
    total_exceptions: int
    exception_rate: float
    expected_exceptions: float
    exceptions: list[VaRException]

    # Statistical tests
    kupiec_test: KupiecTestResult
    christoffersen_test: ChristoffersenTestResult | None

    # Basel traffic light
    zone: VaRBacktestZone
    zone_reason: str

    # Additional metrics
    avg_exception_severity: float
    max_exception_severity: float
    avg_var_estimate: float
    avg_actual_return: float

    # Time series for charting
    var_series: list[tuple[datetime, float]] = field(default_factory=list)
    return_series: list[tuple[datetime, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
                "observations": self.total_observations,
            },
            "var_confidence": self.var_confidence_level,
            "exceptions": {
                "total": self.total_exceptions,
                "expected": self.expected_exceptions,
                "rate_pct": self.exception_rate * 100,
                "avg_severity": self.avg_exception_severity,
                "max_severity": self.max_exception_severity,
            },
            "tests": {
                "kupiec": self.kupiec_test.to_dict(),
                "christoffersen": self.christoffersen_test.to_dict() if self.christoffersen_test else None,
            },
            "traffic_light": {
                "zone": self.zone.value,
                "reason": self.zone_reason,
            },
            "model_stats": {
                "avg_var_estimate": self.avg_var_estimate,
                "avg_actual_return": self.avg_actual_return,
            },
        }


class VaRBacktester:
    """
    VaR model backtesting framework (#R28).

    Validates VaR models using historical exceptions analysis
    and statistical tests.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        significance_level: float = 0.05,
    ):
        """
        Initialize backtester.

        Args:
            confidence_level: VaR confidence level (e.g., 0.95 for 95%)
            significance_level: Statistical test significance level
        """
        self.confidence_level = confidence_level
        self.significance_level = significance_level
        self._var_estimates: list[tuple[datetime, float]] = []
        self._actual_returns: list[tuple[datetime, float]] = []

    def add_observation(
        self,
        date: datetime,
        var_estimate: float,
        actual_return: float,
    ) -> None:
        """
        Add a single observation for backtesting.

        Args:
            date: Observation date
            var_estimate: VaR estimate (as positive number)
            actual_return: Actual realized return (negative for loss)
        """
        self._var_estimates.append((date, var_estimate))
        self._actual_returns.append((date, actual_return))

    def add_observations_bulk(
        self,
        var_estimates: list[tuple[datetime, float]],
        actual_returns: list[tuple[datetime, float]],
    ) -> None:
        """Add multiple observations at once."""
        self._var_estimates.extend(var_estimates)
        self._actual_returns.extend(actual_returns)

    def run_backtest(self) -> VaRBacktestResult:
        """
        Run VaR backtest on accumulated observations.

        Returns:
            VaRBacktestResult with comprehensive analysis
        """
        if len(self._var_estimates) < 30:
            raise ValueError("Need at least 30 observations for backtest")

        if len(self._var_estimates) != len(self._actual_returns):
            raise ValueError("VaR estimates and returns must have same length")

        # Sort by date
        var_sorted = sorted(self._var_estimates, key=lambda x: x[0])
        ret_sorted = sorted(self._actual_returns, key=lambda x: x[0])

        n = len(var_sorted)
        start_date = var_sorted[0][0]
        end_date = var_sorted[-1][0]

        # Identify exceptions
        exceptions = []
        exception_indicators = []

        for (date, var), (_, ret) in zip(var_sorted, ret_sorted):
            # Exception: actual loss exceeds VaR
            # Assuming var is positive and ret is negative for loss
            is_exception = ret < -var

            exception_indicators.append(1 if is_exception else 0)

            if is_exception:
                exceptions.append(VaRException(
                    date=date,
                    var_estimate=var,
                    actual_loss=ret,
                    excess=abs(ret) - var,
                ))

        total_exceptions = len(exceptions)
        exception_rate = total_exceptions / n
        expected_exceptions = n * (1 - self.confidence_level)

        # Kupiec POF test
        kupiec = self._kupiec_test(n, total_exceptions)

        # Christoffersen independence test
        christoffersen = self._christoffersen_test(exception_indicators) if total_exceptions > 1 else None

        # Basel traffic light zone
        zone, zone_reason = self._determine_zone(n, total_exceptions)

        # Calculate additional metrics
        var_values = [v for _, v in var_sorted]
        ret_values = [r for _, r in ret_sorted]

        avg_exception_severity = 0.0
        max_exception_severity = 0.0
        if exceptions:
            severities = [e.exception_severity for e in exceptions]
            avg_exception_severity = statistics.mean(severities)
            max_exception_severity = max(severities)

        return VaRBacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_observations=n,
            var_confidence_level=self.confidence_level,
            total_exceptions=total_exceptions,
            exception_rate=exception_rate,
            expected_exceptions=expected_exceptions,
            exceptions=exceptions,
            kupiec_test=kupiec,
            christoffersen_test=christoffersen,
            zone=zone,
            zone_reason=zone_reason,
            avg_exception_severity=avg_exception_severity,
            max_exception_severity=max_exception_severity,
            avg_var_estimate=statistics.mean(var_values),
            avg_actual_return=statistics.mean(ret_values),
            var_series=var_sorted,
            return_series=ret_sorted,
        )

    def _kupiec_test(self, n: int, x: int) -> KupiecTestResult:
        """
        Kupiec Proportion of Failures test.

        Tests if the number of exceptions is consistent with
        the VaR confidence level.
        """
        p = 1 - self.confidence_level  # Expected failure rate
        p_hat = x / n if n > 0 else 0  # Observed failure rate

        # Likelihood ratio test statistic
        if x == 0:
            lr_stat = -2 * (n * math.log(1 - p) - n * math.log(1 - p_hat) if p_hat < 1 else 0)
        elif x == n:
            lr_stat = -2 * (n * math.log(p) - n * math.log(p_hat) if p_hat > 0 else 0)
        else:
            try:
                lr_stat = -2 * (
                    math.log((1 - p) ** (n - x) * p ** x)
                    - math.log((1 - p_hat) ** (n - x) * p_hat ** x)
                )
            except (ValueError, ZeroDivisionError):
                lr_stat = 0

        # p-value from chi-squared distribution with 1 df
        # Using simple approximation
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lr_stat, 1)

        return KupiecTestResult(
            test_statistic=lr_stat,
            p_value=p_value,
            is_rejected=p_value < self.significance_level,
            observed_exceptions=x,
            expected_exceptions=n * p,
            exception_rate=p_hat,
            confidence_level=self.confidence_level,
            significance_level=self.significance_level,
        )

    def _christoffersen_test(
        self,
        indicators: list[int],
    ) -> ChristoffersenTestResult:
        """
        Christoffersen independence test.

        Tests if exceptions are independently distributed
        (not clustered).
        """
        n = len(indicators)

        # Count transitions
        n00 = n01 = n10 = n11 = 0

        for i in range(n - 1):
            if indicators[i] == 0 and indicators[i + 1] == 0:
                n00 += 1
            elif indicators[i] == 0 and indicators[i + 1] == 1:
                n01 += 1
            elif indicators[i] == 1 and indicators[i + 1] == 0:
                n10 += 1
            else:
                n11 += 1

        # Transition probabilities
        p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        p = (n01 + n11) / (n - 1) if n > 1 else 0

        # Likelihood ratio statistic
        try:
            if 0 < p < 1 and 0 < p01 < 1 and 0 < p11 < 1:
                lr_stat = -2 * (
                    (n00 + n10) * math.log(1 - p)
                    + (n01 + n11) * math.log(p)
                    - n00 * math.log(1 - p01)
                    - n01 * math.log(p01)
                    - n10 * math.log(1 - p11)
                    - n11 * math.log(p11)
                )
            else:
                lr_stat = 0
        except (ValueError, ZeroDivisionError):
            lr_stat = 0

        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lr_stat, 1)

        # Cluster ratio
        cluster_ratio = n11 / (n01 + n11) if (n01 + n11) > 0 else 0

        return ChristoffersenTestResult(
            test_statistic=lr_stat,
            p_value=p_value,
            is_rejected=p_value < self.significance_level,
            cluster_ratio=cluster_ratio,
        )

    def _determine_zone(
        self,
        n: int,
        exceptions: int,
    ) -> tuple[VaRBacktestZone, str]:
        """
        Determine Basel traffic light zone.

        Based on 250-day window at 99% confidence:
        - Green: 0-4 exceptions
        - Yellow: 5-9 exceptions
        - Red: 10+ exceptions

        Scaled for other windows/confidence levels.
        """
        expected = n * (1 - self.confidence_level)

        # Scale thresholds based on expected exceptions
        # For 95% VaR over 250 days: expected = 12.5
        # For 99% VaR over 250 days: expected = 2.5
        green_threshold = expected * 1.5
        yellow_threshold = expected * 3.0

        if exceptions <= green_threshold:
            return VaRBacktestZone.GREEN, (
                f"Exceptions ({exceptions}) within acceptable range "
                f"(<= {green_threshold:.0f})"
            )
        elif exceptions <= yellow_threshold:
            return VaRBacktestZone.YELLOW, (
                f"Exceptions ({exceptions}) elevated "
                f"({green_threshold:.0f} - {yellow_threshold:.0f}), review model"
            )
        else:
            return VaRBacktestZone.RED, (
                f"Exceptions ({exceptions}) exceed threshold "
                f"(> {yellow_threshold:.0f}), model remediation required"
            )

    def reset(self) -> None:
        """Reset accumulated observations."""
        self._var_estimates = []
        self._actual_returns = []


class RollingVaRBacktester:
    """
    Rolling window VaR backtester for ongoing model validation.

    Maintains a rolling window of observations and periodically
    runs backtests.
    """

    def __init__(
        self,
        window_size: int = 250,
        confidence_level: float = 0.95,
        backtest_frequency: int = 20,  # Days between backtests
    ):
        """
        Initialize rolling backtester.

        Args:
            window_size: Number of observations in rolling window
            confidence_level: VaR confidence level
            backtest_frequency: Days between automatic backtests
        """
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.backtest_frequency = backtest_frequency

        self._var_buffer: deque = deque(maxlen=window_size)
        self._return_buffer: deque = deque(maxlen=window_size)
        self._observations_since_backtest: int = 0
        self._last_result: VaRBacktestResult | None = None

    def add_observation(
        self,
        date: datetime,
        var_estimate: float,
        actual_return: float,
    ) -> VaRBacktestResult | None:
        """
        Add observation and potentially trigger backtest.

        Returns:
            VaRBacktestResult if backtest was triggered, None otherwise
        """
        self._var_buffer.append((date, var_estimate))
        self._return_buffer.append((date, actual_return))
        self._observations_since_backtest += 1

        # Check if we should run backtest
        if (
            len(self._var_buffer) >= 30
            and self._observations_since_backtest >= self.backtest_frequency
        ):
            result = self._run_backtest()
            self._observations_since_backtest = 0
            self._last_result = result
            return result

        return None

    def _run_backtest(self) -> VaRBacktestResult:
        """Run backtest on current window."""
        backtester = VaRBacktester(
            confidence_level=self.confidence_level,
        )
        backtester.add_observations_bulk(
            list(self._var_buffer),
            list(self._return_buffer),
        )
        return backtester.run_backtest()

    def force_backtest(self) -> VaRBacktestResult | None:
        """Force immediate backtest if enough data."""
        if len(self._var_buffer) >= 30:
            result = self._run_backtest()
            self._last_result = result
            return result
        return None

    def get_last_result(self) -> VaRBacktestResult | None:
        """Get most recent backtest result."""
        return self._last_result

    def get_exception_count(self, lookback_days: int | None = None) -> int:
        """Get exception count for recent period."""
        if lookback_days is None:
            lookback_days = len(self._var_buffer)

        count = 0
        for (_, var), (_, ret) in zip(
            list(self._var_buffer)[-lookback_days:],
            list(self._return_buffer)[-lookback_days:],
        ):
            if ret < -var:
                count += 1

        return count

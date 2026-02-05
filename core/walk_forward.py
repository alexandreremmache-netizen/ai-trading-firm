"""
Walk-Forward Validation Framework
=================================

Comprehensive walk-forward analysis for strategy validation and robustness testing.

Walk-forward validation is the gold standard for strategy testing because:
1. It prevents overfitting by separating train/test periods
2. It simulates real trading where you optimize on past data and trade on new data
3. It provides realistic out-of-sample performance estimates

This module implements:
- Rolling and anchored (expanding) window walk-forward
- In-sample vs out-of-sample performance degradation analysis
- Parameter stability metrics
- Robustness scoring

References:
- Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies
- Aronson, D. (2006). Evidence-Based Technical Analysis
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Iterator, Protocol, TypeVar

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward validation.

    Attributes:
        train_period_days: Number of days for training/optimization window.
            Default 252 (1 trading year) is standard in academic literature.
        test_period_days: Number of days for out-of-sample testing.
            Default 63 (1 quarter) provides meaningful statistical sample.
        step_days: Number of days to advance between windows.
            Default 21 (1 month) balances granularity vs computation.
        min_train_samples: Minimum number of samples required for training.
            Prevents optimization on insufficient data.
        anchored: If True, use expanding (anchored) window where train start
            is fixed. If False, use rolling window where both start/end move.
        min_trades_per_fold: Minimum trades required per fold for valid results.
        gap_days: Days to skip between train and test (avoid look-ahead).
    """

    train_period_days: int = 252  # 1 trading year
    test_period_days: int = 63  # 1 quarter
    step_days: int = 21  # 1 month
    min_train_samples: int = 50  # Minimum data points for training
    anchored: bool = False  # False = rolling, True = expanding window
    min_trades_per_fold: int = 5  # Minimum trades per test period
    gap_days: int = 0  # Gap between train end and test start

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.train_period_days <= 0:
            raise ValueError("train_period_days must be positive")
        if self.test_period_days <= 0:
            raise ValueError("test_period_days must be positive")
        if self.step_days <= 0:
            raise ValueError("step_days must be positive")
        if self.step_days > self.test_period_days:
            raise ValueError("step_days should not exceed test_period_days")
        if self.min_train_samples <= 0:
            raise ValueError("min_train_samples must be positive")
        if self.gap_days < 0:
            raise ValueError("gap_days cannot be negative")


# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass
class FoldResult:
    """
    Results from a single walk-forward fold.

    Attributes:
        fold_number: Sequential fold number (1-indexed)
        train_start: Start date of training period
        train_end: End date of training period
        test_start: Start date of test period
        test_end: End date of test period
        train_sharpe: In-sample Sharpe ratio
        test_sharpe: Out-of-sample Sharpe ratio
        train_return: In-sample total return
        test_return: Out-of-sample total return
        train_volatility: In-sample annualized volatility
        test_volatility: Out-of-sample annualized volatility
        train_max_drawdown: In-sample maximum drawdown
        test_max_drawdown: Out-of-sample maximum drawdown
        n_train_trades: Number of trades in training period
        n_test_trades: Number of trades in test period
        optimized_params: Parameters selected during training
        daily_returns: Array of daily returns for test period
        is_valid: Whether fold has sufficient data/trades
    """

    fold_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_return: float = 0.0
    test_return: float = 0.0
    train_volatility: float = 0.0
    test_volatility: float = 0.0
    train_max_drawdown: float = 0.0
    test_max_drawdown: float = 0.0
    n_train_trades: int = 0
    n_test_trades: int = 0
    optimized_params: dict[str, Any] = field(default_factory=dict)
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    is_valid: bool = True
    validation_message: str = ""

    @property
    def degradation(self) -> float:
        """
        Calculate performance degradation from train to test.

        Returns percentage drop in Sharpe ratio from in-sample to out-of-sample.
        Negative values indicate OOS outperformance (rare but possible).
        """
        if abs(self.train_sharpe) < 1e-6:
            return 0.0 if abs(self.test_sharpe) < 1e-6 else -100.0
        return ((self.train_sharpe - self.test_sharpe) / abs(self.train_sharpe)) * 100


@dataclass
class WalkForwardResult:
    """
    Aggregate results from walk-forward validation.

    Attributes:
        config: Configuration used for validation
        fold_results: List of individual fold results
        aggregate_sharpe: Combined Sharpe ratio across all test periods
        aggregate_return: Combined return across all test periods
        aggregate_volatility: Combined volatility across all test periods
        aggregate_max_drawdown: Maximum drawdown across all test periods
        aggregate_returns: Concatenated daily returns from all test periods
        in_sample_vs_oos_degradation: Average performance degradation
        parameter_stability: Metrics on parameter consistency across folds
        robustness_score: Overall robustness score (0-100)
        total_folds: Total number of folds
        valid_folds: Number of valid folds
        start_date: Start of entire validation period
        end_date: End of entire validation period
    """

    config: WalkForwardConfig
    fold_results: list[FoldResult] = field(default_factory=list)
    aggregate_sharpe: float = 0.0
    aggregate_return: float = 0.0
    aggregate_volatility: float = 0.0
    aggregate_max_drawdown: float = 0.0
    aggregate_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    in_sample_vs_oos_degradation: float = 0.0
    parameter_stability: dict[str, float] = field(default_factory=dict)
    robustness_score: float = 0.0
    total_folds: int = 0
    valid_folds: int = 0
    start_date: datetime | None = None
    end_date: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": {
                "train_period_days": self.config.train_period_days,
                "test_period_days": self.config.test_period_days,
                "step_days": self.config.step_days,
                "anchored": self.config.anchored,
            },
            "aggregate_metrics": {
                "sharpe_ratio": self.aggregate_sharpe,
                "total_return_pct": self.aggregate_return * 100,
                "volatility_pct": self.aggregate_volatility * 100,
                "max_drawdown_pct": self.aggregate_max_drawdown * 100,
            },
            "robustness": {
                "score": self.robustness_score,
                "degradation_pct": self.in_sample_vs_oos_degradation,
                "parameter_stability": self.parameter_stability,
            },
            "fold_summary": {
                "total_folds": self.total_folds,
                "valid_folds": self.valid_folds,
                "avg_test_sharpe": np.mean([f.test_sharpe for f in self.fold_results if f.is_valid]) if self.fold_results else 0,
                "sharpe_std": np.std([f.test_sharpe for f in self.fold_results if f.is_valid]) if self.fold_results else 0,
            },
            "period": {
                "start": self.start_date.isoformat() if self.start_date else None,
                "end": self.end_date.isoformat() if self.end_date else None,
            },
        }


# =============================================================================
# STRATEGY PROTOCOL
# =============================================================================


class BacktestableStrategy(Protocol):
    """
    Protocol defining the interface for strategies to be backtestable.

    Strategies must implement this interface to be used with WalkForwardValidator.
    This enables loose coupling - any strategy following this protocol can be tested.
    """

    def fit(self, train_data: dict[str, np.ndarray]) -> None:
        """
        Train/optimize the strategy on training data.

        Args:
            train_data: Dictionary containing at minimum:
                - 'prices': Price series (numpy array)
                - 'dates': Date series (numpy array of datetime)
                May also contain: 'volume', 'high', 'low', 'open'
        """
        ...

    def predict(self, test_data: dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate trading signals on test data.

        Args:
            test_data: Same format as train_data

        Returns:
            Array of signals: 1 (long), -1 (short), 0 (flat)
        """
        ...

    def get_parameters(self) -> dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameter names to values
        """
        ...


# =============================================================================
# ROBUSTNESS METRICS
# =============================================================================


def calculate_oos_degradation(
    in_sample_sharpes: list[float],
    out_of_sample_sharpes: list[float],
) -> float:
    """
    Calculate performance degradation from in-sample to out-of-sample.

    A healthy strategy should show <50% degradation. Higher degradation
    indicates potential overfitting.

    Args:
        in_sample_sharpes: List of in-sample Sharpe ratios per fold
        out_of_sample_sharpes: List of out-of-sample Sharpe ratios per fold

    Returns:
        Average percentage degradation (0-100+, lower is better)
    """
    if not in_sample_sharpes or not out_of_sample_sharpes:
        return 0.0

    if len(in_sample_sharpes) != len(out_of_sample_sharpes):
        raise ValueError("IS and OOS lists must have same length")

    degradations = []
    for is_sharpe, oos_sharpe in zip(in_sample_sharpes, out_of_sample_sharpes):
        if abs(is_sharpe) > 1e-6:
            deg = ((is_sharpe - oos_sharpe) / abs(is_sharpe)) * 100
            degradations.append(deg)
        elif abs(oos_sharpe) < 1e-6:
            degradations.append(0.0)  # Both near zero
        else:
            degradations.append(-100.0)  # IS zero but OOS not - unusual

    return np.mean(degradations) if degradations else 0.0


def calculate_parameter_stability(
    fold_params: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Calculate stability of optimized parameters across folds.

    Stable parameters (low CV) suggest the strategy is not overfitting
    to specific market conditions. High CV indicates parameter sensitivity.

    Args:
        fold_params: List of parameter dictionaries from each fold

    Returns:
        Dictionary mapping parameter name to coefficient of variation (CV)
        Lower CV = more stable parameters
    """
    if not fold_params:
        return {}

    # Collect values for each parameter across folds
    param_values: dict[str, list[float]] = {}
    for params in fold_params:
        for name, value in params.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if name not in param_values:
                    param_values[name] = []
                param_values[name].append(float(value))

    # Calculate CV for each parameter
    stability: dict[str, float] = {}
    for name, values in param_values.items():
        if len(values) >= 2:
            mean_val = np.mean(values)
            std_val = np.std(values)
            if abs(mean_val) > 1e-8:
                cv = (std_val / abs(mean_val)) * 100  # CV as percentage
            else:
                cv = 0.0 if std_val < 1e-8 else 100.0
            stability[name] = cv

    return stability


def calculate_drawdown_consistency(
    fold_drawdowns: list[float],
) -> float:
    """
    Calculate consistency of drawdowns across folds.

    Consistent drawdowns (low standard deviation) indicate predictable risk.
    High variation suggests the strategy behaves very differently across periods.

    Args:
        fold_drawdowns: List of max drawdown values per fold (as decimals, e.g., 0.10)

    Returns:
        Coefficient of variation of drawdowns (lower is more consistent)
    """
    if not fold_drawdowns or len(fold_drawdowns) < 2:
        return 0.0

    mean_dd = np.mean(fold_drawdowns)
    std_dd = np.std(fold_drawdowns)

    if abs(mean_dd) > 1e-8:
        return (std_dd / abs(mean_dd)) * 100
    return 0.0 if std_dd < 1e-8 else 100.0


def calculate_robustness_score(
    result: WalkForwardResult,
) -> float:
    """
    Calculate overall robustness score (0-100).

    Components:
    - OOS Sharpe contribution (40%): Higher OOS Sharpe = better
    - Degradation penalty (30%): Lower degradation = better
    - Consistency bonus (20%): Lower Sharpe variance = better
    - Drawdown consistency (10%): Lower DD variance = better

    Args:
        result: Walk-forward validation result

    Returns:
        Robustness score from 0 (poor) to 100 (excellent)
    """
    if not result.fold_results or result.valid_folds == 0:
        return 0.0

    valid_folds = [f for f in result.fold_results if f.is_valid]
    if not valid_folds:
        return 0.0

    # 1. OOS Sharpe contribution (40% weight)
    oos_sharpes = [f.test_sharpe for f in valid_folds]
    avg_oos_sharpe = np.mean(oos_sharpes)
    # Map Sharpe to 0-100 scale: <0 = 0, 0.5 = 25, 1.0 = 50, 2.0 = 75, 3.0+ = 100
    sharpe_score = min(100, max(0, avg_oos_sharpe * 33.33))
    sharpe_contribution = sharpe_score * 0.40

    # 2. Degradation penalty (30% weight)
    # <20% degradation = full score, >100% degradation = 0
    degradation = result.in_sample_vs_oos_degradation
    if degradation <= 20:
        degradation_score = 100
    elif degradation >= 100:
        degradation_score = 0
    else:
        degradation_score = 100 - (degradation - 20) * (100 / 80)
    degradation_contribution = degradation_score * 0.30

    # 3. Consistency bonus (20% weight)
    sharpe_std = np.std(oos_sharpes) if len(oos_sharpes) > 1 else 0
    # Lower std = better. Std of 0.5 = 50 score, std of 0 = 100
    consistency_score = max(0, 100 - sharpe_std * 100)
    consistency_contribution = consistency_score * 0.20

    # 4. Drawdown consistency (10% weight)
    dd_values = [f.test_max_drawdown for f in valid_folds]
    dd_cv = calculate_drawdown_consistency(dd_values)
    # Lower CV = better. CV of 50% = 50 score
    dd_score = max(0, 100 - dd_cv)
    dd_contribution = dd_score * 0.10

    total_score = (
        sharpe_contribution
        + degradation_contribution
        + consistency_contribution
        + dd_contribution
    )

    return round(min(100, max(0, total_score)), 2)


# =============================================================================
# WALK-FORWARD VALIDATOR
# =============================================================================


class WalkForwardValidator:
    """
    Walk-forward validation engine for strategy testing.

    Walk-forward validation simulates real trading by:
    1. Training/optimizing on historical data (in-sample)
    2. Testing on subsequent unseen data (out-of-sample)
    3. Rolling forward and repeating

    This process reveals whether a strategy's performance is due to
    genuine predictive power or merely curve-fitting to past data.

    Example:
        ```python
        config = WalkForwardConfig(
            train_period_days=252,
            test_period_days=63,
            step_days=21,
        )

        validator = WalkForwardValidator(config)

        result = validator.run(
            strategy=MyStrategy(),
            price_data={'AAPL': prices, 'dates': dates},
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2024, 1, 1),
        )

        print(f"Robustness Score: {result.robustness_score}")
        print(f"OOS Sharpe: {result.aggregate_sharpe}")
        ```
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        risk_free_rate: float = 0.02,  # Annual risk-free rate
    ):
        """
        Initialize walk-forward validator.

        Args:
            config: Walk-forward configuration
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.config = config
        self.risk_free_rate = risk_free_rate
        self._fold_results: list[FoldResult] = []

    def run(
        self,
        strategy: BacktestableStrategy,
        price_data: dict[str, np.ndarray],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> WalkForwardResult:
        """
        Execute walk-forward validation.

        Args:
            strategy: Strategy implementing BacktestableStrategy protocol
            price_data: Dictionary with 'prices' and 'dates' arrays
            start_date: Start of validation period (default: data start)
            end_date: End of validation period (default: data end)

        Returns:
            WalkForwardResult with aggregate and per-fold metrics
        """
        # Validate input data
        if "prices" not in price_data or "dates" not in price_data:
            raise ValueError("price_data must contain 'prices' and 'dates' keys")

        prices = price_data["prices"]
        dates = price_data["dates"]

        if len(prices) != len(dates):
            raise ValueError("prices and dates must have same length")

        if len(prices) < self.config.train_period_days + self.config.test_period_days:
            raise ValueError("Insufficient data for walk-forward validation")

        # Determine date range
        if start_date is None:
            start_date = dates[0] if isinstance(dates[0], datetime) else datetime.fromtimestamp(dates[0])
        if end_date is None:
            end_date = dates[-1] if isinstance(dates[-1], datetime) else datetime.fromtimestamp(dates[-1])

        logger.info(
            f"Starting walk-forward validation: {start_date.date()} to {end_date.date()}, "
            f"train={self.config.train_period_days}d, test={self.config.test_period_days}d, "
            f"step={self.config.step_days}d, anchored={self.config.anchored}"
        )

        # Generate folds
        folds = list(self._create_folds(start_date, end_date))
        logger.info(f"Generated {len(folds)} walk-forward folds")

        # Run each fold
        self._fold_results = []
        for fold_num, (train_start, train_end, test_start, test_end) in enumerate(folds, 1):
            fold_result = self._run_fold(
                fold_num=fold_num,
                strategy=strategy,
                price_data=price_data,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            self._fold_results.append(fold_result)

        # Aggregate results
        result = self._aggregate_results(start_date, end_date)

        logger.info(
            f"Walk-forward complete: {result.valid_folds}/{result.total_folds} valid folds, "
            f"OOS Sharpe={result.aggregate_sharpe:.3f}, "
            f"Degradation={result.in_sample_vs_oos_degradation:.1f}%, "
            f"Robustness={result.robustness_score:.1f}"
        )

        return result

    def _create_folds(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Iterator[tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate train/test windows for walk-forward analysis.

        Args:
            start_date: Start of entire validation period
            end_date: End of entire validation period

        Yields:
            Tuples of (train_start, train_end, test_start, test_end)
        """
        train_delta = timedelta(days=self.config.train_period_days)
        test_delta = timedelta(days=self.config.test_period_days)
        step_delta = timedelta(days=self.config.step_days)
        gap_delta = timedelta(days=self.config.gap_days)

        if self.config.anchored:
            # Anchored (expanding window): train start is fixed
            anchor_start = start_date
            current_train_end = anchor_start + train_delta

            while True:
                train_start = anchor_start
                train_end = current_train_end
                test_start = train_end + gap_delta
                test_end = test_start + test_delta

                if test_end > end_date:
                    break

                yield (train_start, train_end, test_start, test_end)

                # Move train end forward (expanding window)
                current_train_end += step_delta
        else:
            # Rolling window: both train start and end move
            current_train_start = start_date

            while True:
                train_start = current_train_start
                train_end = train_start + train_delta
                test_start = train_end + gap_delta
                test_end = test_start + test_delta

                if test_end > end_date:
                    break

                yield (train_start, train_end, test_start, test_end)

                # Move window forward
                current_train_start += step_delta

    def _run_fold(
        self,
        fold_num: int,
        strategy: BacktestableStrategy,
        price_data: dict[str, np.ndarray],
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
    ) -> FoldResult:
        """
        Run a single walk-forward fold.

        Args:
            fold_num: Fold number (1-indexed)
            strategy: Strategy to test
            price_data: Full price data
            train_start: Training period start
            train_end: Training period end
            test_start: Test period start
            test_end: Test period end

        Returns:
            FoldResult with metrics for this fold
        """
        result = FoldResult(
            fold_number=fold_num,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        try:
            # Extract train and test data
            train_data = self._extract_period_data(price_data, train_start, train_end)
            test_data = self._extract_period_data(price_data, test_start, test_end)

            # Validate data sufficiency
            if len(train_data.get("prices", [])) < self.config.min_train_samples:
                result.is_valid = False
                result.validation_message = f"Insufficient training data: {len(train_data.get('prices', []))} < {self.config.min_train_samples}"
                return result

            # Train strategy on in-sample data
            strategy.fit(train_data)

            # Get optimized parameters
            result.optimized_params = strategy.get_parameters()

            # Generate signals on train data (for IS metrics)
            train_signals = strategy.predict(train_data)
            train_returns = self._calculate_returns(train_data["prices"], train_signals)
            result.train_sharpe = self._calculate_sharpe(train_returns)
            result.train_return = self._calculate_total_return(train_returns)
            result.train_volatility = self._calculate_volatility(train_returns)
            result.train_max_drawdown = self._calculate_max_drawdown(train_returns)
            result.n_train_trades = self._count_trades(train_signals)

            # Generate signals on test data (for OOS metrics)
            test_signals = strategy.predict(test_data)
            test_returns = self._calculate_returns(test_data["prices"], test_signals)
            result.test_sharpe = self._calculate_sharpe(test_returns)
            result.test_return = self._calculate_total_return(test_returns)
            result.test_volatility = self._calculate_volatility(test_returns)
            result.test_max_drawdown = self._calculate_max_drawdown(test_returns)
            result.n_test_trades = self._count_trades(test_signals)
            result.daily_returns = test_returns

            # Validate minimum trades
            if result.n_test_trades < self.config.min_trades_per_fold:
                result.is_valid = False
                result.validation_message = f"Insufficient test trades: {result.n_test_trades} < {self.config.min_trades_per_fold}"

            logger.debug(
                f"Fold {fold_num}: Train Sharpe={result.train_sharpe:.3f}, "
                f"Test Sharpe={result.test_sharpe:.3f}, "
                f"Degradation={result.degradation:.1f}%"
            )

        except Exception as e:
            logger.exception(f"Fold {fold_num} failed: {e}")
            result.is_valid = False
            result.validation_message = f"Error: {str(e)}"

        return result

    def _extract_period_data(
        self,
        full_data: dict[str, np.ndarray],
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, np.ndarray]:
        """Extract data for a specific date range."""
        dates = full_data["dates"]

        # Find indices for date range
        start_idx = None
        end_idx = None

        for i, date in enumerate(dates):
            if isinstance(date, datetime):
                d = date
            else:
                d = datetime.fromtimestamp(date)

            if start_idx is None and d >= start_date:
                start_idx = i
            if d <= end_date:
                end_idx = i + 1

        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(dates)

        # Extract all arrays for the period
        period_data = {}
        for key, values in full_data.items():
            if isinstance(values, np.ndarray) and len(values) == len(dates):
                period_data[key] = values[start_idx:end_idx]

        return period_data

    def _calculate_returns(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
    ) -> np.ndarray:
        """Calculate strategy returns from prices and signals."""
        if len(prices) < 2 or len(signals) < 1:
            return np.array([])

        # Calculate price returns
        price_returns = np.diff(prices) / prices[:-1]

        # Align signals with returns (signal at t applies to return from t to t+1)
        aligned_signals = signals[:len(price_returns)]
        if len(aligned_signals) < len(price_returns):
            # Pad with zeros if signals shorter
            aligned_signals = np.concatenate([
                aligned_signals,
                np.zeros(len(price_returns) - len(aligned_signals))
            ])

        # Strategy returns = signal * price return
        strategy_returns = aligned_signals * price_returns

        return strategy_returns

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / 252)
        mean_excess = np.mean(excess_returns)
        std_returns = np.std(returns)

        if std_returns < 1e-8:
            return 0.0

        return (mean_excess / std_returns) * np.sqrt(252)

    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total compounded return."""
        if len(returns) == 0:
            return 0.0
        return np.prod(1 + returns) - 1

    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0
        return np.std(returns) * np.sqrt(252)

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        # Build equity curve
        equity = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max

        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    def _count_trades(self, signals: np.ndarray) -> int:
        """Count number of trades (signal changes)."""
        if len(signals) < 2:
            return 0

        # Count changes in signal (entering/exiting positions)
        signal_changes = np.diff(signals)
        trades = np.sum(signal_changes != 0)

        return int(trades)

    def _aggregate_results(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> WalkForwardResult:
        """Aggregate individual fold results into overall metrics."""
        result = WalkForwardResult(
            config=self.config,
            fold_results=self._fold_results,
            total_folds=len(self._fold_results),
            valid_folds=sum(1 for f in self._fold_results if f.is_valid),
            start_date=start_date,
            end_date=end_date,
        )

        valid_folds = [f for f in self._fold_results if f.is_valid]

        if not valid_folds:
            logger.warning("No valid folds in walk-forward validation")
            return result

        # Concatenate all OOS returns
        all_returns = np.concatenate([f.daily_returns for f in valid_folds if len(f.daily_returns) > 0])
        result.aggregate_returns = all_returns

        # Calculate aggregate metrics from combined returns
        if len(all_returns) > 0:
            result.aggregate_sharpe = self._calculate_sharpe(all_returns)
            result.aggregate_return = self._calculate_total_return(all_returns)
            result.aggregate_volatility = self._calculate_volatility(all_returns)
            result.aggregate_max_drawdown = self._calculate_max_drawdown(all_returns)

        # Calculate degradation
        is_sharpes = [f.train_sharpe for f in valid_folds]
        oos_sharpes = [f.test_sharpe for f in valid_folds]
        result.in_sample_vs_oos_degradation = calculate_oos_degradation(is_sharpes, oos_sharpes)

        # Calculate parameter stability
        fold_params = [f.optimized_params for f in valid_folds]
        result.parameter_stability = calculate_parameter_stability(fold_params)

        # Calculate robustness score
        result.robustness_score = calculate_robustness_score(result)

        return result

    def get_robustness_score(self) -> float:
        """
        Get the robustness score from the last run.

        Returns:
            Robustness score (0-100), or 0 if no run completed
        """
        if not self._fold_results:
            return 0.0

        # Create a temporary result to calculate score
        temp_result = WalkForwardResult(
            config=self.config,
            fold_results=self._fold_results,
            valid_folds=sum(1 for f in self._fold_results if f.is_valid),
        )

        # Calculate degradation for scoring
        valid_folds = [f for f in self._fold_results if f.is_valid]
        if valid_folds:
            is_sharpes = [f.train_sharpe for f in valid_folds]
            oos_sharpes = [f.test_sharpe for f in valid_folds]
            temp_result.in_sample_vs_oos_degradation = calculate_oos_degradation(is_sharpes, oos_sharpes)

        return calculate_robustness_score(temp_result)

    def plot_equity_curves(self) -> dict[str, Any]:
        """
        Generate data for equity curve visualization.

        Returns:
            Dictionary with data for plotting:
            - 'dates': List of date strings
            - 'equity': List of cumulative equity values
            - 'drawdown': List of drawdown values
            - 'fold_boundaries': List of fold start dates
        """
        if not self._fold_results:
            return {
                "dates": [],
                "equity": [],
                "drawdown": [],
                "fold_boundaries": [],
            }

        valid_folds = [f for f in self._fold_results if f.is_valid]

        if not valid_folds:
            return {
                "dates": [],
                "equity": [],
                "drawdown": [],
                "fold_boundaries": [],
            }

        # Concatenate returns and build equity curve
        all_returns = np.concatenate([f.daily_returns for f in valid_folds if len(f.daily_returns) > 0])

        if len(all_returns) == 0:
            return {
                "dates": [],
                "equity": [],
                "drawdown": [],
                "fold_boundaries": [],
            }

        equity = np.cumprod(1 + all_returns)
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max

        # Generate date-like indices (actual dates would need to be passed through)
        dates = list(range(len(equity)))

        # Get fold boundaries
        fold_boundaries = []
        idx = 0
        for fold in valid_folds:
            if len(fold.daily_returns) > 0:
                fold_boundaries.append(idx)
                idx += len(fold.daily_returns)

        return {
            "dates": dates,
            "equity": equity.tolist(),
            "drawdown": drawdown.tolist(),
            "fold_boundaries": fold_boundaries,
        }


# =============================================================================
# STRATEGY WRAPPER FOR SIMPLE FUNCTIONS
# =============================================================================


class SimpleStrategy:
    """
    Wrapper to convert simple signal functions to BacktestableStrategy.

    Example:
        ```python
        def my_signal_func(data: dict) -> np.ndarray:
            prices = data['prices']
            sma = np.convolve(prices, np.ones(20)/20, mode='valid')
            signals = np.where(prices[19:] > sma, 1, -1)
            return signals

        strategy = SimpleStrategy(my_signal_func)
        ```
    """

    def __init__(
        self,
        signal_func: Callable[[dict[str, np.ndarray]], np.ndarray],
        params: dict[str, Any] | None = None,
    ):
        """
        Initialize simple strategy wrapper.

        Args:
            signal_func: Function that takes data dict and returns signal array
            params: Optional parameters dictionary
        """
        self._signal_func = signal_func
        self._params = params or {}

    def fit(self, train_data: dict[str, np.ndarray]) -> None:
        """No fitting for simple strategies."""
        pass

    def predict(self, test_data: dict[str, np.ndarray]) -> np.ndarray:
        """Generate signals using the provided function."""
        return self._signal_func(test_data)

    def get_parameters(self) -> dict[str, Any]:
        """Return stored parameters."""
        return self._params.copy()

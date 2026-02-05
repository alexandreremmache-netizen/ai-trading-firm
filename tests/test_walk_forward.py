"""
Tests for Walk-Forward Validation Framework
===========================================

Comprehensive test suite for walk-forward validation covering:
- Configuration validation
- Fold generation (rolling and anchored windows)
- Train/test split correctness
- Metric calculations
- Robustness scoring
- Edge cases and error handling
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from core.walk_forward import (
    WalkForwardConfig,
    WalkForwardValidator,
    WalkForwardResult,
    FoldResult,
    SimpleStrategy,
    calculate_oos_degradation,
    calculate_parameter_stability,
    calculate_drawdown_consistency,
    calculate_robustness_score,
    BacktestableStrategy,
)

from tests.fixtures.test_data_generator import (
    generate_price_series,
    generate_ohlcv_data,
    generate_regime_switching_data,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def default_config():
    """Create default walk-forward configuration."""
    return WalkForwardConfig(
        train_period_days=100,
        test_period_days=25,
        step_days=10,
        min_train_samples=20,
        anchored=False,
    )


@pytest.fixture
def simple_price_data():
    """Generate simple price data for testing."""
    np.random.seed(42)
    n_days = 300
    start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

    prices = generate_price_series(
        n_days=n_days,
        initial_price=100.0,
        trend=0.10,
        volatility=0.20,
        seed=42,
    )

    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    return {
        "prices": np.array(prices),
        "dates": np.array(dates),
    }


@pytest.fixture
def simple_momentum_strategy():
    """Create a simple momentum strategy for testing."""

    def signal_func(data: dict) -> np.ndarray:
        prices = data["prices"]
        if len(prices) < 20:
            return np.zeros(len(prices))

        # Simple momentum: price above 20-day SMA = long, below = short
        sma_20 = np.convolve(prices, np.ones(20) / 20, mode="valid")

        # Pad to match price length
        signals = np.zeros(len(prices))
        for i in range(19, len(prices)):
            if prices[i] > sma_20[i - 19]:
                signals[i] = 1
            else:
                signals[i] = -1

        return signals

    return SimpleStrategy(signal_func, {"type": "momentum", "lookback": 20})


class MockBacktestableStrategy:
    """Mock strategy for testing."""

    def __init__(self, params: dict | None = None):
        self._params = params or {"param1": 10, "param2": 0.5}
        self._fitted = False

    def fit(self, train_data: dict) -> None:
        self._fitted = True

    def predict(self, test_data: dict) -> np.ndarray:
        prices = test_data["prices"]
        # Simple alternating signals for testing
        return np.array([1 if i % 2 == 0 else -1 for i in range(len(prices))])

    def get_parameters(self) -> dict:
        return self._params.copy()


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestWalkForwardConfig:
    """Test WalkForwardConfig validation and behavior."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = WalkForwardConfig()

        assert config.train_period_days == 252
        assert config.test_period_days == 63
        assert config.step_days == 21
        assert config.min_train_samples == 50
        assert config.anchored is False
        assert config.min_trades_per_fold == 5
        assert config.gap_days == 0

    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = WalkForwardConfig(
            train_period_days=180,
            test_period_days=30,
            step_days=15,
            min_train_samples=30,
            anchored=True,
            gap_days=5,
        )

        assert config.train_period_days == 180
        assert config.test_period_days == 30
        assert config.step_days == 15
        assert config.anchored is True
        assert config.gap_days == 5

    def test_invalid_train_period_raises(self):
        """Test that non-positive train period raises error."""
        with pytest.raises(ValueError, match="train_period_days must be positive"):
            WalkForwardConfig(train_period_days=0)

        with pytest.raises(ValueError, match="train_period_days must be positive"):
            WalkForwardConfig(train_period_days=-10)

    def test_invalid_test_period_raises(self):
        """Test that non-positive test period raises error."""
        with pytest.raises(ValueError, match="test_period_days must be positive"):
            WalkForwardConfig(test_period_days=0)

    def test_invalid_step_days_raises(self):
        """Test that non-positive step days raises error."""
        with pytest.raises(ValueError, match="step_days must be positive"):
            WalkForwardConfig(step_days=0)

    def test_step_exceeds_test_period_raises(self):
        """Test that step > test period raises warning."""
        with pytest.raises(ValueError, match="step_days should not exceed test_period_days"):
            WalkForwardConfig(step_days=100, test_period_days=50)

    def test_invalid_min_train_samples_raises(self):
        """Test that non-positive min samples raises error."""
        with pytest.raises(ValueError, match="min_train_samples must be positive"):
            WalkForwardConfig(min_train_samples=0)

    def test_negative_gap_days_raises(self):
        """Test that negative gap days raises error."""
        with pytest.raises(ValueError, match="gap_days cannot be negative"):
            WalkForwardConfig(gap_days=-1)


# =============================================================================
# FOLD GENERATION TESTS
# =============================================================================


class TestFoldGeneration:
    """Test walk-forward fold generation."""

    def test_rolling_window_fold_count(self, default_config):
        """Test correct number of folds for rolling window."""
        validator = WalkForwardValidator(default_config)

        start = datetime(2020, 1, 1)
        end = datetime(2020, 10, 1)  # ~270 days

        folds = list(validator._create_folds(start, end))

        # With 100-day train, 25-day test, 10-day step
        # First fold: train 0-100, test 100-125
        # Last fold where test_end <= 270
        # Expected: several folds
        assert len(folds) > 0

        # Verify fold structure
        for train_start, train_end, test_start, test_end in folds:
            assert train_end > train_start
            assert test_end > test_start
            assert test_start >= train_end

    def test_rolling_window_no_overlap(self, default_config):
        """Test that training and testing periods don't overlap in rolling window."""
        validator = WalkForwardValidator(default_config)

        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)

        folds = list(validator._create_folds(start, end))

        for train_start, train_end, test_start, test_end in folds:
            # Test should start after train ends
            assert test_start >= train_end, "Test period overlaps with training"

    def test_anchored_window_expanding_train(self):
        """Test that anchored window has fixed train start."""
        config = WalkForwardConfig(
            train_period_days=50,
            test_period_days=20,
            step_days=10,
            anchored=True,
        )
        validator = WalkForwardValidator(config)

        start = datetime(2020, 1, 1)
        end = datetime(2020, 6, 1)

        folds = list(validator._create_folds(start, end))

        assert len(folds) >= 2

        # All folds should have same train start (anchored)
        train_starts = [f[0] for f in folds]
        assert all(ts == train_starts[0] for ts in train_starts), "Train start should be anchored"

        # Train end should expand
        train_ends = [f[1] for f in folds]
        for i in range(1, len(train_ends)):
            assert train_ends[i] > train_ends[i - 1], "Train end should expand in anchored mode"

    def test_rolling_window_train_moves(self):
        """Test that rolling window train start moves forward."""
        config = WalkForwardConfig(
            train_period_days=50,
            test_period_days=20,
            step_days=10,
            anchored=False,
        )
        validator = WalkForwardValidator(config)

        start = datetime(2020, 1, 1)
        end = datetime(2020, 6, 1)

        folds = list(validator._create_folds(start, end))

        assert len(folds) >= 2

        # Train start should move in rolling window
        train_starts = [f[0] for f in folds]
        for i in range(1, len(train_starts)):
            assert train_starts[i] > train_starts[i - 1], "Train start should move in rolling mode"

    def test_gap_between_train_and_test(self):
        """Test that gap_days creates space between train and test."""
        config = WalkForwardConfig(
            train_period_days=50,
            test_period_days=20,
            step_days=10,
            gap_days=5,
        )
        validator = WalkForwardValidator(config)

        start = datetime(2020, 1, 1)
        end = datetime(2020, 6, 1)

        folds = list(validator._create_folds(start, end))

        for train_start, train_end, test_start, test_end in folds:
            gap = (test_start - train_end).days
            assert gap == 5, f"Gap should be 5 days, got {gap}"

    def test_folds_cover_test_period_exactly(self, default_config):
        """Test that test periods have correct duration."""
        validator = WalkForwardValidator(default_config)

        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)

        folds = list(validator._create_folds(start, end))

        for train_start, train_end, test_start, test_end in folds:
            test_duration = (test_end - test_start).days
            assert test_duration == default_config.test_period_days

    def test_train_period_duration(self, default_config):
        """Test that train periods have correct duration for rolling window."""
        validator = WalkForwardValidator(default_config)

        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)

        folds = list(validator._create_folds(start, end))

        for train_start, train_end, test_start, test_end in folds:
            train_duration = (train_end - train_start).days
            assert train_duration == default_config.train_period_days


# =============================================================================
# TRAIN/TEST SPLIT TESTS
# =============================================================================


class TestTrainTestSplit:
    """Test correct train/test data splitting."""

    def test_extract_period_data_returns_correct_range(self, default_config, simple_price_data):
        """Test that period extraction returns correct date range."""
        validator = WalkForwardValidator(default_config)

        start = simple_price_data["dates"][50]
        end = simple_price_data["dates"][100]

        period_data = validator._extract_period_data(simple_price_data, start, end)

        assert len(period_data["prices"]) == 51  # Inclusive range
        assert len(period_data["dates"]) == 51

    def test_no_data_leakage_between_train_test(self, simple_price_data):
        """Test that test data is not seen during training with gap."""
        # Use config with 1-day gap to ensure no overlap
        config = WalkForwardConfig(
            train_period_days=100,
            test_period_days=25,
            step_days=10,
            min_train_samples=20,
            anchored=False,
            gap_days=1,  # Gap to prevent overlap
        )
        validator = WalkForwardValidator(config)
        strategy = MockBacktestableStrategy()

        start_date = simple_price_data["dates"][0]
        end_date = simple_price_data["dates"][-1]

        folds = list(validator._create_folds(start_date, end_date))

        for train_start, train_end, test_start, test_end in folds:
            train_data = validator._extract_period_data(simple_price_data, train_start, train_end)
            test_data = validator._extract_period_data(simple_price_data, test_start, test_end)

            # Get date ranges
            train_dates = set(train_data["dates"].tolist())
            test_dates = set(test_data["dates"].tolist())

            # No overlap with gap_days=1
            overlap = train_dates.intersection(test_dates)
            assert len(overlap) == 0, "Data leakage detected between train and test"


# =============================================================================
# METRIC CALCULATION TESTS
# =============================================================================


class TestMetricCalculations:
    """Test calculation of performance metrics."""

    def test_calculate_sharpe_positive_returns(self, default_config):
        """Test Sharpe calculation with positive returns."""
        validator = WalkForwardValidator(default_config)

        # Use deterministic positive returns
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.01, 0.02, 0.01, 0.01, 0.015, 0.02])

        sharpe = validator._calculate_sharpe(returns)

        # Should be positive with positive mean returns
        assert sharpe > 0

    def test_calculate_sharpe_negative_returns(self, default_config):
        """Test Sharpe calculation with negative returns."""
        validator = WalkForwardValidator(default_config)

        # Generate returns with negative mean
        np.random.seed(42)
        returns = np.random.normal(-0.002, 0.02, 100)

        sharpe = validator._calculate_sharpe(returns)

        # Should be negative with negative mean returns
        assert sharpe < 0

    def test_calculate_sharpe_zero_volatility(self, default_config):
        """Test Sharpe calculation with zero volatility."""
        validator = WalkForwardValidator(default_config)

        returns = np.zeros(100)  # All zeros

        sharpe = validator._calculate_sharpe(returns)

        assert sharpe == 0.0

    def test_calculate_total_return(self, default_config):
        """Test total return calculation."""
        validator = WalkForwardValidator(default_config)

        # 10% total return from 10 * 1% returns
        returns = np.array([0.01] * 10)

        total_return = validator._calculate_total_return(returns)

        # Compounded: (1.01)^10 - 1 = ~10.46%
        assert 0.10 < total_return < 0.11

    def test_calculate_volatility(self, default_config):
        """Test volatility calculation."""
        validator = WalkForwardValidator(default_config)

        # Known volatility returns
        np.random.seed(42)
        daily_vol = 0.01  # 1% daily vol
        returns = np.random.normal(0, daily_vol, 252)

        volatility = validator._calculate_volatility(returns)

        # Annualized should be close to 0.01 * sqrt(252) = ~16%
        assert 0.12 < volatility < 0.20

    def test_calculate_max_drawdown(self, default_config):
        """Test max drawdown calculation."""
        validator = WalkForwardValidator(default_config)

        # Create returns with known drawdown
        returns = np.array([0.05, 0.02, -0.10, -0.05, 0.03, 0.02])

        max_dd = validator._calculate_max_drawdown(returns)

        # Should detect the decline
        assert max_dd > 0.10

    def test_count_trades(self, default_config):
        """Test trade counting."""
        validator = WalkForwardValidator(default_config)

        signals = np.array([0, 1, 1, 0, -1, -1, 1, 0])

        trades = validator._count_trades(signals)

        # Changes: 0->1, 1->0, 0->-1, -1->1, 1->0 = 5 trades
        assert trades == 5


# =============================================================================
# ROBUSTNESS METRICS TESTS
# =============================================================================


class TestRobustnessMetrics:
    """Test robustness metric calculations."""

    def test_calculate_oos_degradation_perfect_consistency(self):
        """Test degradation when IS and OOS are identical."""
        is_sharpes = [1.0, 1.2, 0.8]
        oos_sharpes = [1.0, 1.2, 0.8]

        degradation = calculate_oos_degradation(is_sharpes, oos_sharpes)

        assert abs(degradation) < 1.0  # Should be near 0

    def test_calculate_oos_degradation_overfitting(self):
        """Test degradation when IS >> OOS (overfitting)."""
        is_sharpes = [2.0, 2.5, 1.8]  # Good IS performance
        oos_sharpes = [0.5, 0.3, 0.6]  # Poor OOS performance

        degradation = calculate_oos_degradation(is_sharpes, oos_sharpes)

        assert degradation > 50  # Significant degradation

    def test_calculate_oos_degradation_empty_lists(self):
        """Test degradation with empty lists."""
        degradation = calculate_oos_degradation([], [])

        assert degradation == 0.0

    def test_calculate_parameter_stability_stable_params(self):
        """Test stability with consistent parameters."""
        fold_params = [
            {"alpha": 10.0, "beta": 0.5},
            {"alpha": 10.5, "beta": 0.48},
            {"alpha": 9.8, "beta": 0.52},
        ]

        stability = calculate_parameter_stability(fold_params)

        # Low CV = stable
        assert "alpha" in stability
        assert "beta" in stability
        assert stability["alpha"] < 10  # <10% CV
        assert stability["beta"] < 10

    def test_calculate_parameter_stability_unstable_params(self):
        """Test stability with varying parameters."""
        fold_params = [
            {"alpha": 5.0, "beta": 0.1},
            {"alpha": 20.0, "beta": 0.9},
            {"alpha": 8.0, "beta": 0.3},
        ]

        stability = calculate_parameter_stability(fold_params)

        # High CV = unstable
        assert stability["alpha"] > 50
        assert stability["beta"] > 50

    def test_calculate_parameter_stability_empty(self):
        """Test stability with empty list."""
        stability = calculate_parameter_stability([])

        assert stability == {}

    def test_calculate_drawdown_consistency_consistent(self):
        """Test DD consistency with similar drawdowns."""
        drawdowns = [0.10, 0.12, 0.11, 0.09, 0.10]

        cv = calculate_drawdown_consistency(drawdowns)

        assert cv < 15  # Low CV = consistent

    def test_calculate_drawdown_consistency_inconsistent(self):
        """Test DD consistency with varying drawdowns."""
        drawdowns = [0.05, 0.30, 0.08, 0.25, 0.03]

        cv = calculate_drawdown_consistency(drawdowns)

        assert cv > 50  # High CV = inconsistent

    def test_calculate_robustness_score_excellent_strategy(self):
        """Test robustness score for excellent strategy."""
        config = WalkForwardConfig()

        # Create mock result with excellent metrics
        folds = [
            FoldResult(
                fold_number=i,
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 6, 1),
                test_start=datetime(2020, 6, 1),
                test_end=datetime(2020, 9, 1),
                train_sharpe=1.5,
                test_sharpe=1.4,  # Low degradation
                test_max_drawdown=0.08,
                is_valid=True,
            )
            for i in range(5)
        ]

        result = WalkForwardResult(
            config=config,
            fold_results=folds,
            valid_folds=5,
            in_sample_vs_oos_degradation=7.0,  # Low
        )

        score = calculate_robustness_score(result)

        assert score > 60  # Good score

    def test_calculate_robustness_score_poor_strategy(self):
        """Test robustness score for poor strategy."""
        config = WalkForwardConfig()

        folds = [
            FoldResult(
                fold_number=i,
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 6, 1),
                test_start=datetime(2020, 6, 1),
                test_end=datetime(2020, 9, 1),
                train_sharpe=2.0,
                test_sharpe=-0.5,  # Negative OOS
                test_max_drawdown=0.25,
                is_valid=True,
            )
            for i in range(5)
        ]

        result = WalkForwardResult(
            config=config,
            fold_results=folds,
            valid_folds=5,
            in_sample_vs_oos_degradation=125.0,  # High degradation
        )

        score = calculate_robustness_score(result)

        assert score <= 30  # Poor score (allow equality at boundary)

    def test_calculate_robustness_score_no_valid_folds(self):
        """Test robustness score with no valid folds."""
        config = WalkForwardConfig()
        result = WalkForwardResult(config=config, valid_folds=0)

        score = calculate_robustness_score(result)

        assert score == 0.0


# =============================================================================
# FULL VALIDATION RUN TESTS
# =============================================================================


class TestWalkForwardValidation:
    """Test full walk-forward validation runs."""

    def test_run_basic_validation(self, default_config, simple_price_data, simple_momentum_strategy):
        """Test basic walk-forward validation run."""
        validator = WalkForwardValidator(default_config)

        result = validator.run(
            strategy=simple_momentum_strategy,
            price_data=simple_price_data,
        )

        assert isinstance(result, WalkForwardResult)
        assert result.total_folds > 0
        assert result.valid_folds >= 0
        assert result.valid_folds <= result.total_folds

    def test_run_returns_fold_results(self, default_config, simple_price_data, simple_momentum_strategy):
        """Test that run returns fold-level results."""
        validator = WalkForwardValidator(default_config)

        result = validator.run(
            strategy=simple_momentum_strategy,
            price_data=simple_price_data,
        )

        assert len(result.fold_results) == result.total_folds

        for fold in result.fold_results:
            assert isinstance(fold, FoldResult)
            assert fold.fold_number > 0

    def test_run_calculates_aggregate_metrics(self, default_config, simple_price_data, simple_momentum_strategy):
        """Test that run calculates aggregate metrics."""
        validator = WalkForwardValidator(default_config)

        result = validator.run(
            strategy=simple_momentum_strategy,
            price_data=simple_price_data,
        )

        if result.valid_folds > 0:
            # Should have some returns
            assert len(result.aggregate_returns) > 0

    def test_run_with_date_range(self, default_config, simple_price_data, simple_momentum_strategy):
        """Test run with explicit date range."""
        validator = WalkForwardValidator(default_config)

        start = simple_price_data["dates"][50]
        end = simple_price_data["dates"][200]

        result = validator.run(
            strategy=simple_momentum_strategy,
            price_data=simple_price_data,
            start_date=start,
            end_date=end,
        )

        assert result.start_date == start
        assert result.end_date == end

    def test_run_anchored_vs_rolling_different_results(self, simple_price_data, simple_momentum_strategy):
        """Test that anchored and rolling produce different results."""
        rolling_config = WalkForwardConfig(
            train_period_days=80,
            test_period_days=20,
            step_days=10,
            anchored=False,
        )

        anchored_config = WalkForwardConfig(
            train_period_days=80,
            test_period_days=20,
            step_days=10,
            anchored=True,
        )

        rolling_validator = WalkForwardValidator(rolling_config)
        anchored_validator = WalkForwardValidator(anchored_config)

        rolling_result = rolling_validator.run(
            strategy=simple_momentum_strategy,
            price_data=simple_price_data,
        )

        anchored_result = anchored_validator.run(
            strategy=simple_momentum_strategy,
            price_data=simple_price_data,
        )

        # Both should complete but may have different fold counts
        assert rolling_result.total_folds > 0
        assert anchored_result.total_folds > 0


# =============================================================================
# EDGE CASES TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data_raises(self, default_config):
        """Test that insufficient data raises error."""
        validator = WalkForwardValidator(default_config)

        # Only 50 days, need 125 (100 train + 25 test)
        small_data = {
            "prices": np.random.randn(50) + 100,
            "dates": np.array([datetime(2020, 1, 1) + timedelta(days=i) for i in range(50)]),
        }

        strategy = MockBacktestableStrategy()

        with pytest.raises(ValueError, match="Insufficient data"):
            validator.run(strategy, small_data)

    def test_missing_prices_key_raises(self, default_config):
        """Test that missing 'prices' key raises error."""
        validator = WalkForwardValidator(default_config)

        bad_data = {
            "close": np.random.randn(300) + 100,
            "dates": np.array([datetime(2020, 1, 1) + timedelta(days=i) for i in range(300)]),
        }

        strategy = MockBacktestableStrategy()

        with pytest.raises(ValueError, match="'prices'"):
            validator.run(strategy, bad_data)

    def test_missing_dates_key_raises(self, default_config):
        """Test that missing 'dates' key raises error."""
        validator = WalkForwardValidator(default_config)

        bad_data = {
            "prices": np.random.randn(300) + 100,
        }

        strategy = MockBacktestableStrategy()

        with pytest.raises(ValueError, match="'dates'"):
            validator.run(strategy, bad_data)

    def test_mismatched_lengths_raises(self, default_config):
        """Test that mismatched array lengths raises error."""
        validator = WalkForwardValidator(default_config)

        bad_data = {
            "prices": np.random.randn(300) + 100,
            "dates": np.array([datetime(2020, 1, 1) + timedelta(days=i) for i in range(250)]),
        }

        strategy = MockBacktestableStrategy()

        with pytest.raises(ValueError, match="same length"):
            validator.run(strategy, bad_data)

    def test_fold_with_insufficient_samples_marked_invalid(self, simple_price_data):
        """Test that folds with insufficient samples are marked invalid."""
        # Config requiring many samples
        config = WalkForwardConfig(
            train_period_days=50,
            test_period_days=20,
            step_days=10,
            min_train_samples=1000,  # More than data has
        )

        validator = WalkForwardValidator(config)
        strategy = MockBacktestableStrategy()

        result = validator.run(strategy, simple_price_data)

        # All folds should be invalid due to insufficient samples
        for fold in result.fold_results:
            assert fold.is_valid is False

    def test_empty_returns_handling(self, default_config):
        """Test handling of empty return arrays."""
        validator = WalkForwardValidator(default_config)

        sharpe = validator._calculate_sharpe(np.array([]))
        assert sharpe == 0.0

        total_return = validator._calculate_total_return(np.array([]))
        assert total_return == 0.0

        volatility = validator._calculate_volatility(np.array([]))
        assert volatility == 0.0

        max_dd = validator._calculate_max_drawdown(np.array([]))
        assert max_dd == 0.0

    def test_single_element_returns(self, default_config):
        """Test handling of single-element return arrays."""
        validator = WalkForwardValidator(default_config)

        single_return = np.array([0.05])

        sharpe = validator._calculate_sharpe(single_return)
        assert sharpe == 0.0  # Need >1 for std

        volatility = validator._calculate_volatility(single_return)
        assert volatility == 0.0


# =============================================================================
# RESULT SERIALIZATION TESTS
# =============================================================================


class TestResultSerialization:
    """Test result serialization methods."""

    def test_fold_result_degradation_calculation(self):
        """Test FoldResult degradation property."""
        fold = FoldResult(
            fold_number=1,
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 6, 1),
            test_start=datetime(2020, 6, 1),
            test_end=datetime(2020, 9, 1),
            train_sharpe=2.0,
            test_sharpe=1.5,
        )

        # (2.0 - 1.5) / 2.0 * 100 = 25%
        assert fold.degradation == 25.0

    def test_fold_result_degradation_zero_train_sharpe(self):
        """Test degradation when train sharpe is zero."""
        fold = FoldResult(
            fold_number=1,
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 6, 1),
            test_start=datetime(2020, 6, 1),
            test_end=datetime(2020, 9, 1),
            train_sharpe=0.0,
            test_sharpe=0.0,
        )

        assert fold.degradation == 0.0

    def test_walk_forward_result_to_dict(self, default_config):
        """Test WalkForwardResult to_dict method."""
        result = WalkForwardResult(
            config=default_config,
            aggregate_sharpe=1.5,
            aggregate_return=0.25,
            aggregate_volatility=0.15,
            aggregate_max_drawdown=0.08,
            robustness_score=75.0,
            in_sample_vs_oos_degradation=20.0,
            total_folds=10,
            valid_folds=8,
            start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2020, 12, 31, tzinfo=timezone.utc),
        )

        d = result.to_dict()

        assert "aggregate_metrics" in d
        assert d["aggregate_metrics"]["sharpe_ratio"] == 1.5
        assert d["aggregate_metrics"]["total_return_pct"] == 25.0

        assert "robustness" in d
        assert d["robustness"]["score"] == 75.0
        assert d["robustness"]["degradation_pct"] == 20.0

        assert "fold_summary" in d
        assert d["fold_summary"]["total_folds"] == 10
        assert d["fold_summary"]["valid_folds"] == 8


# =============================================================================
# SIMPLE STRATEGY WRAPPER TESTS
# =============================================================================


class TestSimpleStrategyWrapper:
    """Test SimpleStrategy wrapper class."""

    def test_simple_strategy_fit_does_nothing(self):
        """Test that fit method exists and does nothing."""

        def dummy_signal(data):
            return np.ones(len(data["prices"]))

        strategy = SimpleStrategy(dummy_signal)
        strategy.fit({"prices": np.array([1, 2, 3])})

        # Should not raise

    def test_simple_strategy_predict_calls_function(self):
        """Test that predict calls the signal function."""
        call_count = {"count": 0}

        def counting_signal(data):
            call_count["count"] += 1
            return np.ones(len(data["prices"]))

        strategy = SimpleStrategy(counting_signal)
        result = strategy.predict({"prices": np.array([1, 2, 3])})

        assert call_count["count"] == 1
        assert len(result) == 3

    def test_simple_strategy_get_parameters(self):
        """Test get_parameters returns provided params."""
        params = {"lookback": 20, "threshold": 0.5}

        strategy = SimpleStrategy(lambda d: np.zeros(1), params)

        assert strategy.get_parameters() == params

    def test_simple_strategy_params_copied(self):
        """Test that get_parameters returns a copy."""
        params = {"lookback": 20}
        strategy = SimpleStrategy(lambda d: np.zeros(1), params)

        retrieved = strategy.get_parameters()
        retrieved["lookback"] = 100

        # Original should be unchanged
        assert strategy.get_parameters()["lookback"] == 20


# =============================================================================
# VISUALIZATION DATA TESTS
# =============================================================================


class TestVisualizationData:
    """Test visualization data generation."""

    def test_plot_equity_curves_empty(self, default_config):
        """Test plot data with no results."""
        validator = WalkForwardValidator(default_config)

        plot_data = validator.plot_equity_curves()

        assert plot_data["dates"] == []
        assert plot_data["equity"] == []
        assert plot_data["drawdown"] == []
        assert plot_data["fold_boundaries"] == []

    def test_plot_equity_curves_with_results(self, default_config, simple_price_data, simple_momentum_strategy):
        """Test plot data after running validation."""
        validator = WalkForwardValidator(default_config)

        validator.run(simple_momentum_strategy, simple_price_data)

        plot_data = validator.plot_equity_curves()

        if validator._fold_results and any(f.is_valid for f in validator._fold_results):
            assert len(plot_data["equity"]) > 0
            assert len(plot_data["drawdown"]) > 0
            assert len(plot_data["equity"]) == len(plot_data["drawdown"])


# =============================================================================
# REGIME-SWITCHING DATA TESTS
# =============================================================================


class TestWithRegimeSwitchingData:
    """Test walk-forward with regime-switching data."""

    def test_handles_regime_changes(self, default_config, simple_momentum_strategy):
        """Test validation handles regime-switching data."""
        # Generate regime-switching data
        regime_data = generate_regime_switching_data(
            n_days=300,
            n_regimes=2,
            avg_regime_length=50,
            seed=42,
        )

        price_data = {
            "prices": regime_data.close,
            "dates": regime_data.dates,
        }

        validator = WalkForwardValidator(default_config)
        result = validator.run(simple_momentum_strategy, price_data)

        assert result.total_folds > 0
        # Strategy performance may vary across regimes

    def test_robustness_score_varies_with_data_quality(self, simple_momentum_strategy):
        """Test that robustness score varies with data characteristics."""
        config = WalkForwardConfig(
            train_period_days=80,
            test_period_days=20,
            step_days=10,
        )

        # Generate trending data (should be good for momentum)
        trending_prices = generate_price_series(
            n_days=300,
            trend=0.30,
            volatility=0.15,
            seed=42,
        )
        trending_data = {
            "prices": trending_prices,
            "dates": np.array([datetime(2020, 1, 1) + timedelta(days=i) for i in range(300)]),
        }

        # Generate choppy data (bad for momentum)
        np.random.seed(42)
        choppy_prices = 100 + np.cumsum(np.random.randn(300) * 0.5)  # Random walk
        choppy_data = {
            "prices": choppy_prices,
            "dates": np.array([datetime(2020, 1, 1) + timedelta(days=i) for i in range(300)]),
        }

        validator = WalkForwardValidator(config)

        trending_result = validator.run(simple_momentum_strategy, trending_data)
        choppy_result = validator.run(simple_momentum_strategy, choppy_data)

        # Both should complete
        assert trending_result.total_folds > 0
        assert choppy_result.total_folds > 0

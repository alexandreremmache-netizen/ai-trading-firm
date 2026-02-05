"""
Tests for Kalman Filter Dynamic Hedge Ratio Module
===================================================

Tests the Kalman filter implementation for pairs trading hedge ratio estimation.
"""

import numpy as np
import pytest

from core.kalman_filter import (
    KalmanHedgeRatio,
    KalmanResult,
    KalmanState,
    MultiPairKalmanFilter,
    create_kalman_filter,
    create_multi_pair_filter,
)


class TestKalmanHedgeRatio:
    """Tests for KalmanHedgeRatio class."""

    def test_initialization(self):
        """Test filter initialization with default parameters."""
        kf = KalmanHedgeRatio()

        state = kf.get_state()
        assert state.beta == 1.0  # Initial hedge ratio
        assert state.n_updates == 0
        assert state.spread_value == 0.0

    def test_initialization_custom_params(self):
        """Test filter initialization with custom parameters."""
        kf = KalmanHedgeRatio(
            delta=1e-3,
            ve=1e-2,
            include_intercept=False,
            warmup_period=50,
        )

        state = kf.get_state()
        assert state.beta == 1.0
        assert state.intercept == 0.0  # No intercept

    def test_single_update(self):
        """Test single filter update."""
        kf = KalmanHedgeRatio()

        result = kf.update(price_a=100.0, price_b=50.0)

        assert isinstance(result, KalmanResult)
        assert result.n_observations == 1
        assert np.isfinite(result.hedge_ratio)
        assert np.isfinite(result.spread)

    def test_multiple_updates(self):
        """Test multiple filter updates."""
        kf = KalmanHedgeRatio()

        # Simulate cointegrated pair with known relationship: A = 2*B + noise
        np.random.seed(42)
        n = 100
        prices_b = np.cumsum(np.random.randn(n)) + 100
        prices_a = 2.0 * prices_b + np.random.randn(n) * 0.5 + 10

        results = []
        for i in range(n):
            result = kf.update(prices_a[i], prices_b[i])
            results.append(result)

        # After warmup, hedge ratio should converge towards 2.0
        final_result = results[-1]
        assert final_result.n_observations == n
        # Allow some tolerance
        assert abs(final_result.hedge_ratio - 2.0) < 0.5

    def test_reset(self):
        """Test filter reset."""
        kf = KalmanHedgeRatio()

        # Process some data
        for _ in range(10):
            kf.update(100.0, 50.0)

        assert kf.get_state().n_updates == 10

        # Reset
        kf.reset()

        state = kf.get_state()
        assert state.n_updates == 0
        assert state.beta == 1.0

    def test_get_hedge_ratio(self):
        """Test get_hedge_ratio convenience method."""
        kf = KalmanHedgeRatio()
        kf.update(100.0, 50.0)

        ratio = kf.get_hedge_ratio()
        assert np.isfinite(ratio)
        assert ratio == kf.get_state().beta

    def test_get_spread(self):
        """Test spread calculation."""
        kf = KalmanHedgeRatio(include_intercept=False)

        # Process data to get a hedge ratio
        kf.update(100.0, 50.0)
        kf.update(102.0, 51.0)

        beta = kf.get_hedge_ratio()
        spread = kf.get_spread(104.0, 52.0)

        expected_spread = 104.0 - beta * 52.0
        assert abs(spread - expected_spread) < 0.01

    def test_zscore_calculation(self):
        """Test z-score calculation."""
        kf = KalmanHedgeRatio()

        # Process enough data to have spread statistics
        np.random.seed(42)
        for i in range(50):
            price_a = 100 + np.random.randn() * 2
            price_b = 50 + np.random.randn() * 1
            kf.update(price_a, price_b)

        zscore = kf.get_zscore(105.0, 52.0)
        assert np.isfinite(zscore)

    def test_warmup_period(self):
        """Test warmup period detection."""
        kf = KalmanHedgeRatio(warmup_period=30)

        for i in range(25):
            result = kf.update(100.0 + i, 50.0 + i * 0.5)
            assert not result.is_stable  # Not warmed up yet

        for i in range(10):
            result = kf.update(125.0 + i, 62.5 + i * 0.5)

        # After warmup, should be more stable
        assert result.n_observations >= 30

    def test_process_series(self):
        """Test processing full price series."""
        kf = KalmanHedgeRatio()

        np.random.seed(42)
        n = 100
        prices_a = np.cumsum(np.random.randn(n)) + 100
        prices_b = np.cumsum(np.random.randn(n)) + 50

        results = kf.process_series(prices_a, prices_b)

        assert len(results) == n
        assert all(isinstance(r, KalmanResult) for r in results)
        assert results[-1].n_observations == n

    def test_get_spread_series(self):
        """Test getting spread, hedge ratio, and z-score series."""
        kf = KalmanHedgeRatio()

        np.random.seed(42)
        n = 100
        prices_a = np.cumsum(np.random.randn(n)) + 100
        prices_b = np.cumsum(np.random.randn(n)) + 50

        spreads, hedge_ratios, zscores = kf.get_spread_series(prices_a, prices_b)

        assert len(spreads) == n
        assert len(hedge_ratios) == n
        assert len(zscores) == n
        assert all(np.isfinite(spreads))
        assert all(np.isfinite(hedge_ratios))

    def test_compare_with_ols(self):
        """Test comparison with rolling OLS."""
        kf = KalmanHedgeRatio()

        np.random.seed(42)
        n = 200
        prices_b = np.cumsum(np.random.randn(n)) + 100
        prices_a = 1.5 * prices_b + np.random.randn(n) * 2 + 20

        comparison = kf.compare_with_ols(prices_a, prices_b, ols_window=60)

        assert "kalman_beta_mean" in comparison
        assert "ols_beta_mean" in comparison
        assert "stability_improvement_pct" in comparison
        assert comparison["n_observations"] > 0

    def test_get_hedge_ratio_with_confidence(self):
        """Test confidence interval calculation."""
        kf = KalmanHedgeRatio()

        # Process some data
        np.random.seed(42)
        for i in range(50):
            kf.update(100.0 + np.random.randn(), 50.0 + np.random.randn() * 0.5)

        beta, lower, upper = kf.get_hedge_ratio_with_confidence(confidence=0.95)

        assert lower < beta < upper
        assert np.isfinite(lower)
        assert np.isfinite(upper)

    def test_stability_score(self):
        """Test stability score calculation."""
        kf = KalmanHedgeRatio()

        # Stable relationship
        np.random.seed(42)
        prices_b = np.arange(50, 150, dtype=float)
        prices_a = 2.0 * prices_b + 10  # Perfect linear relationship

        results = kf.process_series(prices_a, prices_b)
        final_stability = results[-1].stability_score

        # Should have high stability with perfect relationship
        assert final_stability > 0.5

    def test_adaptive_noise_estimation(self):
        """Test that R (measurement noise) is adaptively estimated."""
        kf = KalmanHedgeRatio(adaptive_noise=True)

        initial_status = kf.get_status()
        initial_r = initial_status["measurement_noise_R"]

        # Process noisy data
        np.random.seed(42)
        for i in range(50):
            # Add significant noise
            kf.update(100.0 + np.random.randn() * 10, 50.0 + np.random.randn() * 5)

        final_status = kf.get_status()
        final_r = final_status["measurement_noise_R"]

        # R should have adapted
        assert final_r != initial_r


class TestMultiPairKalmanFilter:
    """Tests for MultiPairKalmanFilter class."""

    def test_initialization(self):
        """Test multi-pair filter initialization."""
        mpf = MultiPairKalmanFilter()

        status = mpf.get_status()
        assert status["n_pairs"] == 0

    def test_get_or_create_filter(self):
        """Test filter creation for pairs."""
        mpf = MultiPairKalmanFilter()

        kf1 = mpf.get_or_create_filter("AAPL", "MSFT")
        kf2 = mpf.get_or_create_filter("AAPL", "MSFT")  # Same pair
        kf3 = mpf.get_or_create_filter("GOOGL", "META")  # Different pair

        assert kf1 is kf2  # Same instance
        assert kf1 is not kf3
        assert mpf.get_status()["n_pairs"] == 2

    def test_update(self):
        """Test updating a pair."""
        mpf = MultiPairKalmanFilter()

        result = mpf.update("AAPL", "MSFT", 150.0, 350.0)

        assert isinstance(result, KalmanResult)
        assert result.n_observations == 1

    def test_get_hedge_ratio(self):
        """Test getting hedge ratio for a pair."""
        mpf = MultiPairKalmanFilter()

        # Unknown pair
        assert mpf.get_hedge_ratio("AAPL", "MSFT") is None

        # After update
        mpf.update("AAPL", "MSFT", 150.0, 350.0)
        ratio = mpf.get_hedge_ratio("AAPL", "MSFT")

        assert ratio is not None
        assert np.isfinite(ratio)

    def test_get_all_states(self):
        """Test getting states for all pairs."""
        mpf = MultiPairKalmanFilter()

        mpf.update("AAPL", "MSFT", 150.0, 350.0)
        mpf.update("GOOGL", "META", 140.0, 500.0)

        states = mpf.get_all_states()

        assert len(states) == 2
        assert "AAPL:MSFT" in states
        assert "GOOGL:META" in states

    def test_reset_pair(self):
        """Test resetting a single pair."""
        mpf = MultiPairKalmanFilter()

        for i in range(10):
            mpf.update("AAPL", "MSFT", 150.0 + i, 350.0 + i)

        states_before = mpf.get_all_states()
        assert states_before["AAPL:MSFT"].n_updates == 10

        mpf.reset_pair("AAPL", "MSFT")

        states_after = mpf.get_all_states()
        assert states_after["AAPL:MSFT"].n_updates == 0

    def test_reset_all(self):
        """Test resetting all pairs."""
        mpf = MultiPairKalmanFilter()

        mpf.update("AAPL", "MSFT", 150.0, 350.0)
        mpf.update("GOOGL", "META", 140.0, 500.0)

        mpf.reset_all()

        states = mpf.get_all_states()
        assert all(s.n_updates == 0 for s in states.values())

    def test_remove_pair(self):
        """Test removing a pair."""
        mpf = MultiPairKalmanFilter()

        mpf.update("AAPL", "MSFT", 150.0, 350.0)
        mpf.update("GOOGL", "META", 140.0, 500.0)

        assert mpf.get_status()["n_pairs"] == 2

        mpf.remove_pair("AAPL", "MSFT")

        assert mpf.get_status()["n_pairs"] == 1
        assert "AAPL:MSFT" not in mpf.get_status()["pairs"]


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_kalman_filter_default(self):
        """Test creating filter with default config."""
        kf = create_kalman_filter()

        assert isinstance(kf, KalmanHedgeRatio)
        assert kf.get_state().beta == 1.0

    def test_create_kalman_filter_custom(self):
        """Test creating filter with custom config."""
        config = {
            "delta": 1e-3,
            "ve": 1e-2,
            "include_intercept": False,
            "warmup_period": 50,
        }

        kf = create_kalman_filter(config)

        # Check that config was applied
        status = kf.get_status()
        assert status["process_noise_delta"] == 1e-3

    def test_create_multi_pair_filter_default(self):
        """Test creating multi-pair filter with default config."""
        mpf = create_multi_pair_filter()

        assert isinstance(mpf, MultiPairKalmanFilter)
        assert mpf.get_status()["n_pairs"] == 0

    def test_create_multi_pair_filter_custom(self):
        """Test creating multi-pair filter with custom config."""
        config = {
            "delta": 1e-5,
            "warmup_period": 100,
        }

        mpf = create_multi_pair_filter(config)

        assert isinstance(mpf, MultiPairKalmanFilter)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_prices(self):
        """Test handling of zero prices."""
        kf = KalmanHedgeRatio()

        # This shouldn't crash
        result = kf.update(0.0, 50.0)
        assert np.isfinite(result.spread)

        result = kf.update(100.0, 0.0)
        # With zero price_b, spread calculation might be unusual
        assert np.isfinite(result.spread)

    def test_negative_prices(self):
        """Test handling of negative prices (e.g., spreads)."""
        kf = KalmanHedgeRatio()

        result = kf.update(-5.0, -2.0)
        assert np.isfinite(result.hedge_ratio)

    def test_constant_prices(self):
        """Test handling of constant prices."""
        kf = KalmanHedgeRatio()

        for _ in range(20):
            result = kf.update(100.0, 50.0)

        # With constant prices, beta should stay near initial
        assert np.isfinite(result.hedge_ratio)

    def test_very_large_prices(self):
        """Test handling of very large prices."""
        kf = KalmanHedgeRatio()

        result = kf.update(1e10, 5e9)
        assert np.isfinite(result.hedge_ratio)
        assert np.isfinite(result.spread)

    def test_mismatched_series_length(self):
        """Test that mismatched series lengths raise error."""
        kf = KalmanHedgeRatio()

        prices_a = np.array([100, 101, 102])
        prices_b = np.array([50, 51])  # Different length

        with pytest.raises(ValueError):
            kf.process_series(prices_a, prices_b)

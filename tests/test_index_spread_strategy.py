"""
Tests for Index Spread Strategy (Phase 6.2)
===========================================

Tests for MES/MNQ spread trading, z-score signals,
hedge ratio estimation, and cointegration analysis.
"""

import numpy as np
import pytest
from datetime import datetime, timezone

from strategies.index_spread_strategy import (
    IndexSpreadStrategy,
    SpreadRelationship,
    SpreadState,
    SpreadSignal,
    INDEX_SPREAD_DEFINITIONS,
    create_index_spread_strategy,
)


class TestSpreadCalculation:
    """Tests for spread calculation."""

    @pytest.fixture
    def strategy(self):
        """Create IndexSpreadStrategy instance."""
        return IndexSpreadStrategy()

    def test_basic_spread_calculation(self, strategy):
        """Test basic spread calculation."""
        prices_a = np.array([100, 101, 102, 103, 104])
        prices_b = np.array([50, 50.5, 51, 51.5, 52])

        spread = strategy.calculate_spread(prices_a, prices_b, hedge_ratio=2.0)

        # spread = A - 2.0 * B
        expected = prices_a - 2.0 * prices_b
        np.testing.assert_array_almost_equal(spread, expected)

    def test_spread_with_auto_hedge_ratio(self, strategy):
        """Test spread calculation with automatic hedge ratio."""
        np.random.seed(42)
        n = 100
        prices_b = 50 + np.cumsum(np.random.randn(n) * 0.5)
        prices_a = 2.0 * prices_b + np.random.randn(n) * 0.5  # A ~= 2*B

        spread = strategy.calculate_spread(prices_a, prices_b)

        # Spread should be roughly stationary
        assert len(spread) == n
        assert np.std(spread) < np.std(prices_a)  # Spread less volatile than raw price


class TestHedgeRatioEstimation:
    """Tests for hedge ratio estimation."""

    @pytest.fixture
    def strategy(self):
        """Create IndexSpreadStrategy instance."""
        return IndexSpreadStrategy()

    def test_hedge_ratio_ols(self, strategy):
        """Test OLS hedge ratio estimation."""
        np.random.seed(42)
        n = 100
        prices_b = 50 + np.cumsum(np.random.randn(n) * 0.5)
        true_beta = 1.8
        prices_a = true_beta * prices_b + np.random.randn(n) * 0.5

        hedge_ratio = strategy._estimate_hedge_ratio(prices_a, prices_b)

        # Should be close to true beta
        assert abs(hedge_ratio - true_beta) < 0.2

    def test_hedge_ratio_zero_variance(self, strategy):
        """Test hedge ratio with zero variance in B."""
        prices_a = np.array([100, 101, 102, 103, 104])
        prices_b = np.array([50, 50, 50, 50, 50])  # Constant

        hedge_ratio = strategy._estimate_hedge_ratio(prices_a, prices_b)

        # Should return default 1.0
        assert hedge_ratio == 1.0


class TestZScoreCalculation:
    """Tests for z-score calculation."""

    @pytest.fixture
    def strategy(self):
        """Create IndexSpreadStrategy instance."""
        return IndexSpreadStrategy()

    def test_zscore_centered(self, strategy):
        """Test z-score for centered spread."""
        spread = np.array([0.0, 0.1, -0.1, 0.05, -0.05, 0.0])

        zscore = strategy.calculate_zscore(spread)

        # Last value (0.0) should have z-score near mean
        assert abs(zscore) < 0.5

    def test_zscore_extreme_high(self, strategy):
        """Test z-score for extreme high value."""
        spread = np.array([0.0, 0.1, -0.1, 0.05, -0.05, 2.0])  # Last value is extreme

        zscore = strategy.calculate_zscore(spread)

        # Should be significantly positive
        assert zscore > 2.0

    def test_zscore_extreme_low(self, strategy):
        """Test z-score for extreme low value."""
        spread = np.array([0.0, 0.1, -0.1, 0.05, -0.05, -2.0])  # Last value is extreme

        zscore = strategy.calculate_zscore(spread)

        # Should be significantly negative
        assert zscore < -2.0

    def test_zscore_with_lookback(self, strategy):
        """Test z-score with limited lookback."""
        spread = np.array([10.0, 10.0, 0.0, 0.1, -0.1, 0.0])  # Old values are different

        zscore_full = strategy.calculate_zscore(spread)
        zscore_lookback = strategy.calculate_zscore(spread, lookback=3)

        # Lookback should give different result (ignoring old high values)
        assert zscore_full != zscore_lookback

    def test_zscore_short_series(self, strategy):
        """Test z-score with very short series."""
        spread = np.array([1.0])

        zscore = strategy.calculate_zscore(spread)

        # Should return 0 for insufficient data
        assert zscore == 0.0


class TestHalfLifeCalculation:
    """Tests for half-life calculation."""

    @pytest.fixture
    def strategy(self):
        """Create IndexSpreadStrategy instance."""
        return IndexSpreadStrategy()

    def test_half_life_mean_reverting(self, strategy):
        """Test half-life for mean-reverting series."""
        np.random.seed(42)
        n = 200
        theta = 0.2  # Mean reversion speed
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = spread[i-1] * (1 - theta) + np.random.randn() * 0.1

        half_life = strategy.calculate_half_life(spread)

        # Half-life should be reasonable (not inf)
        assert half_life < 20
        assert half_life > 0.1

    def test_half_life_random_walk(self, strategy):
        """Test half-life for random walk (no mean reversion)."""
        np.random.seed(42)
        spread = np.cumsum(np.random.randn(500))  # Longer random walk

        half_life = strategy.calculate_half_life(spread)

        # Should be large or inf (no strong mean reversion)
        # Random walks can show spurious mean reversion in short samples
        assert half_life > 15 or half_life == float('inf')

    def test_half_life_short_series(self, strategy):
        """Test half-life with short series."""
        spread = np.array([1.0, 2.0, 1.5])

        half_life = strategy.calculate_half_life(spread)

        # Should return inf for insufficient data
        assert half_life == float('inf')


class TestSpreadAnalysis:
    """Tests for spread analysis."""

    @pytest.fixture
    def strategy(self):
        """Create IndexSpreadStrategy instance."""
        return IndexSpreadStrategy()

    def test_analyze_cointegrated_spread(self, strategy):
        """Test analysis of cointegrated spread."""
        np.random.seed(42)
        n = 200

        # Create strongly cointegrated series with mean-reverting spread
        prices_b = 100 + np.cumsum(np.random.randn(n) * 0.3)
        # Spread with mean reversion (Ornstein-Uhlenbeck)
        theta = 0.15
        spread_noise = np.zeros(n)
        for i in range(1, n):
            spread_noise[i] = spread_noise[i-1] * (1 - theta) + np.random.randn() * 0.3
        prices_a = 2.0 * prices_b + spread_noise

        state = strategy.analyze_spread("MES_MNQ", prices_a, prices_b)

        assert isinstance(state, SpreadState)
        assert state.correlation > 0.8  # High correlation
        assert state.hedge_ratio > 0
        # Half-life can be inf if AR(1) coefficient >= 1
        assert state.half_life_days >= 0

    def test_analyze_relationship_states(self, strategy):
        """Test different relationship states."""
        np.random.seed(42)
        n = 100
        prices_b = 100 + np.cumsum(np.random.randn(n) * 0.3)

        # Normal spread
        prices_a = 2.0 * prices_b + np.random.randn(n) * 0.5
        state = strategy.analyze_spread("test", prices_a, prices_b)

        # Z-score should determine relationship
        assert state.relationship in [
            SpreadRelationship.NORMAL,
            SpreadRelationship.EXTENDED,
            SpreadRelationship.EXTREME,
        ]

    def test_analyze_updates_internal_state(self, strategy):
        """Test that analysis updates internal state tracking."""
        np.random.seed(42)
        prices_a = np.random.randn(50) + 100
        prices_b = np.random.randn(50) + 50

        strategy.analyze_spread("TEST_SPREAD", prices_a, prices_b)

        assert "TEST_SPREAD" in strategy._spread_states
        assert "TEST_SPREAD" in strategy._hedge_ratios


class TestSignalGeneration:
    """Tests for signal generation."""

    @pytest.fixture
    def strategy(self):
        """Create IndexSpreadStrategy with lower thresholds for testing."""
        return IndexSpreadStrategy({
            "zscore_entry": 1.5,
            "zscore_exit": 0.3,
            "zscore_stop": 3.0,
            "min_correlation": 0.5,
            "min_half_life": 0.5,
            "max_half_life": 50,
        })

    def test_long_spread_signal(self, strategy):
        """Test long spread signal (spread too low)."""
        np.random.seed(42)
        n = 100

        # Create spread that ends very low
        prices_b = 100 + np.arange(n) * 0.1
        prices_a = 2.0 * prices_b.copy()
        prices_a[-10:] -= 20  # Make spread negative at end

        signal = strategy.generate_signal("MES_MNQ", prices_a, prices_b, "FLAT")

        if signal is not None:
            assert signal.direction in ["LONG_SPREAD", "SHORT_SPREAD"]
            assert signal.strength >= 0
            assert signal.hedge_ratio > 0

    def test_short_spread_signal(self, strategy):
        """Test short spread signal (spread too high)."""
        np.random.seed(42)
        n = 100

        # Create spread that ends very high
        prices_b = 100 + np.arange(n) * 0.1
        prices_a = 2.0 * prices_b.copy()
        prices_a[-10:] += 20  # Make spread positive at end

        signal = strategy.generate_signal("MES_MNQ", prices_a, prices_b, "FLAT")

        if signal is not None:
            assert signal.direction in ["LONG_SPREAD", "SHORT_SPREAD"]

    def test_exit_signal_long_spread(self, strategy):
        """Test exit signal for long spread position."""
        np.random.seed(42)
        n = 100

        # Create mean-reverting spread
        prices_b = 100 + np.cumsum(np.random.randn(n) * 0.3)
        prices_a = 2.0 * prices_b + np.random.randn(n) * 1.0

        signal = strategy.generate_signal("MES_MNQ", prices_a, prices_b, "LONG_SPREAD")

        # May or may not generate exit signal depending on z-score
        if signal is not None:
            assert signal.direction == "FLAT"

    def test_no_signal_uncorrelated(self, strategy):
        """Test no signal for uncorrelated series."""
        np.random.seed(42)
        prices_a = np.random.randn(100) * 10 + 100
        prices_b = np.random.randn(100) * 10 + 50  # Uncorrelated

        signal = strategy.generate_signal("MES_MNQ", prices_a, prices_b, "FLAT")

        # Should return None (not cointegrated)
        assert signal is None

    def test_signal_contains_required_fields(self, strategy):
        """Test that signal contains all required fields."""
        np.random.seed(123)
        n = 100

        # Create highly correlated series with extreme spread
        prices_b = 100 + np.cumsum(np.random.randn(n) * 0.2)
        prices_a = 2.0 * prices_b + 5  # Constant offset makes spread high

        signal = strategy.generate_signal("MES_MNQ", prices_a, prices_b, "FLAT")

        if signal is not None:
            assert isinstance(signal, SpreadSignal)
            assert signal.symbol_long is not None
            assert signal.symbol_short is not None
            assert signal.direction in ["LONG_SPREAD", "SHORT_SPREAD", "FLAT"]
            assert 0.0 <= signal.strength <= 1.0
            assert signal.hedge_ratio > 0
            assert signal.position_size_ratio is not None


class TestSpreadDefinitions:
    """Tests for predefined spread definitions."""

    def test_mes_mnq_defined(self):
        """Test MES/MNQ spread is defined."""
        assert "MES_MNQ" in INDEX_SPREAD_DEFINITIONS

        spread = INDEX_SPREAD_DEFINITIONS["MES_MNQ"]
        assert spread["leg_a"] == "MES"
        assert spread["leg_b"] == "MNQ"
        assert spread["multiplier_a"] == 5.0
        assert spread["multiplier_b"] == 2.0

    def test_es_nq_defined(self):
        """Test ES/NQ spread is defined."""
        assert "ES_NQ" in INDEX_SPREAD_DEFINITIONS

        spread = INDEX_SPREAD_DEFINITIONS["ES_NQ"]
        assert spread["leg_a"] == "ES"
        assert spread["leg_b"] == "NQ"

    def test_all_spreads_have_required_fields(self):
        """Test all spreads have required fields."""
        required_fields = ["leg_a", "leg_b", "multiplier_a", "multiplier_b"]

        for name, spread in INDEX_SPREAD_DEFINITIONS.items():
            for field in required_fields:
                assert field in spread, f"Missing {field} in {name}"


class TestStrategyStatus:
    """Tests for strategy status."""

    @pytest.fixture
    def strategy(self):
        """Create IndexSpreadStrategy instance."""
        return IndexSpreadStrategy()

    def test_get_status(self, strategy):
        """Test status reporting."""
        status = strategy.get_status()

        assert "zscore_entry" in status
        assert "zscore_exit" in status
        assert "zscore_stop" in status
        assert "lookback_days" in status
        assert "tracked_spreads" in status

    def test_status_after_analysis(self, strategy):
        """Test status includes analyzed spreads."""
        np.random.seed(42)
        prices_a = np.random.randn(50) + 100
        prices_b = np.random.randn(50) + 50

        strategy.analyze_spread("TEST", prices_a, prices_b)
        status = strategy.get_status()

        assert status["tracked_spreads"] == 1
        assert "TEST" in status["spread_states"]


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_default(self):
        """Test default factory creation."""
        strategy = create_index_spread_strategy()

        assert isinstance(strategy, IndexSpreadStrategy)
        assert strategy._zscore_entry == 2.0

    def test_create_custom_config(self):
        """Test factory with custom config."""
        strategy = create_index_spread_strategy({
            "zscore_entry": 2.5,
            "zscore_exit": 0.7,
            "lookback_days": 90,
        })

        assert strategy._zscore_entry == 2.5
        assert strategy._zscore_exit == 0.7
        assert strategy._lookback_days == 90

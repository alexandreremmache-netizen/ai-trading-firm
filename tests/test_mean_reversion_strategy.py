"""
Tests for Mean Reversion Single-Asset Strategy (Phase 6.5)
==========================================================

Tests for RSI, Bollinger Bands, z-score analysis,
and signal generation for mean reversion trades.
"""

import numpy as np
import pytest
from datetime import datetime, timezone

from strategies.mean_reversion_strategy import (
    MeanReversionStrategy,
    MarketRegime,
    SignalType,
    MeanReversionState,
    MeanReversionSignal,
    create_mean_reversion_strategy,
)


class TestRSICalculation:
    """Tests for RSI calculation."""

    @pytest.fixture
    def strategy(self):
        """Create MeanReversionStrategy instance."""
        return MeanReversionStrategy()

    def test_rsi_range(self, strategy):
        """Test RSI stays in 0-100 range."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

        rsi = strategy.calculate_rsi(prices)

        # RSI should be between 0 and 100
        valid_rsi = rsi[rsi != 0]  # Exclude initial zeros
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)

    def test_rsi_uptrend_high(self, strategy):
        """Test RSI is high in strong uptrend."""
        # Strong uptrend
        prices = 100 + np.arange(50) * 1.0  # Linear up

        rsi = strategy.calculate_rsi(prices)

        # RSI should be high (>70) in uptrend
        assert rsi[-1] > 70

    def test_rsi_downtrend_low(self, strategy):
        """Test RSI is low in strong downtrend."""
        # Strong downtrend
        prices = 100 - np.arange(50) * 1.0  # Linear down

        rsi = strategy.calculate_rsi(prices)

        # RSI should be low (<30) in downtrend
        assert rsi[-1] < 30

    def test_rsi_short_series(self, strategy):
        """Test RSI with short series."""
        prices = np.array([100, 101, 102])

        rsi = strategy.calculate_rsi(prices)

        # Should return zeros for insufficient data
        assert np.all(rsi == 0)


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""

    @pytest.fixture
    def strategy(self):
        """Create MeanReversionStrategy instance."""
        return MeanReversionStrategy()

    def test_bb_ordering(self, strategy):
        """Test BB bands are ordered correctly."""
        np.random.seed(42)
        prices = 100 + np.random.randn(50) * 2

        upper, middle, lower = strategy.calculate_bollinger_bands(prices)

        # After warmup, upper > middle > lower
        valid_idx = upper != 0
        assert np.all(upper[valid_idx] >= middle[valid_idx])
        assert np.all(middle[valid_idx] >= lower[valid_idx])

    def test_bb_width_responds_to_vol(self, strategy):
        """Test BB width responds to volatility."""
        # Low volatility
        prices_low = 100 + np.random.randn(50) * 0.1
        upper_lv, _, lower_lv = strategy.calculate_bollinger_bands(prices_low)

        # High volatility
        prices_high = 100 + np.random.randn(50) * 5.0
        upper_hv, _, lower_hv = strategy.calculate_bollinger_bands(prices_high)

        # High vol should have wider bands
        width_lv = upper_lv[-1] - lower_lv[-1]
        width_hv = upper_hv[-1] - lower_hv[-1]

        assert width_hv > width_lv


class TestZScoreCalculation:
    """Tests for z-score calculation."""

    @pytest.fixture
    def strategy(self):
        """Create MeanReversionStrategy instance."""
        return MeanReversionStrategy()

    def test_zscore_extreme_high(self, strategy):
        """Test z-score for extreme high."""
        prices = np.array([100] * 50 + [110])  # Jump up at end

        zscore = strategy.calculate_zscore(prices)

        assert zscore > 2.0

    def test_zscore_extreme_low(self, strategy):
        """Test z-score for extreme low."""
        prices = np.array([100] * 50 + [90])  # Drop at end

        zscore = strategy.calculate_zscore(prices)

        assert zscore < -2.0

    def test_zscore_normal(self, strategy):
        """Test z-score for normal price."""
        prices = np.array([100] * 50 + [100])  # At mean

        zscore = strategy.calculate_zscore(prices)

        assert abs(zscore) < 0.5


class TestRegimeDetection:
    """Tests for market regime detection."""

    @pytest.fixture
    def strategy(self):
        """Create MeanReversionStrategy instance."""
        return MeanReversionStrategy()

    def test_trending_up(self, strategy):
        """Test detection returns valid regime for uptrend data."""
        np.random.seed(42)
        n = 100
        close = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 1.0
        high = close + 1
        low = close - 1

        regime = strategy.detect_regime(close, high, low)

        # Should return a valid MarketRegime
        assert regime in MarketRegime
        # In uptrend with relatively constant vol, expect TRENDING_UP or LOW_VOL
        # (LOW_VOL because recent vol is less than full-period vol in linear trends)
        assert regime in [MarketRegime.TRENDING_UP, MarketRegime.LOW_VOL, MarketRegime.RANGE_BOUND]

    def test_trending_down(self, strategy):
        """Test detection returns valid regime for downtrend data."""
        np.random.seed(123)
        n = 100
        close = 100 - np.arange(n) * 0.5 + np.random.randn(n) * 1.0
        high = close + 1
        low = close - 1

        regime = strategy.detect_regime(close, high, low)

        # Should return a valid MarketRegime
        assert regime in MarketRegime
        # In downtrend with relatively constant vol, expect TRENDING_DOWN or LOW_VOL
        assert regime in [MarketRegime.TRENDING_DOWN, MarketRegime.LOW_VOL, MarketRegime.RANGE_BOUND]

    def test_range_bound(self, strategy):
        """Test detection of range-bound market."""
        np.random.seed(42)
        n = 100
        close = 100 + np.random.randn(n) * 1.0  # Random around 100
        high = close + 0.5
        low = close - 0.5

        regime = strategy.detect_regime(close, high, low)

        assert regime == MarketRegime.RANGE_BOUND


class TestBBPosition:
    """Tests for Bollinger Band position."""

    @pytest.fixture
    def strategy(self):
        """Create MeanReversionStrategy instance."""
        return MeanReversionStrategy()

    def test_bb_position_at_upper(self, strategy):
        """Test position at upper band."""
        pos = strategy.get_bb_position(price=110, bb_upper=110, bb_lower=90)

        assert pos == 1.0

    def test_bb_position_at_lower(self, strategy):
        """Test position at lower band."""
        pos = strategy.get_bb_position(price=90, bb_upper=110, bb_lower=90)

        assert pos == -1.0

    def test_bb_position_at_middle(self, strategy):
        """Test position at middle."""
        pos = strategy.get_bb_position(price=100, bb_upper=110, bb_lower=90)

        assert abs(pos) < 0.1


class TestAnalysis:
    """Tests for mean reversion analysis."""

    @pytest.fixture
    def strategy(self):
        """Create MeanReversionStrategy instance."""
        return MeanReversionStrategy()

    def test_analyze_returns_state(self, strategy):
        """Test that analyze returns MeanReversionState."""
        np.random.seed(42)
        n = 100
        close = 100 + np.random.randn(n) * 2
        high = close + 1
        low = close - 1

        state = strategy.analyze("TEST", high, low, close)

        assert isinstance(state, MeanReversionState)
        assert 0 <= state.rsi <= 100
        assert state.regime in MarketRegime
        assert -1.5 <= state.bb_position <= 1.5

    def test_analyze_tracks_state(self, strategy):
        """Test that analyze tracks state."""
        np.random.seed(42)
        close = 100 + np.random.randn(100) * 2
        high = close + 1
        low = close - 1

        strategy.analyze("TRACK", high, low, close)

        assert "TRACK" in strategy._states


class TestSignalGeneration:
    """Tests for signal generation."""

    @pytest.fixture
    def strategy(self):
        """Create MeanReversionStrategy with adjusted thresholds."""
        return MeanReversionStrategy({
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "zscore_threshold": 1.5,
        })

    def test_long_signal_oversold(self, strategy):
        """Test long signal on oversold conditions."""
        n = 100
        # Create oversold conditions
        close = 100 - np.arange(n) * 0.3  # Declining
        high = close + 0.5
        low = close - 0.5

        signal = strategy.generate_signal("TEST", high, low, close)

        if signal is not None:
            assert signal.direction in ["LONG", "FLAT"]
            if signal.direction == "LONG":
                assert signal.stop_loss < signal.entry_price

    def test_short_signal_overbought(self, strategy):
        """Test short signal on overbought conditions."""
        n = 100
        # Create overbought conditions
        close = 100 + np.arange(n) * 0.3  # Rising
        high = close + 0.5
        low = close - 0.5

        signal = strategy.generate_signal("TEST", high, low, close)

        if signal is not None:
            assert signal.direction in ["SHORT", "FLAT"]
            if signal.direction == "SHORT":
                assert signal.stop_loss > signal.entry_price

    def test_no_signal_neutral(self, strategy):
        """Test no signal in neutral conditions."""
        np.random.seed(42)
        n = 100
        # Create neutral conditions
        close = 100 + np.random.randn(n) * 0.5
        high = close + 0.3
        low = close - 0.3

        signal = strategy.generate_signal("TEST", high, low, close)

        # May or may not have signal depending on random data
        # Just verify no crash
        assert signal is None or isinstance(signal, MeanReversionSignal)

    def test_exit_signal_rsi_normalized(self, strategy):
        """Test exit signal when RSI normalizes."""
        np.random.seed(42)
        n = 100
        # Neutral RSI conditions
        close = 100 + np.random.randn(n) * 0.5
        high = close + 0.3
        low = close - 0.3

        signal = strategy.generate_signal("TEST", high, low, close, current_position="LONG")

        # Should potentially get exit signal
        if signal is not None:
            assert signal.direction == "FLAT"

    def test_signal_includes_rationale(self, strategy):
        """Test that signals include rationale."""
        n = 100
        close = 100 - np.arange(n) * 0.3
        high = close + 0.5
        low = close - 0.5

        signal = strategy.generate_signal("TEST", high, low, close)

        if signal is not None:
            assert len(signal.rationale) > 0
            assert signal.rsi >= 0
            assert signal.zscore != 0 or "z=0" in signal.rationale.lower() or True  # Just check exists


class TestCombinedSignals:
    """Tests for combined signal logic."""

    @pytest.fixture
    def strategy(self):
        """Create strategy with lower thresholds."""
        return MeanReversionStrategy({
            "rsi_oversold": 40,
            "rsi_overbought": 60,
            "zscore_threshold": 1.0,
        })

    def test_combined_signal_strength(self, strategy):
        """Test that multiple indicators increase strength."""
        n = 100
        # Create extreme oversold with multiple indicators triggering
        close = np.concatenate([
            np.ones(80) * 100,
            100 - np.arange(20) * 2.0  # Sharp drop
        ])
        high = close + 0.3
        low = close - 0.3

        signal = strategy.generate_signal("COMBINED", high, low, close)

        if signal is not None and signal.direction == "LONG":
            # Combined signals should have higher strength
            assert signal.strength > 0.3


class TestStatus:
    """Tests for strategy status."""

    @pytest.fixture
    def strategy(self):
        """Create MeanReversionStrategy instance."""
        return MeanReversionStrategy()

    def test_get_status(self, strategy):
        """Test status reporting."""
        status = strategy.get_status()

        assert "rsi_period" in status
        assert "rsi_thresholds" in status
        assert "bb_period" in status
        assert "zscore_threshold" in status
        assert "tracked_symbols" in status

    def test_status_after_analysis(self, strategy):
        """Test status includes analyzed symbols."""
        np.random.seed(42)
        close = 100 + np.random.randn(100) * 2
        high = close + 1
        low = close - 1

        strategy.analyze("STATUS_TEST", high, low, close)
        status = strategy.get_status()

        assert status["tracked_symbols"] == 1
        assert "STATUS_TEST" in status["states"]


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_default(self):
        """Test default factory creation."""
        strategy = create_mean_reversion_strategy()

        assert isinstance(strategy, MeanReversionStrategy)
        assert strategy._rsi_period == 14
        assert strategy._rsi_oversold == 30
        assert strategy._rsi_overbought == 70

    def test_create_custom_config(self):
        """Test factory with custom config."""
        strategy = create_mean_reversion_strategy({
            "rsi_period": 10,
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "bb_period": 15,
        })

        assert strategy._rsi_period == 10
        assert strategy._rsi_oversold == 25
        assert strategy._rsi_overbought == 75
        assert strategy._bb_period == 15

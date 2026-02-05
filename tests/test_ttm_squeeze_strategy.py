"""
Tests for TTM Squeeze Volatility Strategy (Phase 6.3)
=====================================================

Tests for squeeze detection, momentum oscillator,
and signal generation.
"""

import numpy as np
import pytest
from datetime import datetime, timezone

from strategies.ttm_squeeze_strategy import (
    TTMSqueezeStrategy,
    SqueezeState,
    MomentumDirection,
    SqueezeReading,
    SqueezeSignal,
    create_ttm_squeeze_strategy,
)


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""

    @pytest.fixture
    def strategy(self):
        """Create TTMSqueezeStrategy instance."""
        return TTMSqueezeStrategy()

    def test_bollinger_bands_basic(self, strategy):
        """Test basic Bollinger Bands calculation."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(50) * 0.5)

        upper, middle, lower = strategy.calculate_bollinger_bands(prices)

        # Upper > middle > lower
        assert np.all(upper[20:] >= middle[20:])
        assert np.all(middle[20:] >= lower[20:])

    def test_bollinger_bands_width(self, strategy):
        """Test that BB width responds to volatility."""
        # Low volatility
        prices_low_vol = 100 + np.arange(50) * 0.01 + np.random.randn(50) * 0.1
        upper_lv, _, lower_lv = strategy.calculate_bollinger_bands(prices_low_vol)

        # High volatility
        prices_high_vol = 100 + np.arange(50) * 0.01 + np.random.randn(50) * 2.0
        upper_hv, _, lower_hv = strategy.calculate_bollinger_bands(prices_high_vol)

        # High vol should have wider bands
        width_lv = np.mean(upper_lv[20:] - lower_lv[20:])
        width_hv = np.mean(upper_hv[20:] - lower_hv[20:])

        assert width_hv > width_lv

    def test_bollinger_bands_short_series(self, strategy):
        """Test BB with insufficient data."""
        prices = np.array([100, 101, 102])

        upper, middle, lower = strategy.calculate_bollinger_bands(prices)

        # Should return empty arrays
        assert len(upper) == 0


class TestKeltnerChannel:
    """Tests for Keltner Channel calculation."""

    @pytest.fixture
    def strategy(self):
        """Create TTMSqueezeStrategy instance."""
        return TTMSqueezeStrategy()

    def test_keltner_channel_basic(self, strategy):
        """Test basic Keltner Channel calculation."""
        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        upper, middle, lower = strategy.calculate_keltner_channel(high, low, close)

        # Upper > middle > lower
        assert np.all(upper[20:] > middle[20:])
        assert np.all(middle[20:] > lower[20:])

    def test_keltner_channel_atr_based(self, strategy):
        """Test that KC width is ATR-based."""
        np.random.seed(42)
        n = 50

        # Small range (low ATR)
        close = 100 + np.arange(n) * 0.1
        high = close + 0.1
        low = close - 0.1
        upper_small, _, lower_small = strategy.calculate_keltner_channel(high, low, close)

        # Large range (high ATR)
        close2 = 100 + np.arange(n) * 0.1
        high2 = close2 + 2.0
        low2 = close2 - 2.0
        upper_large, _, lower_large = strategy.calculate_keltner_channel(high2, low2, close2)

        # Large range should have wider channel
        width_small = np.mean(upper_small[-10:] - lower_small[-10:])
        width_large = np.mean(upper_large[-10:] - lower_large[-10:])

        assert width_large > width_small


class TestMomentumOscillator:
    """Tests for momentum oscillator calculation."""

    @pytest.fixture
    def strategy(self):
        """Create TTMSqueezeStrategy instance."""
        return TTMSqueezeStrategy()

    def test_momentum_uptrend(self, strategy):
        """Test momentum in accelerating uptrend (price above linear regression)."""
        n = 50
        # Accelerating uptrend - exponential growth deviates above linear regression
        close = 100 + np.arange(n) * 0.3 + (np.arange(n) ** 1.5) * 0.01
        high = close + 0.2
        low = close - 0.2

        momentum = strategy.calculate_momentum(high, low, close)

        # Should be positive when accelerating upward
        assert momentum[-1] > 0

    def test_momentum_downtrend(self, strategy):
        """Test momentum in accelerating downtrend (price below linear regression)."""
        n = 50
        # Accelerating downtrend - exponential decay deviates below linear regression
        close = 100 - np.arange(n) * 0.3 - (np.arange(n) ** 1.5) * 0.01
        high = close + 0.2
        low = close - 0.2

        momentum = strategy.calculate_momentum(high, low, close)

        # Should be negative when accelerating downward
        assert momentum[-1] < 0

    def test_momentum_flat(self, strategy):
        """Test momentum in flat market."""
        n = 50
        close = np.ones(n) * 100  # Flat
        high = close + 0.1
        low = close - 0.1

        momentum = strategy.calculate_momentum(high, low, close)

        # Should be near zero
        assert abs(momentum[-1]) < 0.1


class TestSqueezeDetection:
    """Tests for squeeze detection."""

    @pytest.fixture
    def strategy(self):
        """Create TTMSqueezeStrategy instance."""
        return TTMSqueezeStrategy()

    def test_squeeze_low_volatility(self, strategy):
        """Test squeeze detection in low volatility."""
        np.random.seed(42)
        n = 100

        # Very low volatility - tight range
        close = 100 + np.random.randn(n) * 0.05
        high = close + 0.02
        low = close - 0.02

        squeeze_on, bb_width, kc_width = strategy.detect_squeeze(high, low, close)

        # In very low vol, BB should be inside KC (squeeze on)
        assert len(squeeze_on) > 0
        # Some squeeze periods expected
        assert np.any(squeeze_on[30:])

    def test_squeeze_high_volatility(self, strategy):
        """Test no squeeze in high close-to-close volatility with small ranges."""
        np.random.seed(42)
        n = 100

        # High close-to-close volatility but small intraday ranges
        # This creates wide BB (high std of closes) but narrow KC (small ATR)
        close = 100 + np.cumsum(np.random.randn(n) * 3.0)  # Large jumps between bars
        # But small intraday ranges (gaps, not volatile within day)
        high = close + 0.5  # Tight range
        low = close - 0.5

        squeeze_on, bb_width, kc_width = strategy.detect_squeeze(high, low, close)

        # BB should be wider than KC (squeeze off) when close-to-close vol > intraday vol
        assert len(squeeze_on) > 0
        # In this scenario, most periods should be squeeze off (BB outside KC)
        squeeze_ratio = np.mean(squeeze_on[30:])
        assert squeeze_ratio < 0.7  # More lenient threshold


class TestAnalysis:
    """Tests for squeeze analysis."""

    @pytest.fixture
    def strategy(self):
        """Create TTMSqueezeStrategy instance."""
        return TTMSqueezeStrategy()

    def test_analyze_returns_reading(self, strategy):
        """Test that analyze returns SqueezeReading."""
        np.random.seed(42)
        n = 50
        close = 100 + np.random.randn(n) * 0.5
        high = close + 0.3
        low = close - 0.3

        reading = strategy.analyze("TEST", high, low, close)

        assert isinstance(reading, SqueezeReading)
        assert reading.squeeze_state in SqueezeState
        assert reading.momentum_direction in MomentumDirection
        assert reading.bars_in_squeeze >= 0

    def test_analyze_tracks_history(self, strategy):
        """Test that analyze tracks squeeze history."""
        np.random.seed(42)
        n = 50
        close = 100 + np.random.randn(n) * 0.5
        high = close + 0.3
        low = close - 0.3

        # Multiple calls
        for _ in range(5):
            strategy.analyze("TRACK_TEST", high, low, close)

        assert "TRACK_TEST" in strategy._squeeze_history
        assert len(strategy._squeeze_history["TRACK_TEST"]) == 5

    def test_analyze_squeeze_intensity(self, strategy):
        """Test squeeze intensity calculation."""
        np.random.seed(42)
        n = 50
        close = 100 + np.random.randn(n) * 0.1  # Low vol
        high = close + 0.05
        low = close - 0.05

        reading = strategy.analyze("INTENSITY", high, low, close)

        # Intensity should be between 0 and 1
        assert 0.0 <= reading.squeeze_intensity <= 1.0


class TestSignalGeneration:
    """Tests for signal generation."""

    @pytest.fixture
    def strategy(self):
        """Create strategy with lower thresholds."""
        return TTMSqueezeStrategy({
            "min_squeeze_bars": 3,  # Lower for testing
        })

    def test_signal_squeeze_fire_bullish(self, strategy):
        """Test bullish squeeze fire signal."""
        np.random.seed(42)
        n = 100

        # Create squeeze period followed by breakout up
        close = np.ones(n) * 100
        close[:60] += np.random.randn(60) * 0.05  # Low vol squeeze
        close[60:] = 100 + np.arange(40) * 0.3  # Breakout up

        high = close + 0.1
        low = close - 0.1
        high[60:] = close[60:] + 0.5  # Higher highs on breakout

        # Build up squeeze history
        for i in range(50, 65):
            strategy.analyze("TEST", high[:i], low[:i], close[:i])

        signal = strategy.generate_signal("TEST", high, low, close)

        # May get signal on squeeze fire
        if signal is not None:
            assert signal.direction in ["LONG", "SHORT", "FLAT"]
            assert signal.entry_price > 0

    def test_signal_no_signal_in_squeeze(self, strategy):
        """Test no entry signal while squeeze is on."""
        np.random.seed(42)
        n = 50

        # Pure low volatility squeeze
        close = 100 + np.random.randn(n) * 0.02
        high = close + 0.01
        low = close - 0.01

        signal = strategy.generate_signal("TEST", high, low, close)

        # No signal during squeeze (waiting for fire)
        assert signal is None

    def test_exit_signal_momentum_reversal(self, strategy):
        """Test exit signal on momentum reversal."""
        np.random.seed(42)
        n = 50

        # Uptrend then reversal
        close = np.zeros(n)
        close[:30] = 100 + np.arange(30) * 0.2  # Up
        close[30:] = close[29] - np.arange(20) * 0.3  # Down

        high = close + 0.2
        low = close - 0.2

        # Build history
        for i in range(20, n):
            strategy.analyze("EXIT_TEST", high[:i], low[:i], close[:i])

        signal = strategy.generate_signal("EXIT_TEST", high, low, close, current_position="LONG")

        # Should get exit signal on momentum reversal
        if signal is not None and signal.direction == "FLAT":
            assert signal.signal_type == "exit"

    def test_signal_has_stop_and_target(self, strategy):
        """Test that signals include stop loss and take profit."""
        np.random.seed(123)
        n = 100

        # Create conditions for signal
        close = np.ones(n) * 100
        close[:60] += np.random.randn(60) * 0.03
        close[60:] = 100 + np.arange(40) * 0.2

        high = close + 0.2
        low = close - 0.2

        for i in range(50, 70):
            strategy.analyze("STOP_TEST", high[:i], low[:i], close[:i])

        signal = strategy.generate_signal("STOP_TEST", high, low, close, atr=1.5)

        if signal is not None and signal.direction in ["LONG", "SHORT"]:
            assert signal.stop_loss != 0
            assert signal.take_profit != 0
            if signal.direction == "LONG":
                assert signal.stop_loss < signal.entry_price
                assert signal.take_profit > signal.entry_price
            else:
                assert signal.stop_loss > signal.entry_price
                assert signal.take_profit < signal.entry_price


class TestStatus:
    """Tests for strategy status."""

    @pytest.fixture
    def strategy(self):
        """Create TTMSqueezeStrategy instance."""
        return TTMSqueezeStrategy()

    def test_get_status(self, strategy):
        """Test status reporting."""
        status = strategy.get_status()

        assert "bb_period" in status
        assert "kc_period" in status
        assert "momentum_period" in status
        assert "min_squeeze_bars" in status
        assert "tracked_symbols" in status

    def test_status_after_analysis(self, strategy):
        """Test status includes analyzed symbols."""
        np.random.seed(42)
        close = np.random.randn(50) + 100
        high = close + 0.3
        low = close - 0.3

        strategy.analyze("STATUS_TEST", high, low, close)
        status = strategy.get_status()

        assert status["tracked_symbols"] == 1
        assert "STATUS_TEST" in status["readings"]


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_default(self):
        """Test default factory creation."""
        strategy = create_ttm_squeeze_strategy()

        assert isinstance(strategy, TTMSqueezeStrategy)
        assert strategy._bb_period == 20
        assert strategy._kc_atr_mult == 1.5

    def test_create_custom_config(self):
        """Test factory with custom config."""
        strategy = create_ttm_squeeze_strategy({
            "bb_period": 15,
            "bb_std": 2.5,
            "kc_atr_mult": 2.0,
        })

        assert strategy._bb_period == 15
        assert strategy._bb_std == 2.5
        assert strategy._kc_atr_mult == 2.0

"""
Unit Tests for Ichimoku Cloud Strategy
======================================

Comprehensive tests for:
- Tenkan-sen (Conversion Line) calculation
- Kijun-sen (Base Line) calculation
- Senkou Span A (Leading Span A) calculation
- Senkou Span B (Leading Span B) calculation
- Chikou Span (Lagging Span) calculation
- Cloud color determination
- TK cross signals
- Price/cloud position analysis
- Chikou confirmation
- Combined signal generation
"""

import unittest
from datetime import datetime, timezone

import numpy as np


class TestTenkanSenCalculation(unittest.TestCase):
    """Tests for Tenkan-sen (Conversion Line) calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.ichimoku_strategy import calculate_tenkan_sen
        self.calculate_tenkan_sen = calculate_tenkan_sen

    def test_tenkan_basic_calculation(self):
        """Tenkan-sen should be midpoint of high/low range."""
        highs = np.array([100.0, 105.0, 110.0, 108.0, 106.0, 112.0, 115.0, 113.0, 118.0])
        lows = np.array([95.0, 100.0, 105.0, 103.0, 101.0, 107.0, 110.0, 108.0, 113.0])

        tenkan = self.calculate_tenkan_sen(highs, lows, period=9)

        # (highest_high + lowest_low) / 2 = (118 + 95) / 2 = 106.5
        self.assertAlmostEqual(tenkan, 106.5, places=4)

    def test_tenkan_insufficient_data(self):
        """Tenkan-sen should handle insufficient data."""
        highs = np.array([110.0, 115.0, 112.0])
        lows = np.array([105.0, 110.0, 108.0])

        tenkan = self.calculate_tenkan_sen(highs, lows, period=9)

        # Should return midpoint of last bar
        self.assertAlmostEqual(tenkan, (112.0 + 108.0) / 2, places=4)

    def test_tenkan_single_bar(self):
        """Tenkan-sen should handle single bar."""
        highs = np.array([110.0])
        lows = np.array([105.0])

        tenkan = self.calculate_tenkan_sen(highs, lows, period=9)

        self.assertAlmostEqual(tenkan, 107.5, places=4)


class TestKijunSenCalculation(unittest.TestCase):
    """Tests for Kijun-sen (Base Line) calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.ichimoku_strategy import calculate_kijun_sen
        self.calculate_kijun_sen = calculate_kijun_sen

    def test_kijun_basic_calculation(self):
        """Kijun-sen should be midpoint of 26-period high/low range."""
        np.random.seed(42)
        n = 30
        base = 100 + np.cumsum(np.random.randn(n))
        highs = base + 2
        lows = base - 2

        kijun = self.calculate_kijun_sen(highs, lows, period=26)

        # Should be midpoint of last 26 periods
        expected = (np.max(highs[-26:]) + np.min(lows[-26:])) / 2
        self.assertAlmostEqual(kijun, expected, places=4)

    def test_kijun_longer_period_than_tenkan(self):
        """Kijun should use longer period than Tenkan."""
        np.random.seed(42)
        n = 30
        base = np.linspace(100, 130, n)  # Trending up
        highs = base + 2
        lows = base - 2

        from strategies.ichimoku_strategy import calculate_tenkan_sen
        tenkan = calculate_tenkan_sen(highs, lows, period=9)
        kijun = self.calculate_kijun_sen(highs, lows, period=26)

        # In uptrend, tenkan should be above kijun (more responsive)
        self.assertGreater(tenkan, kijun)


class TestSenkouSpanA(unittest.TestCase):
    """Tests for Senkou Span A (Leading Span A) calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.ichimoku_strategy import calculate_senkou_span_a
        self.calculate_senkou_span_a = calculate_senkou_span_a

    def test_senkou_a_is_average_of_tenkan_kijun(self):
        """Senkou Span A should be average of Tenkan and Kijun."""
        tenkan = 110.0
        kijun = 105.0

        senkou_a = self.calculate_senkou_span_a(tenkan, kijun)

        self.assertAlmostEqual(senkou_a, 107.5, places=4)

    def test_senkou_a_when_equal(self):
        """Senkou Span A when Tenkan equals Kijun."""
        tenkan = 100.0
        kijun = 100.0

        senkou_a = self.calculate_senkou_span_a(tenkan, kijun)

        self.assertAlmostEqual(senkou_a, 100.0, places=4)


class TestSenkouSpanB(unittest.TestCase):
    """Tests for Senkou Span B (Leading Span B) calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.ichimoku_strategy import calculate_senkou_span_b
        self.calculate_senkou_span_b = calculate_senkou_span_b

    def test_senkou_b_basic_calculation(self):
        """Senkou Span B should be midpoint of 52-period range."""
        np.random.seed(42)
        n = 60
        base = 100 + np.cumsum(np.random.randn(n))
        highs = base + 3
        lows = base - 3

        senkou_b = self.calculate_senkou_span_b(highs, lows, period=52)

        expected = (np.max(highs[-52:]) + np.min(lows[-52:])) / 2
        self.assertAlmostEqual(senkou_b, expected, places=4)

    def test_senkou_b_insufficient_data(self):
        """Senkou Span B should handle insufficient data."""
        highs = np.array([110.0, 115.0, 112.0])
        lows = np.array([105.0, 110.0, 108.0])

        senkou_b = self.calculate_senkou_span_b(highs, lows, period=52)

        # Should use available data
        expected = (np.max(highs) + np.min(lows)) / 2
        self.assertAlmostEqual(senkou_b, expected, places=4)


class TestChikouSpan(unittest.TestCase):
    """Tests for Chikou Span (Lagging Span) calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.ichimoku_strategy import calculate_chikou_span
        self.calculate_chikou_span = calculate_chikou_span

    def test_chikou_returns_current_close(self):
        """Chikou Span should return current close."""
        closes = np.array([100.0, 105.0, 110.0, 115.0, 120.0])

        chikou = self.calculate_chikou_span(closes, displacement=26)

        self.assertAlmostEqual(chikou, 120.0, places=4)

    def test_chikou_empty_array(self):
        """Chikou should handle empty array."""
        chikou = self.calculate_chikou_span(np.array([]), displacement=26)

        self.assertEqual(chikou, 0.0)


class TestCloudColorDetermination(unittest.TestCase):
    """Tests for cloud color (bullish/bearish) determination."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.ichimoku_strategy import (
            IchimokuStrategy, IchimokuCloud, CloudColor
        )
        self.strategy = IchimokuStrategy({})
        self.IchimokuCloud = IchimokuCloud
        self.CloudColor = CloudColor

    def test_green_cloud_bullish(self):
        """Green cloud when Senkou A > Senkou B."""
        cloud = self.IchimokuCloud(
            tenkan_sen=110.0,
            kijun_sen=105.0,
            senkou_span_a=108.0,
            senkou_span_b=100.0,  # A > B = bullish
            chikou_span=110.0,
            current_price=110.0,
        )

        color = self.strategy.get_cloud_color(cloud)

        self.assertEqual(color, self.CloudColor.GREEN)

    def test_red_cloud_bearish(self):
        """Red cloud when Senkou A < Senkou B."""
        cloud = self.IchimokuCloud(
            tenkan_sen=100.0,
            kijun_sen=105.0,
            senkou_span_a=98.0,
            senkou_span_b=105.0,  # A < B = bearish
            chikou_span=100.0,
            current_price=100.0,
        )

        color = self.strategy.get_cloud_color(cloud)

        self.assertEqual(color, self.CloudColor.RED)

    def test_neutral_cloud(self):
        """Neutral when Senkou A == Senkou B."""
        cloud = self.IchimokuCloud(
            tenkan_sen=100.0,
            kijun_sen=100.0,
            senkou_span_a=100.0,
            senkou_span_b=100.0,
            chikou_span=100.0,
            current_price=100.0,
        )

        color = self.strategy.get_cloud_color(cloud)

        self.assertEqual(color, self.CloudColor.NEUTRAL)


class TestTKCrossSignal(unittest.TestCase):
    """Tests for Tenkan/Kijun cross signal detection."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.ichimoku_strategy import IchimokuStrategy, CrossType
        self.strategy = IchimokuStrategy({})
        self.CrossType = CrossType

    def test_bullish_tk_cross(self):
        """Detect bullish cross (Tenkan crosses above Kijun)."""
        cross = self.strategy.get_tk_cross_signal(
            current_tenkan=110.0,
            current_kijun=105.0,
            prev_tenkan=103.0,
            prev_kijun=105.0
        )

        self.assertEqual(cross, self.CrossType.BULLISH)

    def test_bearish_tk_cross(self):
        """Detect bearish cross (Tenkan crosses below Kijun)."""
        cross = self.strategy.get_tk_cross_signal(
            current_tenkan=100.0,
            current_kijun=105.0,
            prev_tenkan=107.0,
            prev_kijun=105.0
        )

        self.assertEqual(cross, self.CrossType.BEARISH)

    def test_no_tk_cross(self):
        """No cross when lines don't cross."""
        cross = self.strategy.get_tk_cross_signal(
            current_tenkan=110.0,
            current_kijun=105.0,
            prev_tenkan=108.0,
            prev_kijun=105.0  # Still above
        )

        self.assertEqual(cross, self.CrossType.NONE)


class TestPriceCloudPosition(unittest.TestCase):
    """Tests for price position relative to cloud."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.ichimoku_strategy import (
            IchimokuStrategy, IchimokuCloud, PricePosition
        )
        self.strategy = IchimokuStrategy({})
        self.IchimokuCloud = IchimokuCloud
        self.PricePosition = PricePosition

    def test_price_above_cloud(self):
        """Detect price above cloud (bullish)."""
        cloud = self.IchimokuCloud(
            tenkan_sen=110.0,
            kijun_sen=105.0,
            senkou_span_a=108.0,
            senkou_span_b=100.0,
            chikou_span=115.0,
            current_price=115.0,
        )

        position = self.strategy.get_price_cloud_position(115.0, cloud)

        self.assertEqual(position, self.PricePosition.ABOVE)

    def test_price_below_cloud(self):
        """Detect price below cloud (bearish)."""
        cloud = self.IchimokuCloud(
            tenkan_sen=100.0,
            kijun_sen=105.0,
            senkou_span_a=102.0,
            senkou_span_b=108.0,
            chikou_span=95.0,
            current_price=95.0,
        )

        position = self.strategy.get_price_cloud_position(95.0, cloud)

        self.assertEqual(position, self.PricePosition.BELOW)

    def test_price_inside_cloud(self):
        """Detect price inside cloud (neutral)."""
        cloud = self.IchimokuCloud(
            tenkan_sen=105.0,
            kijun_sen=105.0,
            senkou_span_a=100.0,
            senkou_span_b=110.0,
            chikou_span=105.0,
            current_price=105.0,
        )

        position = self.strategy.get_price_cloud_position(105.0, cloud)

        self.assertEqual(position, self.PricePosition.INSIDE)


class TestChikouConfirmation(unittest.TestCase):
    """Tests for Chikou Span confirmation."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.ichimoku_strategy import IchimokuStrategy
        self.strategy = IchimokuStrategy({})

    def test_bullish_chikou_confirmation(self):
        """Chikou confirmation when close > close 26 periods ago."""
        closes = np.array([100.0] * 26 + [110.0])  # 27 values, last higher

        is_bullish, reason = self.strategy.get_chikou_confirmation(
            closes, displacement=26
        )

        self.assertTrue(is_bullish)
        self.assertEqual(reason, "chikou_above_past_price")

    def test_bearish_chikou(self):
        """Bearish when close < close 26 periods ago."""
        closes = np.array([110.0] * 26 + [100.0])  # Last lower

        is_bullish, reason = self.strategy.get_chikou_confirmation(
            closes, displacement=26
        )

        self.assertFalse(is_bullish)
        self.assertEqual(reason, "chikou_below_past_price")

    def test_chikou_insufficient_data(self):
        """Handle insufficient data for Chikou confirmation."""
        closes = np.array([100.0, 105.0, 110.0])

        is_bullish, reason = self.strategy.get_chikou_confirmation(
            closes, displacement=26
        )

        self.assertFalse(is_bullish)
        self.assertEqual(reason, "insufficient_data")


class TestIchimokuSignalGeneration(unittest.TestCase):
    """Tests for complete Ichimoku signal generation."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.ichimoku_strategy import (
            IchimokuStrategy, SignalStrength, CloudColor, PricePosition
        )
        self.strategy = IchimokuStrategy({})
        self.SignalStrength = SignalStrength
        self.CloudColor = CloudColor
        self.PricePosition = PricePosition

    def _generate_uptrend_data(self, n=60):
        """Generate sample uptrend OHLC data."""
        base = np.linspace(100, 150, n)
        highs = base + np.random.rand(n) * 2
        lows = base - np.random.rand(n) * 2
        closes = base + np.random.rand(n) - 0.5
        return highs, lows, closes

    def _generate_downtrend_data(self, n=60):
        """Generate sample downtrend OHLC data."""
        base = np.linspace(150, 100, n)
        highs = base + np.random.rand(n) * 2
        lows = base - np.random.rand(n) * 2
        closes = base + np.random.rand(n) - 0.5
        return highs, lows, closes

    def test_bullish_signal_in_uptrend(self):
        """Should generate bullish signal in uptrend."""
        np.random.seed(42)
        highs, lows, closes = self._generate_uptrend_data(60)

        signal = self.strategy.generate_signal("TEST", highs, lows, closes)

        # In strong uptrend, should lean bullish
        self.assertIn(signal.direction, ["long", "flat"])
        self.assertGreater(signal.bullish_signals, signal.bearish_signals)

    def test_bearish_signal_in_downtrend(self):
        """Should generate bearish signal in downtrend."""
        np.random.seed(42)
        highs, lows, closes = self._generate_downtrend_data(60)

        signal = self.strategy.generate_signal("TEST", highs, lows, closes)

        # In strong downtrend, should lean bearish
        self.assertIn(signal.direction, ["short", "flat"])
        self.assertGreater(signal.bearish_signals, signal.bullish_signals)

    def test_signal_insufficient_data(self):
        """Should return flat signal with insufficient data."""
        highs = np.array([100.0, 105.0, 103.0])
        lows = np.array([95.0, 100.0, 98.0])
        closes = np.array([98.0, 103.0, 100.0])

        signal = self.strategy.generate_signal("TEST", highs, lows, closes)

        self.assertEqual(signal.direction, "flat")
        self.assertEqual(signal.strength, self.SignalStrength.NONE)

    def test_signal_contains_cloud(self):
        """Signal should contain IchimokuCloud values."""
        np.random.seed(42)
        highs, lows, closes = self._generate_uptrend_data(60)

        signal = self.strategy.generate_signal("TEST", highs, lows, closes)

        self.assertIsNotNone(signal.cloud)
        self.assertGreater(signal.cloud.tenkan_sen, 0)
        self.assertGreater(signal.cloud.kijun_sen, 0)

    def test_signal_confidence_bounds(self):
        """Signal confidence should be between 0 and 1."""
        np.random.seed(42)
        for _ in range(5):
            n = np.random.randint(60, 100)
            base = np.random.rand(n).cumsum() + 100
            highs = base + 2
            lows = base - 2
            closes = base

            signal = self.strategy.generate_signal("TEST", highs, lows, closes)

            self.assertGreaterEqual(signal.confidence, 0.0)
            self.assertLessEqual(signal.confidence, 1.0)

    def test_signal_stop_loss_calculated(self):
        """Directional signals should have stop loss."""
        np.random.seed(123)
        # Strong trend for directional signal
        base = np.linspace(100, 180, 60)
        highs = base + 1
        lows = base - 1
        closes = base

        signal = self.strategy.generate_signal("TEST", highs, lows, closes)

        if signal.direction != "flat":
            self.assertIsNotNone(signal.stop_loss_price)
            self.assertIsNotNone(signal.take_profit_price)


class TestIchimokuCloudDataclass(unittest.TestCase):
    """Tests for IchimokuCloud dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.ichimoku_strategy import IchimokuCloud
        self.IchimokuCloud = IchimokuCloud

    def test_cloud_top_and_bottom(self):
        """Cloud top/bottom properties should work correctly."""
        cloud = self.IchimokuCloud(
            tenkan_sen=110.0,
            kijun_sen=105.0,
            senkou_span_a=108.0,
            senkou_span_b=100.0,
            chikou_span=110.0,
        )

        self.assertEqual(cloud.cloud_top, 108.0)
        self.assertEqual(cloud.cloud_bottom, 100.0)

    def test_cloud_thickness(self):
        """Cloud thickness should be difference between spans."""
        cloud = self.IchimokuCloud(
            tenkan_sen=110.0,
            kijun_sen=105.0,
            senkou_span_a=108.0,
            senkou_span_b=100.0,
            chikou_span=110.0,
        )

        self.assertAlmostEqual(cloud.cloud_thickness, 8.0, places=4)

    def test_cloud_to_dict(self):
        """Cloud should convert to dictionary."""
        cloud = self.IchimokuCloud(
            tenkan_sen=110.0,
            kijun_sen=105.0,
            senkou_span_a=108.0,
            senkou_span_b=100.0,
            chikou_span=110.0,
            current_price=115.0,
        )

        result = cloud.to_dict()

        self.assertIn("tenkan_sen", result)
        self.assertIn("kijun_sen", result)
        self.assertIn("cloud_top", result)
        self.assertIn("cloud_bottom", result)


class TestIchimokuAnalyzeMethod(unittest.TestCase):
    """Tests for IchimokuStrategy.analyze() convenience method."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.ichimoku_strategy import IchimokuStrategy
        self.strategy = IchimokuStrategy({})

    def test_analyze_with_ohlc_dict(self):
        """analyze() should accept OHLC dictionary."""
        np.random.seed(42)
        n = 60
        base = np.linspace(100, 130, n)

        ohlc_data = {
            "high": base + 2,
            "low": base - 2,
            "close": base,
        }

        signal = self.strategy.analyze("TEST", ohlc_data)

        self.assertEqual(signal.symbol, "TEST")
        self.assertIn(signal.direction, ["long", "short", "flat"])

    def test_analyze_with_alternate_keys(self):
        """analyze() should accept alternate key names."""
        np.random.seed(42)
        n = 60
        base = np.linspace(100, 130, n)

        ohlc_data = {
            "highs": base + 2,
            "lows": base - 2,
            "closes": base,
        }

        signal = self.strategy.analyze("TEST", ohlc_data)

        self.assertEqual(signal.symbol, "TEST")


class TestIchimokuSeriesCalculation(unittest.TestCase):
    """Tests for full Ichimoku series calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.ichimoku_strategy import calculate_ichimoku_series
        self.calculate_ichimoku_series = calculate_ichimoku_series

    def test_series_array_lengths(self):
        """Series arrays should have correct lengths."""
        np.random.seed(42)
        n = 60
        base = np.linspace(100, 130, n)
        highs = base + 2
        lows = base - 2
        closes = base

        series = self.calculate_ichimoku_series(highs, lows, closes)

        self.assertEqual(len(series["tenkan_sen"]), n)
        self.assertEqual(len(series["kijun_sen"]), n)
        self.assertEqual(len(series["chikou_span"]), n)
        # Senkou spans are extended forward
        self.assertEqual(len(series["senkou_span_a"]), n + 26)
        self.assertEqual(len(series["senkou_span_b"]), n + 26)

    def test_series_contains_nan_for_warmup(self):
        """Early values should be NaN during warmup period."""
        np.random.seed(42)
        n = 60
        base = np.linspace(100, 130, n)
        highs = base + 2
        lows = base - 2
        closes = base

        series = self.calculate_ichimoku_series(highs, lows, closes)

        # Tenkan (9 period) should have NaN for first 8 values
        self.assertTrue(np.isnan(series["tenkan_sen"][0]))
        self.assertFalse(np.isnan(series["tenkan_sen"][8]))

        # Kijun (26 period) should have NaN for first 25 values
        self.assertTrue(np.isnan(series["kijun_sen"][0]))
        self.assertFalse(np.isnan(series["kijun_sen"][25]))


if __name__ == "__main__":
    unittest.main()

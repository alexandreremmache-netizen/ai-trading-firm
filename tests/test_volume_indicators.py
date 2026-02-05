"""
Unit Tests for Volume-Weighted Indicators
==========================================

Comprehensive tests for:
- VWMA (Volume-Weighted Moving Average)
- VWAP (Volume-Weighted Average Price)
- Volume Profile (POC, Value Area)
- OBV (On-Balance Volume)
- Volume RSI
- Climax detection
"""

import unittest
from datetime import datetime, timezone, timedelta

import numpy as np


class TestVWMACalculation(unittest.TestCase):
    """Tests for VWMA calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from core.volume_indicators import calculate_vwma, calculate_vwma_series
        self.calculate_vwma = calculate_vwma
        self.calculate_vwma_series = calculate_vwma_series

    def test_vwma_basic_calculation(self):
        """VWMA should weight prices by volume."""
        prices = np.array([100.0, 110.0, 120.0])
        volumes = np.array([1000.0, 2000.0, 1000.0])  # More volume at 110

        vwma = self.calculate_vwma(prices, volumes, period=3)

        # Expected: (100*1000 + 110*2000 + 120*1000) / 4000 = 110
        self.assertAlmostEqual(vwma, 110.0, places=4)

    def test_vwma_equal_volume_equals_sma(self):
        """VWMA with equal volumes should equal SMA."""
        prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])

        vwma = self.calculate_vwma(prices, volumes, period=5)
        sma = np.mean(prices)

        self.assertAlmostEqual(vwma, sma, places=6)

    def test_vwma_insufficient_data(self):
        """VWMA should handle insufficient data gracefully."""
        prices = np.array([100.0, 110.0])
        volumes = np.array([1000.0, 2000.0])

        vwma = self.calculate_vwma(prices, volumes, period=5)

        # Should return mean of available data
        self.assertAlmostEqual(vwma, 105.0, places=4)

    def test_vwma_zero_volume_handling(self):
        """VWMA should handle zero total volume."""
        prices = np.array([100.0, 110.0, 120.0])
        volumes = np.array([0.0, 0.0, 0.0])

        vwma = self.calculate_vwma(prices, volumes, period=3)

        # Should fallback to simple mean
        self.assertAlmostEqual(vwma, 110.0, places=4)

    def test_vwma_mismatched_lengths_raises(self):
        """VWMA should raise error for mismatched array lengths."""
        prices = np.array([100.0, 110.0, 120.0])
        volumes = np.array([1000.0, 2000.0])

        with self.assertRaises(ValueError):
            self.calculate_vwma(prices, volumes, period=3)

    def test_vwma_series_calculation(self):
        """VWMA series should calculate rolling values."""
        prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])

        vwma_series = self.calculate_vwma_series(prices, volumes, period=3)

        # First 2 values should be NaN
        self.assertTrue(np.isnan(vwma_series[0]))
        self.assertTrue(np.isnan(vwma_series[1]))

        # Value at index 2 should be average of first 3
        self.assertAlmostEqual(vwma_series[2], 110.0, places=4)

    def test_vwma_high_volume_weighting(self):
        """VWMA should weight heavily toward high-volume prices."""
        prices = np.array([100.0, 200.0, 300.0])
        volumes = np.array([100.0, 10000.0, 100.0])  # Extreme volume at 200

        vwma = self.calculate_vwma(prices, volumes, period=3)

        # Should be very close to 200 due to volume weighting
        self.assertGreater(vwma, 190.0)
        self.assertLess(vwma, 210.0)


class TestVWAPCalculation(unittest.TestCase):
    """Tests for VWAP calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from core.volume_indicators import calculate_vwap
        self.calculate_vwap = calculate_vwap

    def test_vwap_basic_calculation(self):
        """VWAP should calculate volume-weighted average price."""
        prices = np.array([100.0, 110.0, 120.0])
        volumes = np.array([1000.0, 2000.0, 1000.0])

        result = self.calculate_vwap(prices, volumes)

        # VWAP = (100*1000 + 110*2000 + 120*1000) / 4000 = 110
        self.assertAlmostEqual(result.vwap, 110.0, places=4)

    def test_vwap_cumulative_volume(self):
        """VWAP result should include cumulative volume."""
        prices = np.array([100.0, 110.0, 120.0])
        volumes = np.array([1000.0, 2000.0, 3000.0])

        result = self.calculate_vwap(prices, volumes)

        self.assertAlmostEqual(result.cumulative_volume, 6000.0, places=2)

    def test_vwap_bands_calculation(self):
        """VWAP bands should be symmetric around VWAP."""
        np.random.seed(42)
        prices = 100 + np.random.randn(50)
        volumes = 1000 + np.random.rand(50) * 1000

        result = self.calculate_vwap(prices, volumes, std_dev_bands=2.0)

        # Bands should be symmetric
        upper_diff = result.upper_band - result.vwap
        lower_diff = result.vwap - result.lower_band
        self.assertAlmostEqual(upper_diff, lower_diff, places=6)

    def test_vwap_session_reset(self):
        """VWAP should reset on new trading day."""
        day1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        day2 = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)

        # Prices and volumes for two days
        timestamps = [
            day1, day1 + timedelta(hours=1), day1 + timedelta(hours=2),
            day2, day2 + timedelta(hours=1)
        ]
        prices = np.array([100.0, 102.0, 101.0, 150.0, 152.0])
        volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])

        result = self.calculate_vwap(prices, volumes, timestamps)

        # VWAP should be based on day 2 prices only
        self.assertGreater(result.vwap, 140.0)

    def test_vwap_empty_input(self):
        """VWAP should handle empty arrays."""
        result = self.calculate_vwap(np.array([]), np.array([]))

        self.assertEqual(result.vwap, 0.0)
        self.assertEqual(result.cumulative_volume, 0.0)

    def test_vwap_result_to_dict(self):
        """VWAP result should convert to dictionary."""
        prices = np.array([100.0, 110.0])
        volumes = np.array([1000.0, 1000.0])

        result = self.calculate_vwap(prices, volumes)
        result_dict = result.to_dict()

        self.assertIn("vwap", result_dict)
        self.assertIn("upper_band", result_dict)
        self.assertIn("lower_band", result_dict)


class TestVolumeProfile(unittest.TestCase):
    """Tests for Volume Profile calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from core.volume_indicators import (
            calculate_volume_profile, get_poc, get_value_area
        )
        self.calculate_volume_profile = calculate_volume_profile
        self.get_poc = get_poc
        self.get_value_area = get_value_area

    def test_volume_profile_poc(self):
        """Volume Profile should identify Point of Control."""
        # Concentrated volume at 110
        prices = np.array([100.0, 105.0, 110.0, 110.0, 110.0, 115.0, 120.0])
        volumes = np.array([100.0, 100.0, 1000.0, 1000.0, 1000.0, 100.0, 100.0])

        profile = self.calculate_volume_profile(prices, volumes, n_bins=5)

        # POC should be near 110
        self.assertGreater(profile.poc_price, 105.0)
        self.assertLess(profile.poc_price, 115.0)

    def test_volume_profile_total_volume(self):
        """Volume Profile should track total volume correctly."""
        prices = np.array([100.0, 110.0, 120.0])
        volumes = np.array([1000.0, 2000.0, 3000.0])

        profile = self.calculate_volume_profile(prices, volumes, n_bins=3)

        self.assertAlmostEqual(profile.total_volume, 6000.0, places=2)

    def test_volume_profile_value_area(self):
        """Value Area should contain specified percentage of volume."""
        np.random.seed(42)
        # Normal distribution of prices with volume
        prices = np.random.normal(100, 10, 100)
        volumes = np.ones(100) * 100

        profile = self.calculate_volume_profile(prices, volumes, n_bins=20)

        # Value area should exist and be reasonable
        self.assertGreater(profile.value_area_high, profile.poc_price)
        self.assertLess(profile.value_area_low, profile.poc_price)

    def test_get_poc_function(self):
        """get_poc should return POC price and volume."""
        prices = np.array([100.0, 105.0, 110.0, 110.0, 115.0])
        volumes = np.array([100.0, 100.0, 1000.0, 1000.0, 100.0])

        profile = self.calculate_volume_profile(prices, volumes, n_bins=4)
        poc_price, poc_volume = self.get_poc(profile)

        self.assertEqual(poc_price, profile.poc_price)
        self.assertEqual(poc_volume, profile.poc_volume)

    def test_get_value_area_function(self):
        """get_value_area should return VAH and VAL."""
        prices = np.array([90.0, 95.0, 100.0, 100.0, 105.0, 110.0])
        volumes = np.array([100.0, 200.0, 500.0, 500.0, 200.0, 100.0])

        profile = self.calculate_volume_profile(prices, volumes, n_bins=5)
        vah, val = self.get_value_area(profile, pct=0.70)

        self.assertGreater(vah, val)
        self.assertGreaterEqual(vah, profile.poc_price)
        self.assertLessEqual(val, profile.poc_price)

    def test_volume_profile_same_prices(self):
        """Volume Profile should handle identical prices."""
        prices = np.array([100.0, 100.0, 100.0])
        volumes = np.array([1000.0, 2000.0, 3000.0])

        profile = self.calculate_volume_profile(prices, volumes, n_bins=5)

        self.assertEqual(profile.poc_price, 100.0)
        self.assertAlmostEqual(profile.total_volume, 6000.0, places=2)

    def test_volume_profile_empty_input(self):
        """Volume Profile should handle empty arrays."""
        profile = self.calculate_volume_profile(np.array([]), np.array([]), n_bins=5)

        self.assertEqual(profile.poc_price, 0.0)
        self.assertEqual(profile.total_volume, 0.0)


class TestOBV(unittest.TestCase):
    """Tests for On-Balance Volume calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from core.volume_indicators import calculate_obv, calculate_obv_ma
        self.calculate_obv = calculate_obv
        self.calculate_obv_ma = calculate_obv_ma

    def test_obv_increases_on_up_day(self):
        """OBV should increase when price increases."""
        prices = np.array([100.0, 105.0])
        volumes = np.array([1000.0, 2000.0])

        obv = self.calculate_obv(prices, volumes)

        self.assertEqual(obv[-1], 2000.0)  # Added volume on up day

    def test_obv_decreases_on_down_day(self):
        """OBV should decrease when price decreases."""
        prices = np.array([100.0, 95.0])
        volumes = np.array([1000.0, 2000.0])

        obv = self.calculate_obv(prices, volumes)

        self.assertEqual(obv[-1], -2000.0)  # Subtracted volume on down day

    def test_obv_unchanged_on_flat_day(self):
        """OBV should not change when price is unchanged."""
        prices = np.array([100.0, 100.0])
        volumes = np.array([1000.0, 2000.0])

        obv = self.calculate_obv(prices, volumes)

        self.assertEqual(obv[-1], 0.0)  # No change

    def test_obv_cumulative(self):
        """OBV should be cumulative."""
        prices = np.array([100.0, 105.0, 110.0, 105.0])
        volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0])

        obv = self.calculate_obv(prices, volumes)

        # Up, up, down: 0 + 1000 + 1000 - 1000 = 1000
        self.assertEqual(obv[-1], 1000.0)

    def test_obv_with_ma(self):
        """OBV MA should smooth the values."""
        prices = np.array([100.0, 105.0, 103.0, 108.0, 106.0, 110.0])
        volumes = np.array([1000.0] * 6)

        obv, obv_ma = self.calculate_obv_ma(prices, volumes, period=3)

        self.assertEqual(len(obv), len(prices))
        self.assertEqual(len(obv_ma), len(prices))

    def test_obv_empty_input(self):
        """OBV should handle empty arrays."""
        obv = self.calculate_obv(np.array([]), np.array([]))

        self.assertEqual(len(obv), 0)


class TestVolumeRSI(unittest.TestCase):
    """Tests for Volume RSI calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from core.volume_indicators import (
            calculate_volume_rsi, calculate_volume_rsi_series
        )
        self.calculate_volume_rsi = calculate_volume_rsi
        self.calculate_volume_rsi_series = calculate_volume_rsi_series

    def test_volume_rsi_bounds(self):
        """Volume RSI should be between 0 and 100."""
        np.random.seed(42)
        volumes = 1000 + np.random.rand(50) * 1000

        rsi = self.calculate_volume_rsi(volumes, period=14)

        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)

    def test_volume_rsi_high_on_increasing_volume(self):
        """Volume RSI should be high when volume is consistently increasing."""
        volumes = np.array([1000 + i * 100 for i in range(20)])

        rsi = self.calculate_volume_rsi(volumes, period=14)

        self.assertGreater(rsi, 70)

    def test_volume_rsi_low_on_decreasing_volume(self):
        """Volume RSI should be low when volume is consistently decreasing."""
        volumes = np.array([2000 - i * 100 for i in range(20)])

        rsi = self.calculate_volume_rsi(volumes, period=14)

        self.assertLess(rsi, 30)

    def test_volume_rsi_insufficient_data(self):
        """Volume RSI should return 50 with insufficient data."""
        volumes = np.array([1000.0, 1100.0, 1200.0])

        rsi = self.calculate_volume_rsi(volumes, period=14)

        self.assertEqual(rsi, 50.0)

    def test_volume_rsi_series(self):
        """Volume RSI series should calculate rolling values."""
        volumes = 1000 + np.cumsum(np.random.randn(30))
        volumes = np.abs(volumes)  # Ensure positive

        rsi_series = self.calculate_volume_rsi_series(volumes, period=14)

        self.assertEqual(len(rsi_series), len(volumes))
        # All values should be in bounds
        self.assertTrue(np.all(rsi_series >= 0))
        self.assertTrue(np.all(rsi_series <= 100))


class TestVolumeClimax(unittest.TestCase):
    """Tests for volume climax detection."""

    def setUp(self):
        """Set up test fixtures."""
        from core.volume_indicators import detect_volume_climax
        self.detect_volume_climax = detect_volume_climax

    def test_detect_buying_climax(self):
        """Should detect buying climax on high volume with close near high."""
        # Normal trading followed by climax bar
        prices = np.array([100.0] * 20 + [105.0])  # Price jumps
        volumes = np.array([1000.0] * 20 + [5000.0])  # Volume spikes
        highs = np.array([101.0] * 20 + [106.0])
        lows = np.array([99.0] * 20 + [100.0])

        # Close near high of range
        prices[-1] = 105.8  # Close near high

        result = self.detect_volume_climax(
            prices, volumes, highs, lows,
            volume_threshold=2.0, lookback=20
        )

        # Volume ratio should be high
        self.assertGreater(result.volume_ratio, 2.0)

    def test_detect_selling_climax(self):
        """Should detect selling climax on high volume with close near low."""
        prices = np.array([100.0] * 20 + [95.0])  # Price drops
        volumes = np.array([1000.0] * 20 + [5000.0])  # Volume spikes
        highs = np.array([101.0] * 20 + [100.0])
        lows = np.array([99.0] * 20 + [94.0])

        # Close near low of range
        prices[-1] = 94.2

        result = self.detect_volume_climax(
            prices, volumes, highs, lows,
            volume_threshold=2.0, lookback=20
        )

        self.assertGreater(result.volume_ratio, 2.0)

    def test_no_climax_normal_volume(self):
        """Should not detect climax with normal volume."""
        prices = np.array([100.0] * 25)
        volumes = np.array([1000.0] * 25)

        result = self.detect_volume_climax(
            prices, volumes,
            volume_threshold=2.0, lookback=20
        )

        self.assertFalse(result.is_climax)

    def test_volume_climax_insufficient_data(self):
        """Should handle insufficient data."""
        prices = np.array([100.0, 105.0])
        volumes = np.array([1000.0, 5000.0])

        result = self.detect_volume_climax(
            prices, volumes,
            volume_threshold=2.0, lookback=20
        )

        self.assertFalse(result.is_climax)

    def test_volume_climax_result_fields(self):
        """Result should have all expected fields."""
        prices = np.array([100.0] * 25)
        volumes = np.array([1000.0] * 25)

        result = self.detect_volume_climax(prices, volumes)

        self.assertIsInstance(result.is_climax, bool)
        self.assertIsInstance(result.volume_ratio, float)
        self.assertIsInstance(result.price_range_ratio, float)
        self.assertIsInstance(result.close_position, float)
        self.assertIsInstance(result.strength, float)


class TestAdditionalVolumeIndicators(unittest.TestCase):
    """Tests for additional volume analysis functions."""

    def setUp(self):
        """Set up test fixtures."""
        from core.volume_indicators import (
            calculate_volume_price_trend,
            calculate_force_index,
            analyze_volume_trend
        )
        self.calculate_vpt = calculate_volume_price_trend
        self.calculate_force_index = calculate_force_index
        self.analyze_volume_trend = analyze_volume_trend

    def test_volume_price_trend(self):
        """VPT should accumulate based on price percentage changes."""
        prices = np.array([100.0, 110.0, 121.0])  # 10% increase each
        volumes = np.array([1000.0, 1000.0, 1000.0])

        vpt = self.calculate_vpt(prices, volumes)

        self.assertEqual(len(vpt), 3)
        self.assertGreater(vpt[-1], 0)  # Should be positive after increases

    def test_force_index(self):
        """Force index should measure price*volume momentum."""
        prices = np.array([100.0, 105.0, 110.0, 108.0, 112.0])
        volumes = np.array([1000.0, 2000.0, 1500.0, 1000.0, 2500.0])

        force = self.calculate_force_index(prices, volumes, period=3)

        self.assertEqual(len(force), len(prices))

    def test_analyze_volume_trend_increasing(self):
        """Should detect increasing volume trend."""
        volumes = np.array([1000.0 + i * 100 for i in range(25)])

        analysis = self.analyze_volume_trend(volumes, period=20)

        self.assertEqual(analysis["trend"], "increasing")
        self.assertTrue(analysis["is_increasing"])

    def test_analyze_volume_trend_decreasing(self):
        """Should detect decreasing volume trend."""
        volumes = np.array([2500.0 - i * 100 for i in range(25)])

        analysis = self.analyze_volume_trend(volumes, period=20)

        self.assertEqual(analysis["trend"], "decreasing")
        self.assertFalse(analysis["is_increasing"])

    def test_analyze_volume_trend_flat(self):
        """Should detect flat volume trend."""
        volumes = np.array([1000.0] * 25)

        analysis = self.analyze_volume_trend(volumes, period=20)

        self.assertEqual(analysis["trend"], "flat")


if __name__ == "__main__":
    unittest.main()

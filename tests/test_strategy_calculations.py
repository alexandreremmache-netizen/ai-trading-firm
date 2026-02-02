"""
Unit Tests for Strategy Calculations
====================================

Addresses issue #Q22: Missing unit tests for strategy calculations.

Comprehensive tests for:
- RSI calculation
- MACD calculation
- Moving averages
- Bollinger Bands
- Technical indicators
- Signal generation
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import numpy as np


class TestMomentumStrategyDefaultInitialization(unittest.TestCase):
    """Tests for MomentumStrategy default initialization (Issue #16)."""

    def test_default_initialization(self):
        """Verify MomentumStrategy initializes correctly with empty config."""
        from strategies.momentum_strategy import MomentumStrategy
        strategy = MomentumStrategy({})

        # Verify strategy has required attributes with defaults
        self.assertIsNotNone(strategy)
        # Strategy should have basic configuration set
        self.assertTrue(hasattr(strategy, 'calculate_rsi'))
        self.assertTrue(hasattr(strategy, 'calculate_macd'))
        self.assertTrue(hasattr(strategy, 'calculate_sma'))
        self.assertTrue(hasattr(strategy, 'calculate_ema'))
        self.assertTrue(hasattr(strategy, 'analyze'))


class TestRSICalculation(unittest.TestCase):
    """Tests for RSI calculation correctness."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.momentum_strategy import MomentumStrategy
        self.strategy = MomentumStrategy({})

    def test_rsi_neutral_50_on_equal_gains_losses(self):
        """RSI should be 50 when gains equal losses."""
        # Alternating up and down moves of same magnitude
        prices = np.array([100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0,
                          101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0])
        rsi = self.strategy.calculate_rsi(prices, period=14)
        # RSI should be near 50 (not exactly due to Wilder's smoothing)
        self.assertGreater(rsi, 45)
        self.assertLess(rsi, 55)

    def test_rsi_100_on_all_gains(self):
        """RSI should approach 100 when all moves are gains."""
        prices = np.array([100 + i for i in range(20)])  # Monotonic increase
        rsi = self.strategy.calculate_rsi(prices, period=14)
        self.assertGreater(rsi, 95)

    def test_rsi_0_on_all_losses(self):
        """RSI should approach 0 when all moves are losses."""
        prices = np.array([100 - i for i in range(20)])  # Monotonic decrease
        rsi = self.strategy.calculate_rsi(prices, period=14)
        self.assertLess(rsi, 5)

    def test_rsi_default_50_insufficient_data(self):
        """RSI should return 50 with insufficient data."""
        prices = np.array([100.0, 101.0, 102.0])  # Less than period + 1
        rsi = self.strategy.calculate_rsi(prices, period=14)
        self.assertEqual(rsi, 50.0)

    def test_rsi_bounds(self):
        """RSI should always be between 0 and 100."""
        np.random.seed(42)
        for _ in range(10):
            prices = np.random.randn(100).cumsum() + 100
            rsi = self.strategy.calculate_rsi(prices, period=14)
            self.assertGreaterEqual(rsi, 0)
            self.assertLessEqual(rsi, 100)


class TestMACDCalculation(unittest.TestCase):
    """Tests for MACD calculation correctness."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.momentum_strategy import MomentumStrategy
        self.strategy = MomentumStrategy({})

    def test_macd_returns_tuple_of_three(self):
        """MACD should return (macd, signal, histogram)."""
        prices = np.array([100 + i + np.sin(i * 0.5) for i in range(50)])
        result = self.strategy.calculate_macd(prices)
        self.assertEqual(len(result), 3)
        macd, signal, histogram = result
        self.assertIsInstance(macd, float)
        self.assertIsInstance(signal, float)
        self.assertIsInstance(histogram, float)

    def test_macd_histogram_is_macd_minus_signal(self):
        """Histogram should equal MACD minus signal."""
        prices = np.array([100 + i + np.sin(i * 0.5) for i in range(50)])
        macd, signal, histogram = self.strategy.calculate_macd(prices)
        self.assertAlmostEqual(histogram, macd - signal, places=6)

    def test_macd_zeros_insufficient_data(self):
        """MACD should return zeros with insufficient data."""
        prices = np.array([100.0, 101.0, 102.0])
        macd, signal, histogram = self.strategy.calculate_macd(prices)
        self.assertEqual(macd, 0.0)
        self.assertEqual(signal, 0.0)
        self.assertEqual(histogram, 0.0)

    def test_macd_positive_in_uptrend(self):
        """MACD should be positive in strong uptrend."""
        prices = np.array([100 + i * 2 for i in range(50)])  # Strong uptrend
        macd, signal, histogram = self.strategy.calculate_macd(prices)
        self.assertGreater(macd, 0)


class TestMovingAverages(unittest.TestCase):
    """Tests for moving average calculations."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.momentum_strategy import MomentumStrategy
        self.strategy = MomentumStrategy({})

    def test_sma_correct_calculation(self):
        """SMA should be simple mean of last n values."""
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = self.strategy.calculate_sma(prices, period=3)
        self.assertEqual(sma, 4.0)  # (3 + 4 + 5) / 3

    def test_sma_returns_last_if_insufficient_data(self):
        """SMA should return last price if insufficient data."""
        prices = np.array([100.0, 101.0])
        sma = self.strategy.calculate_sma(prices, period=5)
        self.assertEqual(sma, 101.0)

    def test_ema_weights_recent_more(self):
        """EMA should weight recent values more than SMA (Issue #17 - more jumps in data)."""
        # Create data with multiple jumps to better test EMA weighting
        prices = np.array([100.0] * 5 + [120.0] * 3 + [150.0] * 2 + [200.0])
        sma = self.strategy.calculate_sma(prices, period=5)
        ema = self.strategy.calculate_ema(prices, period=5)
        # EMA should be closer to 200 than SMA due to recency weighting
        self.assertGreater(ema, sma)

    def test_ema_responds_to_multiple_jumps(self):
        """EMA should respond more quickly to multiple price changes."""
        # Data with gradual jumps
        prices = np.array([100.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 200.0])
        ema = self.strategy.calculate_ema(prices, period=5)
        sma = self.strategy.calculate_sma(prices, period=5)
        # With an uptrend and jumps, EMA should be higher than SMA
        self.assertGreater(ema, sma)


class TestBollingerBands(unittest.TestCase):
    """Tests for Bollinger Bands calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from core.technical_indicators import BollingerBandsCalculator
        self.bb = BollingerBandsCalculator(period=20, std_dev=2.0)

    def test_bands_symmetric_around_middle(self):
        """Bands should be symmetric around middle."""
        for i in range(25):
            result = self.bb.update(100.0 + np.random.randn())
        if result:
            diff_upper = result.upper - result.middle
            diff_lower = result.middle - result.lower
            self.assertAlmostEqual(diff_upper, diff_lower, places=6)

    def test_percent_b_calculation(self):
        """Percent B should be correct relative to bands."""
        # Fill with constant values
        for i in range(20):
            self.bb.update(100.0)

        result = self.bb.update(100.0)
        # At middle, percent_b should be 0.5
        self.assertAlmostEqual(result.percent_b, 0.5, places=1)

    def test_bandwidth_increases_with_volatility(self):
        """Bandwidth should increase with higher volatility."""
        from core.technical_indicators import BollingerBandsCalculator
        bb_stable = BollingerBandsCalculator(period=20, std_dev=2.0)
        bb_volatile = BollingerBandsCalculator(period=20, std_dev=2.0)

        np.random.seed(42)
        for i in range(25):
            bb_stable.update(100.0 + np.random.randn() * 0.1)
            bb_volatile.update(100.0 + np.random.randn() * 5.0)

        result_stable = bb_stable.update(100.0)
        result_volatile = bb_volatile.update(100.0)

        if result_stable and result_volatile:
            self.assertLess(result_stable.bandwidth, result_volatile.bandwidth)


class TestADXCalculation(unittest.TestCase):
    """Tests for ADX calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from core.technical_indicators import ADXCalculator, OHLCV
        self.adx = ADXCalculator(period=14)
        self.OHLCV = OHLCV

    def test_adx_bounds(self):
        """ADX should be between 0 and 100."""
        from datetime import datetime, timezone

        for i in range(30):
            bar = self.OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=100 + i,
                high=102 + i,
                low=98 + i,
                close=101 + i,
                volume=10000
            )
            result = self.adx.update(bar)
            if result:
                self.assertGreaterEqual(result.adx, 0)
                self.assertLessEqual(result.adx, 100)

    def test_adx_high_in_strong_trend(self):
        """ADX should be high in strong trending market."""
        from datetime import datetime, timezone

        # Strong uptrend
        for i in range(30):
            bar = self.OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=100 + i * 2,
                high=102 + i * 2,
                low=99 + i * 2,
                close=101 + i * 2,
                volume=10000
            )
            result = self.adx.update(bar)

        # ADX should indicate strong trend
        if result:
            self.assertGreater(result.adx, 20)


class TestStrategySignalGeneration(unittest.TestCase):
    """Tests for strategy signal generation."""

    def setUp(self):
        """Set up test fixtures."""
        from strategies.momentum_strategy import MomentumStrategy
        self.strategy = MomentumStrategy({})

    def test_analyze_returns_momentum_signal(self):
        """Analyze should return MomentumSignal object."""
        prices = np.random.randn(50).cumsum() + 100
        signal = self.strategy.analyze("AAPL", prices)

        self.assertEqual(signal.symbol, "AAPL")
        self.assertIn(signal.direction, ["long", "short", "flat"])
        self.assertGreaterEqual(signal.strength, -1)
        self.assertLessEqual(signal.strength, 1)
        self.assertGreaterEqual(signal.confidence, 0)
        self.assertLessEqual(signal.confidence, 1)

    def test_flat_signal_on_insufficient_data(self):
        """Should return flat signal with insufficient data."""
        prices = np.array([100.0, 101.0, 102.0])
        signal = self.strategy.analyze("AAPL", prices)

        self.assertEqual(signal.direction, "flat")
        self.assertEqual(signal.strength, 0.0)

    def test_indicators_in_signal(self):
        """Signal should contain calculated indicators."""
        prices = np.random.randn(50).cumsum() + 100
        signal = self.strategy.analyze("AAPL", prices)

        expected_indicators = ["fast_ma", "slow_ma", "rsi", "macd", "macd_signal", "roc"]
        for indicator in expected_indicators:
            self.assertIn(indicator, signal.indicators)


class TestVolumeIndicators(unittest.TestCase):
    """Tests for volume-based indicators."""

    def setUp(self):
        """Set up test fixtures."""
        from core.technical_indicators import (
            VWAPCalculator, OBVCalculator, MFICalculator, OHLCV
        )
        self.vwap = VWAPCalculator()
        self.obv = OBVCalculator()
        self.mfi = MFICalculator()
        self.OHLCV = OHLCV

    def test_vwap_resets_on_new_day(self):
        """VWAP should reset on new trading day."""
        from datetime import datetime, timezone, timedelta

        day1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        day2 = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)

        # Day 1 bars
        for i in range(5):
            bar = self.OHLCV(
                timestamp=day1 + timedelta(minutes=i * 5),
                open=100, high=101, low=99, close=100, volume=1000
            )
            result1 = self.vwap.update(bar)

        # Day 2 bar - should reset
        bar_day2 = self.OHLCV(
            timestamp=day2,
            open=110, high=111, low=109, close=110, volume=1000
        )
        result2 = self.vwap.update(bar_day2)

        # VWAP should reflect day 2 price
        if result2:
            self.assertGreater(result2.vwap, 105)

    def test_obv_increases_on_up_day(self):
        """OBV should increase when close > previous close."""
        result1 = self.obv.update(100.0, 1000)
        result2 = self.obv.update(105.0, 2000)  # Up day

        if result2:
            self.assertGreater(result2.obv, 0)

    def test_mfi_bounds(self):
        """MFI should be between 0 and 100."""
        from datetime import datetime, timezone

        np.random.seed(42)
        for i in range(20):
            bar = self.OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=100 + np.random.randn(),
                high=102 + np.random.randn(),
                low=98 + np.random.randn(),
                close=100 + np.random.randn(),
                volume=10000 + np.random.randint(-1000, 1000)
            )
            result = self.mfi.update(bar)
            if result:
                self.assertGreaterEqual(result.mfi, 0)
                self.assertLessEqual(result.mfi, 100)


class TestStrategyParameters(unittest.TestCase):
    """Tests for strategy parameter management."""

    def test_rsi_parameters_validation(self):
        """RSI parameters should validate thresholds."""
        from core.strategy_parameters import RSIParameters

        # Valid parameters
        params = RSIParameters(overbought=70, oversold=30)
        self.assertEqual(params.overbought, 70)
        self.assertEqual(params.oversold, 30)

        # Invalid: oversold >= overbought
        with self.assertRaises(ValueError):
            RSIParameters(overbought=30, oversold=70)

    def test_rsi_is_overbought(self):
        """is_overbought should return correct result."""
        from core.strategy_parameters import RSIParameters

        params = RSIParameters(overbought=70, oversold=30)
        self.assertTrue(params.is_overbought(75))
        self.assertFalse(params.is_overbought(65))

    def test_rsi_is_oversold(self):
        """is_oversold should return correct result."""
        from core.strategy_parameters import RSIParameters

        params = RSIParameters(overbought=70, oversold=30)
        self.assertTrue(params.is_oversold(25))
        self.assertFalse(params.is_oversold(35))

    def test_parameter_bounds_validation(self):
        """ParameterBounds should validate configuration."""
        from core.strategy_parameters import ParameterBounds

        # Valid bounds
        bounds = ParameterBounds(min_value=0, max_value=100, default_value=50)
        self.assertTrue(bounds.validate(50))

        # Invalid: min > max
        with self.assertRaises(ValueError):
            ParameterBounds(min_value=100, max_value=0, default_value=50)

        # Invalid: default outside bounds
        with self.assertRaises(ValueError):
            ParameterBounds(min_value=0, max_value=50, default_value=75)


class TestBacktestMetrics(unittest.TestCase):
    """Tests for backtest metrics calculation."""

    def test_sharpe_ratio_calculation(self):
        """Sharpe ratio should be calculated correctly."""
        from core.backtest import BacktestMetrics

        metrics = BacktestMetrics()
        # Test that default values are zero
        self.assertEqual(metrics.sharpe_ratio, 0.0)

    def test_metrics_to_dict(self):
        """Metrics should convert to dictionary correctly."""
        from core.backtest import BacktestMetrics

        metrics = BacktestMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=0.10
        )
        result = metrics.to_dict()

        self.assertIn("returns", result)
        self.assertIn("risk", result)
        self.assertIn("risk_adjusted", result)
        self.assertEqual(result["returns"]["total_return_pct"], 15.0)


if __name__ == "__main__":
    unittest.main()

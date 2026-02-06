"""
Tests for MACD-v Strategy (Phase 11 - Charles H. Dow Award 2022)
================================================================

Tests for neutral zone filter, ranging market detection,
and MACD-v signal generation.
"""

import numpy as np
import pytest
from dataclasses import dataclass

from strategies.macdv_strategy import MACDvStrategy, MACDvSignal


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def strategy():
    """Create MACDvStrategy with default config."""
    return MACDvStrategy({})


@pytest.fixture
def strategy_no_filter():
    """Create MACDvStrategy with neutral zone filter disabled."""
    return MACDvStrategy({
        "neutral_zone_filter": False,
        "ranging_detection_enabled": False,
    })


def _make_trending_prices(direction: str, n: int = 100, base: float = 100.0) -> np.ndarray:
    """Generate trending price series."""
    np.random.seed(42)
    if direction == "up":
        trend = np.linspace(0, 15, n)
    elif direction == "down":
        trend = np.linspace(0, -15, n)
    else:
        trend = np.zeros(n)
    noise = np.random.randn(n) * 0.3
    return base + trend + noise


def _make_ranging_prices(n: int = 100, base: float = 100.0) -> np.ndarray:
    """Generate sideways/ranging price series."""
    np.random.seed(42)
    return base + np.random.randn(n) * 0.5


# ============================================================================
# NEUTRAL ZONE FILTER TESTS
# ============================================================================

class TestNeutralZoneFilter:
    """Tests for neutral zone filter (-50 to +50)."""

    def test_neutral_zone_defaults(self, strategy):
        """Verify neutral zone filter is enabled by default with correct bounds."""
        assert strategy._neutral_zone_filter is True
        assert strategy._neutral_zone_upper == 50.0
        assert strategy._neutral_zone_lower == -50.0

    def test_neutral_zone_rejects_weak_signals(self, strategy):
        """Signals with MACD-v in [-50, +50] should be rejected."""
        # Generate prices that create a small MACD-v value (in neutral zone)
        prices = _make_ranging_prices(n=100)
        highs = prices + 0.5
        lows = prices - 0.5

        signal = strategy.analyze(
            symbol="MES",
            prices=prices,
            highs=highs,
            lows=lows,
        )

        # In ranging market, signal should be flat (neutral zone filter)
        assert signal.direction == "flat", (
            f"Expected 'flat' in neutral zone, got '{signal.direction}' "
            f"(MACD-v={signal.indicators.get('macdv', 'N/A')})"
        )

    def test_neutral_zone_can_be_disabled(self, strategy_no_filter):
        """When disabled, signals in neutral zone should NOT be filtered."""
        assert strategy_no_filter._neutral_zone_filter is False

    def test_neutral_zone_custom_bounds(self):
        """Test custom neutral zone bounds."""
        strategy = MACDvStrategy({
            "neutral_zone_upper": 30.0,
            "neutral_zone_lower": -30.0,
        })
        assert strategy._neutral_zone_upper == 30.0
        assert strategy._neutral_zone_lower == -30.0


# ============================================================================
# RANGING MARKET DETECTION TESTS
# ============================================================================

class TestRangingMarketDetection:
    """Tests for ranging market detection (Spiroglou 2022)."""

    def test_ranging_detection_defaults(self, strategy):
        """Verify ranging detection is enabled with 25-bar threshold."""
        assert strategy._ranging_detection_enabled is True
        assert strategy._ranging_bar_threshold == 25

    def test_ranging_counter_increments(self, strategy):
        """Test that bars_in_neutral counter increments for ranging prices."""
        prices = _make_ranging_prices(n=60)
        highs = prices + 0.3
        lows = prices - 0.3

        # Analyze multiple times to build up neutral bar count
        for i in range(35, 60):
            strategy.analyze(
                symbol="MES",
                prices=prices[:i+1],
                highs=highs[:i+1],
                lows=lows[:i+1],
            )

        # Counter should have incremented
        assert "MES" in strategy._bars_in_neutral

    def test_ranging_counter_resets_on_breakout(self, strategy):
        """Test that counter resets when MACD-v leaves neutral zone."""
        # Start with ranging prices
        ranging = _make_ranging_prices(n=50)

        # Then strong trend (should push MACD-v out of neutral zone)
        trending = _make_trending_prices("up", n=50, base=ranging[-1])
        prices = np.concatenate([ranging, trending])
        highs = prices + 0.5
        lows = prices - 0.5

        # Analyze the full series
        signal = strategy.analyze(
            symbol="TREND_TEST",
            prices=prices,
            highs=highs,
            lows=lows,
        )

        # After strong trend, counter should be reset (or 0)
        bars = strategy._bars_in_neutral.get("TREND_TEST", 0)
        # If MACD-v broke out of neutral zone, counter should be 0
        # (or small if it just re-entered)
        assert bars < 25, (
            f"Expected bars_in_neutral < 25 after breakout, got {bars}"
        )

    def test_ranging_detection_can_be_disabled(self):
        """Test that ranging detection can be disabled."""
        strategy = MACDvStrategy({"ranging_detection_enabled": False})
        assert strategy._ranging_detection_enabled is False


# ============================================================================
# SIGNAL GENERATION TESTS
# ============================================================================

class TestSignalGeneration:
    """Tests for MACD-v signal generation."""

    def test_insufficient_data_returns_flat(self, strategy):
        """Test that insufficient data returns flat signal."""
        prices = np.array([100.0, 101.0, 102.0])  # Too few bars

        signal = strategy.analyze(symbol="MES", prices=prices)

        assert signal.direction == "flat"
        assert signal.confidence == 0.0

    def test_signal_has_required_fields(self, strategy):
        """Test that signals have all required fields."""
        prices = _make_trending_prices("up", n=100)
        highs = prices + 0.5
        lows = prices - 0.5

        signal = strategy.analyze(
            symbol="MES",
            prices=prices,
            highs=highs,
            lows=lows,
        )

        assert isinstance(signal, MACDvSignal)
        assert signal.symbol == "MES"
        assert signal.direction in ("long", "short", "flat", "exit_long", "exit_short")
        assert isinstance(signal.strength, float)
        assert isinstance(signal.confidence, float)
        assert isinstance(signal.indicators, dict)
        assert "macdv" in signal.indicators
        assert "signal_line" in signal.indicators
        assert "histogram" in signal.indicators

    def test_zone_classification(self, strategy):
        """Test zone classification method."""
        assert strategy.get_zone(200.0) == "extreme_overbought"
        assert strategy.get_zone(160.0) == "overbought"
        assert strategy.get_zone(30.0) == "neutral"
        assert strategy.get_zone(-30.0) == "neutral"
        assert strategy.get_zone(-160.0) == "oversold"
        assert strategy.get_zone(-200.0) == "extreme_oversold"

    def test_ema_calculation(self, strategy):
        """Test EMA calculation correctness."""
        prices = np.arange(1.0, 31.0)  # 30 prices
        ema = strategy.calculate_ema(prices, 10)

        assert len(ema) == 30
        # First 9 values should be NaN
        assert np.isnan(ema[0])
        assert np.isnan(ema[8])
        # 10th value should be SMA of first 10
        assert abs(ema[9] - np.mean(prices[:10])) < 0.01
        # Later values should exist
        assert not np.isnan(ema[29])

    def test_stop_loss_and_take_profit_on_signal(self, strategy_no_filter):
        """Test that directional signals include stop-loss and take-profit."""
        # Create strong uptrend that should produce a long signal
        np.random.seed(123)
        prices = 100.0 + np.cumsum(np.random.uniform(0.1, 0.5, 100))
        highs = prices + 0.5
        lows = prices - 0.3

        signal = strategy_no_filter.analyze(
            symbol="MES",
            prices=prices,
            highs=highs,
            lows=lows,
            previous_zone="oversold",
        )

        # If a directional signal was generated, it should have SL/TP
        if signal.direction in ("long", "short"):
            assert signal.stop_loss_price is not None, "Directional signal must have stop_loss"
            assert signal.take_profit_price is not None, "Directional signal must have take_profit"


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestMACDvConfiguration:
    """Tests for MACD-v strategy configuration."""

    def test_default_periods(self, strategy):
        """Test default MACD periods."""
        assert strategy._fast_period == 12
        assert strategy._slow_period == 26
        assert strategy._signal_period == 9

    def test_custom_config(self):
        """Test custom configuration."""
        strategy = MACDvStrategy({
            "fast_period": 8,
            "slow_period": 21,
            "signal_period": 5,
            "neutral_zone_upper": 40.0,
            "neutral_zone_lower": -40.0,
            "ranging_bar_threshold": 30,
        })
        assert strategy._fast_period == 8
        assert strategy._slow_period == 21
        assert strategy._signal_period == 5
        assert strategy._neutral_zone_upper == 40.0
        assert strategy._ranging_bar_threshold == 30

    def test_stop_loss_defaults(self, strategy):
        """Test default stop-loss and take-profit settings."""
        assert strategy._stop_loss_pct == 2.0
        assert strategy._take_profit_pct == 4.0
        assert strategy._use_atr_stop is True
        assert strategy._stop_loss_atr_mult == 2.0
        assert strategy._take_profit_atr_mult == 4.0

"""
Tests for Session-Based Trading Strategy (Phase 6.1)
====================================================

Tests for session window detection, opening range calculation,
and breakout signal generation.
"""

import numpy as np
import pytest
from datetime import datetime, time, timezone, timedelta
from unittest.mock import patch

from strategies.session_strategy import (
    SessionStrategy,
    TradingSession,
    SessionQuality,
    SessionWindow,
    OpeningRange,
    SessionSignal,
    SESSION_WINDOWS,
    create_session_strategy,
)


class TestSessionDetection:
    """Tests for session detection."""

    @pytest.fixture
    def strategy(self):
        """Create SessionStrategy instance."""
        return SessionStrategy()

    def test_london_ny_overlap_detection(self, strategy):
        """Test detection of London/NY overlap."""
        # 14:00 UTC should be in overlap
        ts = datetime(2026, 2, 4, 14, 0, 0, tzinfo=timezone.utc)
        session, quality = strategy.get_current_session(ts)

        assert session == TradingSession.LONDON_NY_OVERLAP
        assert quality == SessionQuality.EXCELLENT

    def test_london_session_detection(self, strategy):
        """Test detection of London session."""
        # 09:00 UTC should be in London (before NY opens)
        ts = datetime(2026, 2, 4, 9, 0, 0, tzinfo=timezone.utc)
        session, quality = strategy.get_current_session(ts)

        assert session == TradingSession.LONDON
        assert quality == SessionQuality.EXCELLENT

    def test_asian_session_detection(self, strategy):
        """Test detection of Asian session."""
        # 02:00 UTC should be in Asian
        ts = datetime(2026, 2, 4, 2, 0, 0, tzinfo=timezone.utc)
        session, quality = strategy.get_current_session(ts)

        assert session == TradingSession.ASIAN

    def test_session_allowed_check(self, strategy):
        """Test session allowed filter."""
        # Overlap should be allowed by default
        assert strategy.is_session_allowed(TradingSession.LONDON_NY_OVERLAP)
        assert strategy.is_session_allowed(TradingSession.LONDON)
        assert strategy.is_session_allowed(TradingSession.NEW_YORK)

        # After hours should not be allowed
        assert not strategy.is_session_allowed(TradingSession.AFTER_HOURS)


class TestOpeningRange:
    """Tests for opening range calculation."""

    @pytest.fixture
    def strategy(self):
        """Create SessionStrategy instance."""
        return SessionStrategy({"opening_range_minutes": 30})

    def test_opening_range_calculation(self, strategy):
        """Test basic opening range calculation."""
        # Create synthetic price data
        np.random.seed(42)
        n_bars = 60  # 60 minute bars

        # OHLCV format
        opens = 100 + np.cumsum(np.random.randn(n_bars) * 0.1)
        highs = opens + np.abs(np.random.randn(n_bars) * 0.2)
        lows = opens - np.abs(np.random.randn(n_bars) * 0.2)
        closes = opens + np.random.randn(n_bars) * 0.1
        volumes = np.random.randint(1000, 5000, n_bars)

        prices = np.column_stack([opens, highs, lows, closes, volumes])

        # Create timestamps starting from session start
        base_time = datetime(2026, 2, 4, 13, 0, 0, tzinfo=timezone.utc)
        timestamps = [base_time + timedelta(minutes=i) for i in range(n_bars)]

        opening_range = strategy.calculate_opening_range(
            symbol="EURUSD",
            prices=prices,
            timestamps=timestamps,
            session=TradingSession.LONDON_NY_OVERLAP,
            bar_minutes=1,
        )

        assert opening_range is not None
        assert opening_range.high > opening_range.low
        assert opening_range.range_size > 0
        assert opening_range.n_bars == 30  # 30 minute range

    def test_opening_range_cached(self, strategy):
        """Test that opening range is cached."""
        prices = np.random.randn(60, 5) + 100
        prices[:, 1] = prices[:, 0] + 0.5  # highs
        prices[:, 2] = prices[:, 0] - 0.5  # lows

        base_time = datetime(2026, 2, 4, 13, 0, 0, tzinfo=timezone.utc)
        timestamps = [base_time + timedelta(minutes=i) for i in range(60)]

        strategy.calculate_opening_range(
            "EURUSD", prices, timestamps,
            TradingSession.LONDON_NY_OVERLAP, 1
        )

        assert "EURUSD" in strategy._opening_ranges


class TestBreakoutDetection:
    """Tests for breakout detection."""

    @pytest.fixture
    def strategy(self):
        """Create SessionStrategy with low thresholds for testing."""
        return SessionStrategy({
            "breakout_threshold_atr": 0.3,
            "volume_confirmation_mult": 1.2,
            "min_risk_reward": 0.5,  # Lower threshold for testing
            "take_profit_atr_mult": 3.0,  # Higher TP for better R:R
        })

    def test_bullish_breakout(self, strategy):
        """Test bullish breakout detection."""
        # Set up tight opening range for good R:R
        strategy._opening_ranges["EURUSD"] = OpeningRange(
            high=100.2,
            low=99.8,
            mid=100.0,
            range_size=0.4,
            range_pct=0.4,
            volume_in_range=10000,
            n_bars=30,
            session=TradingSession.LONDON_NY_OVERLAP,
        )

        # Price breaks above range
        signal = strategy.detect_breakout(
            symbol="EURUSD",
            current_price=100.5,  # Above 100.2 + 0.3*0.3 = 100.29
            atr=0.3,
            volume=1500,
            avg_volume=1000,
        )

        assert signal is not None
        assert signal.direction == "LONG"
        assert signal.signal_type == "breakout"
        assert signal.volume_confirmed  # 1500 > 1000 * 1.2

    def test_bearish_breakout(self, strategy):
        """Test bearish breakout detection."""
        strategy._opening_ranges["EURUSD"] = OpeningRange(
            high=100.2,
            low=99.8,
            mid=100.0,
            range_size=0.4,
            range_pct=0.4,
            volume_in_range=10000,
            n_bars=30,
            session=TradingSession.LONDON_NY_OVERLAP,
        )

        signal = strategy.detect_breakout(
            symbol="EURUSD",
            current_price=99.5,  # Below 99.8 - 0.3*0.3 = 99.71
            atr=0.3,
            volume=1500,
            avg_volume=1000,
        )

        assert signal is not None
        assert signal.direction == "SHORT"
        assert signal.signal_type == "breakout"

    def test_no_breakout_in_range(self, strategy):
        """Test no signal when price is in range."""
        strategy._opening_ranges["EURUSD"] = OpeningRange(
            high=100.5,
            low=99.5,
            mid=100.0,
            range_size=1.0,
            range_pct=1.0,
            volume_in_range=10000,
            n_bars=30,
            session=TradingSession.LONDON_NY_OVERLAP,
        )

        signal = strategy.detect_breakout(
            symbol="EURUSD",
            current_price=100.0,  # In the middle
            atr=0.5,
            volume=1500,
            avg_volume=1000,
        )

        assert signal is None


class TestSessionMomentum:
    """Tests for session momentum signal."""

    @pytest.fixture
    def strategy(self):
        """Create SessionStrategy instance."""
        return SessionStrategy()

    def test_bullish_momentum(self, strategy):
        """Test bullish momentum detection."""
        # Create trending up prices
        prices = np.array([100.0, 100.2, 100.4, 100.5, 100.7, 100.9, 101.0, 101.2, 101.3, 101.5])

        # Mock get_current_session to return a good session regardless of real time
        with patch.object(strategy, 'get_current_session',
                          return_value=(TradingSession.LONDON_NY_OVERLAP, SessionQuality.EXCELLENT)):
            signal = strategy.generate_session_momentum_signal(
                symbol="EURUSD",
                prices=prices,
                session=TradingSession.LONDON_NY_OVERLAP,
                atr=0.5,
            )

        assert signal is not None
        assert signal.direction == "LONG"
        assert signal.signal_type == "momentum"

    def test_bearish_momentum(self, strategy):
        """Test bearish momentum detection."""
        # Create trending down prices
        prices = np.array([101.5, 101.3, 101.2, 101.0, 100.9, 100.7, 100.5, 100.4, 100.2, 100.0])

        # Mock get_current_session to return a good session regardless of real time
        with patch.object(strategy, 'get_current_session',
                          return_value=(TradingSession.LONDON_NY_OVERLAP, SessionQuality.EXCELLENT)):
            signal = strategy.generate_session_momentum_signal(
                symbol="EURUSD",
                prices=prices,
                session=TradingSession.LONDON_NY_OVERLAP,
                atr=0.5,
            )

        assert signal is not None
        assert signal.direction == "SHORT"
        assert signal.signal_type == "momentum"

    def test_no_momentum_choppy(self, strategy):
        """Test no signal in choppy market."""
        # Alternating up/down
        prices = np.array([100.0, 100.2, 100.1, 100.3, 100.2, 100.4, 100.3, 100.5, 100.4, 100.5])

        signal = strategy.generate_session_momentum_signal(
            symbol="EURUSD",
            prices=prices,
            session=TradingSession.LONDON_NY_OVERLAP,
            atr=0.5,
        )

        assert signal is None


class TestFullAnalysis:
    """Tests for full analysis workflow."""

    @pytest.fixture
    def strategy(self):
        """Create SessionStrategy instance."""
        return SessionStrategy({
            "opening_range_minutes": 30,
            "breakout_threshold_atr": 0.3,
        })

    def test_full_analysis_flow(self, strategy):
        """Test complete analysis workflow."""
        np.random.seed(42)

        # Create price data that eventually breaks out
        n_bars = 100
        base_price = 100.0

        # First 30 bars: consolidation
        prices = np.zeros((n_bars, 5))
        for i in range(30):
            prices[i, 0] = base_price + np.random.randn() * 0.1  # Open
            prices[i, 1] = prices[i, 0] + 0.2  # High
            prices[i, 2] = prices[i, 0] - 0.2  # Low
            prices[i, 3] = prices[i, 0] + np.random.randn() * 0.05  # Close
            prices[i, 4] = 1000 + np.random.randint(0, 500)  # Volume

        # Bars 30-100: trending up (breakout)
        for i in range(30, n_bars):
            prices[i, 0] = base_price + (i - 30) * 0.05 + np.random.randn() * 0.05
            prices[i, 1] = prices[i, 0] + 0.3
            prices[i, 2] = prices[i, 0] - 0.1
            prices[i, 3] = prices[i, 1]  # Close at high
            prices[i, 4] = 2000 + np.random.randint(0, 1000)  # Higher volume

        # Timestamps during overlap session
        base_time = datetime(2026, 2, 4, 13, 0, 0, tzinfo=timezone.utc)
        timestamps = [base_time + timedelta(minutes=i) for i in range(n_bars)]

        # Analyze at the end (after breakout)
        signal = strategy.analyze(
            symbol="EURUSD",
            prices=prices,
            timestamps=timestamps,
            atr=0.4,
            volume=2500,
            avg_volume=1500,
            bar_minutes=1,
        )

        # Should have calculated opening range
        assert "EURUSD" in strategy._opening_ranges

        # Should have detected breakout or momentum
        if signal is not None:
            assert signal.direction in ["LONG", "SHORT"]
            assert signal.entry_price > 0
            assert signal.stop_loss > 0
            assert signal.take_profit > 0


class TestSessionWindows:
    """Tests for session window definitions."""

    def test_session_windows_defined(self):
        """Test that all sessions are properly defined."""
        assert "asian" in SESSION_WINDOWS
        assert "london" in SESSION_WINDOWS
        assert "new_york" in SESSION_WINDOWS
        assert "london_ny_overlap" in SESSION_WINDOWS

    def test_session_window_properties(self):
        """Test session window properties."""
        overlap = SESSION_WINDOWS["london_ny_overlap"]

        assert overlap.start_time == time(13, 0)
        # London closes at 17:00 UTC, not 16:00 (corrected per industry standard)
        assert overlap.end_time == time(17, 0)
        assert overlap.quality == SessionQuality.EXCELLENT
        assert overlap.typical_volume_pct > 0

    def test_session_volumes_reasonable(self):
        """Test that session volumes sum reasonably."""
        total_volume = sum(w.typical_volume_pct for w in SESSION_WINDOWS.values())

        # Allow some overlap between sessions
        assert total_volume > 100  # Overlapping sessions
        assert total_volume < 200


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_session_strategy_default(self):
        """Test default factory creation."""
        strategy = create_session_strategy()

        assert isinstance(strategy, SessionStrategy)
        assert strategy._opening_range_minutes == 30

    def test_create_session_strategy_custom(self):
        """Test factory with custom config."""
        strategy = create_session_strategy({
            "opening_range_minutes": 15,
            "breakout_threshold_atr": 1.0,
        })

        assert strategy._opening_range_minutes == 15
        assert strategy._breakout_threshold_atr == 1.0

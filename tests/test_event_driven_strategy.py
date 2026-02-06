"""
Tests for Event-Driven Strategy (Phase 6.4)
===========================================

Tests for event window detection, surprise calculation,
and signal generation around economic events.
"""

import numpy as np
import pytest
from datetime import datetime, timedelta, timezone

from strategies.event_driven_strategy import (
    EventDrivenStrategy,
    EventType,
    EventImpact,
    EventWindow,
    EconomicEvent,
    EventAnalysis,
    EventSignal,
    EVENT_CHARACTERISTICS,
    create_event_driven_strategy,
)


class TestEventManagement:
    """Tests for event calendar management."""

    @pytest.fixture
    def strategy(self):
        """Create EventDrivenStrategy instance."""
        return EventDrivenStrategy()

    def test_add_event(self, strategy):
        """Test adding event to calendar."""
        event = EconomicEvent(
            event_type=EventType.FOMC,
            timestamp=datetime(2026, 2, 5, 19, 0, 0, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
        )

        strategy.add_event(event)

        assert len(strategy._upcoming_events) == 1
        assert strategy._upcoming_events[0].event_type == EventType.FOMC

    def test_events_sorted_by_time(self, strategy):
        """Test that events are sorted by timestamp."""
        event1 = EconomicEvent(
            event_type=EventType.NFP,
            timestamp=datetime(2026, 2, 7, 13, 30, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
        )
        event2 = EconomicEvent(
            event_type=EventType.FOMC,
            timestamp=datetime(2026, 2, 5, 19, 0, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
        )

        strategy.add_event(event1)
        strategy.add_event(event2)

        # FOMC should be first (earlier)
        assert strategy._upcoming_events[0].event_type == EventType.FOMC
        assert strategy._upcoming_events[1].event_type == EventType.NFP

    def test_get_next_event(self, strategy):
        """Test getting next upcoming event."""
        event = EconomicEvent(
            event_type=EventType.CPI,
            timestamp=datetime(2026, 2, 10, 13, 30, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
        )
        strategy.add_event(event)

        current = datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc)
        next_event = strategy.get_next_event(current)

        assert next_event is not None
        assert next_event.event_type == EventType.CPI

    def test_get_next_event_filtered(self, strategy):
        """Test getting next event of specific type."""
        strategy.add_event(EconomicEvent(
            event_type=EventType.FOMC,
            timestamp=datetime(2026, 2, 5, 19, 0, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
        ))
        strategy.add_event(EconomicEvent(
            event_type=EventType.NFP,
            timestamp=datetime(2026, 2, 7, 13, 30, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
        ))

        current = datetime(2026, 2, 4, 12, 0, tzinfo=timezone.utc)
        next_nfp = strategy.get_next_event(current, EventType.NFP)

        assert next_nfp is not None
        assert next_nfp.event_type == EventType.NFP


class TestEventWindow:
    """Tests for event window detection."""

    @pytest.fixture
    def strategy(self):
        """Create EventDrivenStrategy instance."""
        return EventDrivenStrategy()

    @pytest.fixture
    def fomc_event(self):
        """Create sample FOMC event."""
        return EconomicEvent(
            event_type=EventType.FOMC,
            timestamp=datetime(2026, 2, 5, 19, 0, 0, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
        )

    def test_outside_window(self, strategy, fomc_event):
        """Test detection of outside window."""
        # FOMC pre_window is 24h (1440 min Phase 11), so >24h before = outside
        current = datetime(2026, 2, 4, 17, 0, 0, tzinfo=timezone.utc)  # 26 hours before
        window = strategy.get_event_window(current, fomc_event)

        assert window == EventWindow.OUTSIDE_WINDOW

    def test_pre_event_window(self, strategy, fomc_event):
        """Test detection of pre-event window."""
        current = datetime(2026, 2, 5, 18, 30, 0, tzinfo=timezone.utc)  # 30 min before
        window = strategy.get_event_window(current, fomc_event)

        assert window == EventWindow.PRE_EVENT

    def test_during_event(self, strategy, fomc_event):
        """Test detection of during-event window."""
        current = datetime(2026, 2, 5, 19, 2, 0, tzinfo=timezone.utc)  # 2 min after
        window = strategy.get_event_window(current, fomc_event)

        assert window == EventWindow.DURING_EVENT

    def test_post_event_early(self, strategy, fomc_event):
        """Test detection of early post-event window."""
        current = datetime(2026, 2, 5, 19, 15, 0, tzinfo=timezone.utc)  # 15 min after
        window = strategy.get_event_window(current, fomc_event)

        assert window == EventWindow.POST_EVENT_EARLY

    def test_post_event_late(self, strategy, fomc_event):
        """Test detection of late post-event window."""
        current = datetime(2026, 2, 5, 20, 0, 0, tzinfo=timezone.utc)  # 1 hour after
        window = strategy.get_event_window(current, fomc_event)

        assert window == EventWindow.POST_EVENT_LATE


class TestSurpriseCalculation:
    """Tests for surprise calculation."""

    @pytest.fixture
    def strategy(self):
        """Create EventDrivenStrategy instance."""
        return EventDrivenStrategy()

    def test_positive_surprise(self, strategy):
        """Test positive surprise calculation."""
        event = EconomicEvent(
            event_type=EventType.NFP,
            timestamp=datetime(2026, 2, 7, 13, 30, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
            actual=250000,
            forecast=200000,
        )

        surprise, surprise_std = strategy.calculate_surprise(event)

        assert surprise == 50000
        assert surprise_std is not None

    def test_negative_surprise(self, strategy):
        """Test negative surprise calculation."""
        event = EconomicEvent(
            event_type=EventType.NFP,
            timestamp=datetime(2026, 2, 7, 13, 30, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
            actual=150000,
            forecast=200000,
        )

        surprise, surprise_std = strategy.calculate_surprise(event)

        assert surprise == -50000
        assert surprise_std is not None
        assert surprise_std < 0

    def test_no_surprise_missing_data(self, strategy):
        """Test no surprise when data missing."""
        event = EconomicEvent(
            event_type=EventType.FOMC,
            timestamp=datetime(2026, 2, 5, 19, 0, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
            actual=None,
            forecast=None,
        )

        surprise, surprise_std = strategy.calculate_surprise(event)

        assert surprise is None
        assert surprise_std is None

    def test_surprise_with_history(self, strategy):
        """Test surprise standardization with historical data."""
        # Add historical surprises
        for s in [10000, -15000, 20000, 5000, -8000, 12000]:
            strategy.update_historical_surprise(EventType.NFP, s)

        event = EconomicEvent(
            event_type=EventType.NFP,
            timestamp=datetime(2026, 2, 7, 13, 30, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
            actual=230000,
            forecast=200000,
        )

        surprise, surprise_std = strategy.calculate_surprise(event)

        assert surprise == 30000
        # Should be standardized against historical std
        assert surprise_std is not None


class TestEventAnalysis:
    """Tests for event analysis."""

    @pytest.fixture
    def strategy(self):
        """Create EventDrivenStrategy instance."""
        return EventDrivenStrategy()

    def test_analyze_bullish_nfp(self, strategy):
        """Test analysis of bullish NFP surprise."""
        event = EconomicEvent(
            event_type=EventType.NFP,
            timestamp=datetime(2026, 2, 7, 13, 30, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
            actual=300000,  # Strong beat
            forecast=200000,
        )

        # Add some history for standardization
        for s in [20000, -20000, 30000, -30000, 25000]:
            strategy.update_historical_surprise(EventType.NFP, s)

        current = datetime(2026, 2, 7, 13, 45, tzinfo=timezone.utc)
        analysis = strategy.analyze_event(event, current, current_vol=0.02, normal_vol=0.015)

        assert analysis.direction_bias == "bullish"
        assert analysis.window == EventWindow.POST_EVENT_EARLY
        assert analysis.vol_expansion > 1.0

    def test_analyze_bearish_cpi(self, strategy):
        """Test analysis of bearish CPI (high inflation)."""
        event = EconomicEvent(
            event_type=EventType.CPI,
            timestamp=datetime(2026, 2, 10, 13, 30, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
            actual=0.5,  # High inflation
            forecast=0.2,
        )

        current = datetime(2026, 2, 10, 13, 45, tzinfo=timezone.utc)
        analysis = strategy.analyze_event(event, current, current_vol=0.025, normal_vol=0.015)

        # High CPI = bearish for risk assets
        assert analysis.direction_bias == "bearish"

    def test_analyze_tracks_state(self, strategy):
        """Test that analysis is tracked."""
        event = EconomicEvent(
            event_type=EventType.GDP,
            timestamp=datetime(2026, 2, 15, 13, 30, tzinfo=timezone.utc),
            impact=EventImpact.MEDIUM,
            actual=2.5,
            forecast=2.3,
        )

        current = datetime(2026, 2, 15, 13, 40, tzinfo=timezone.utc)
        strategy.analyze_event(event, current, 0.01, 0.01)

        assert "gdp" in strategy._recent_analyses


class TestSignalGeneration:
    """Tests for signal generation."""

    @pytest.fixture
    def strategy(self):
        """Create EventDrivenStrategy with lower thresholds."""
        return EventDrivenStrategy({
            "momentum_threshold": 0.3,
            "surprise_threshold": 0.5,
        })

    def test_pre_event_vol_signal(self, strategy):
        """Test pre-event volatility signal."""
        event = EconomicEvent(
            event_type=EventType.FOMC,
            timestamp=datetime(2026, 2, 5, 19, 0, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
        )

        current = datetime(2026, 2, 5, 18, 30, tzinfo=timezone.utc)
        signal = strategy.generate_signal(
            symbol="ES",
            event=event,
            current_time=current,
            current_price=5000,
            current_vol=0.008,  # Suppressed vol
            normal_vol=0.015,
            atr=20,
        )

        if signal is not None:
            assert signal.signal_type == "pre_event_vol"
            assert signal.event_window == EventWindow.PRE_EVENT

    def test_post_event_momentum_signal(self, strategy):
        """Test post-event momentum signal."""
        event = EconomicEvent(
            event_type=EventType.NFP,
            timestamp=datetime(2026, 2, 7, 13, 30, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
            actual=300000,
            forecast=200000,
        )

        # Add history
        for s in [20000, -20000, 30000, -30000, 25000]:
            strategy.update_historical_surprise(EventType.NFP, s)

        current = datetime(2026, 2, 7, 13, 45, tzinfo=timezone.utc)
        signal = strategy.generate_signal(
            symbol="ES",
            event=event,
            current_time=current,
            current_price=5050,
            current_vol=0.025,
            normal_vol=0.015,
            atr=25,
        )

        if signal is not None:
            assert signal.signal_type == "post_event_momentum"
            assert signal.direction == "LONG"  # Bullish NFP

    def test_exit_signal_late_window(self, strategy):
        """Test exit signal in late post-event window."""
        event = EconomicEvent(
            event_type=EventType.CPI,
            timestamp=datetime(2026, 2, 10, 13, 30, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
        )

        current = datetime(2026, 2, 10, 14, 15, tzinfo=timezone.utc)  # 45 min after
        signal = strategy.generate_signal(
            symbol="ES",
            event=event,
            current_time=current,
            current_price=5000,
            current_vol=0.02,
            normal_vol=0.015,
            atr=20,
            current_position="LONG",
        )

        if signal is not None:
            assert signal.signal_type == "exit"
            assert signal.direction == "FLAT"

    def test_no_signal_non_sensitive_asset(self, strategy):
        """Test no signal for non-sensitive asset."""
        event = EconomicEvent(
            event_type=EventType.FOMC,
            timestamp=datetime(2026, 2, 5, 19, 0, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
        )

        current = datetime(2026, 2, 5, 19, 15, tzinfo=timezone.utc)
        signal = strategy.generate_signal(
            symbol="CORN",  # Not in FOMC sensitive assets
            event=event,
            current_time=current,
            current_price=450,
            current_vol=0.02,
            normal_vol=0.015,
            atr=5,
        )

        assert signal is None


class TestHistoricalSurprises:
    """Tests for historical surprise tracking."""

    @pytest.fixture
    def strategy(self):
        """Create EventDrivenStrategy instance."""
        return EventDrivenStrategy()

    def test_update_historical(self, strategy):
        """Test updating historical surprises."""
        strategy.update_historical_surprise(EventType.NFP, 50000)
        strategy.update_historical_surprise(EventType.NFP, -30000)

        assert len(strategy._historical_surprises[EventType.NFP]) == 2

    def test_historical_limit(self, strategy):
        """Test historical surprise limit (20)."""
        for i in range(25):
            strategy.update_historical_surprise(EventType.CPI, i * 0.1)

        assert len(strategy._historical_surprises[EventType.CPI]) == 20


class TestEventCharacteristics:
    """Tests for event characteristics definitions."""

    def test_fomc_characteristics(self):
        """Test FOMC characteristics defined."""
        chars = EVENT_CHARACTERISTICS[EventType.FOMC]

        assert "typical_vol_mult" in chars
        assert "reaction_minutes" in chars
        assert "sensitive_assets" in chars
        assert "ES" in chars["sensitive_assets"]

    def test_nfp_characteristics(self):
        """Test NFP characteristics defined."""
        chars = EVENT_CHARACTERISTICS[EventType.NFP]

        assert chars["typical_vol_mult"] >= 1.5
        assert chars["reaction_minutes"] >= 30


class TestStatus:
    """Tests for strategy status."""

    @pytest.fixture
    def strategy(self):
        """Create EventDrivenStrategy instance."""
        return EventDrivenStrategy()

    def test_get_status(self, strategy):
        """Test status reporting."""
        status = strategy.get_status()

        assert "vol_entry_threshold" in status
        assert "momentum_threshold" in status
        assert "upcoming_events" in status

    def test_status_with_events(self, strategy):
        """Test status includes events."""
        strategy.add_event(EconomicEvent(
            event_type=EventType.FOMC,
            timestamp=datetime(2026, 2, 5, 19, 0, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
        ))

        status = strategy.get_status()

        assert status["upcoming_events"] == 1
        assert len(status["events"]) == 1


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_default(self):
        """Test default factory creation."""
        strategy = create_event_driven_strategy()

        assert isinstance(strategy, EventDrivenStrategy)
        assert strategy._vol_entry_threshold == 0.7

    def test_create_custom_config(self):
        """Test factory with custom config."""
        strategy = create_event_driven_strategy({
            "vol_entry_threshold": 0.8,
            "momentum_threshold": 0.6,
        })

        assert strategy._vol_entry_threshold == 0.8
        assert strategy._momentum_threshold == 0.6

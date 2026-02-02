"""
Tests for Event System
======================
"""

import pytest
from datetime import datetime, timezone

from core.events import (
    Event,
    MarketDataEvent,
    SignalEvent,
    DecisionEvent,
    ValidatedDecisionEvent,
    SignalDirection,
    OrderSide,
    OrderType,
)


class TestEvents:
    """Test event creation and serialization."""

    def test_market_data_event_creation(self):
        """Test MarketDataEvent creation."""
        event = MarketDataEvent(
            source_agent="test_broker",
            symbol="AAPL",
            exchange="SMART",
            bid=175.50,
            ask=175.55,
            last=175.52,
            volume=1000,
        )

        assert event.symbol == "AAPL"
        assert event.mid == pytest.approx(175.525)
        assert event.spread == pytest.approx(0.05)
        assert event.source_agent == "test_broker"

    def test_signal_event_creation(self):
        """Test SignalEvent creation."""
        event = SignalEvent(
            source_agent="MacroAgent",
            strategy_name="macro_vix",
            symbol="SPY",
            direction=SignalDirection.LONG,
            strength=0.75,
            confidence=0.8,
            rationale="VIX below 15, risk-on regime",
            data_sources=("VIX", "IB_market_data"),
        )

        assert event.symbol == "SPY"
        assert event.direction == SignalDirection.LONG
        assert event.strength == 0.75
        assert len(event.data_sources) == 2

    def test_decision_event_creation(self):
        """Test DecisionEvent creation."""
        event = DecisionEvent(
            source_agent="CIOAgent",
            symbol="AAPL",
            action=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            rationale="Strong momentum signals",
            contributing_signals=("signal_1", "signal_2"),
            data_sources=("AAPL", "IB_market_data"),
            conviction_score=0.85,
        )

        assert event.symbol == "AAPL"
        assert event.action == OrderSide.BUY
        assert event.quantity == 100
        assert event.conviction_score == 0.85

    def test_event_immutability(self):
        """Test that events are immutable (Issue #18 - verify original value preserved)."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=100.0,
            ask=100.05,
        )

        # Store original value
        original_symbol = event.symbol

        with pytest.raises(AttributeError):
            event.symbol = "MSFT"

        # Assert original value is preserved after failed mutation attempt
        assert event.symbol == original_symbol
        assert event.symbol == "AAPL"

    def test_event_audit_dict(self):
        """Test event serialization for audit."""
        event = SignalEvent(
            source_agent="TestAgent",
            strategy_name="test_strategy",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.5,
            confidence=0.7,
            rationale="Test rationale",
            data_sources=("source1",),
        )

        audit_dict = event.to_audit_dict()

        assert "event_id" in audit_dict
        assert "timestamp" in audit_dict
        assert audit_dict["symbol"] == "AAPL"
        assert audit_dict["direction"] == "long"
        assert audit_dict["rationale"] == "Test rationale"

    def test_validated_decision_approved(self):
        """Test ValidatedDecisionEvent for approved decision."""
        event = ValidatedDecisionEvent(
            source_agent="RiskComplianceAgent",
            original_decision_id="decision_123",
            approved=True,
            compliance_checks=("position_limit", "daily_loss", "rate_limit"),
        )

        assert event.approved is True
        assert len(event.compliance_checks) == 3

    def test_validated_decision_rejected(self):
        """Test ValidatedDecisionEvent for rejected decision."""
        event = ValidatedDecisionEvent(
            source_agent="RiskComplianceAgent",
            original_decision_id="decision_456",
            approved=False,
            rejection_reason="Position size exceeds limit",
            compliance_checks=("position_limit",),
        )

        assert event.approved is False
        assert "exceeds limit" in event.rejection_reason


class TestSignalDirection:
    """Test SignalDirection enum."""

    def test_signal_directions(self):
        """Test all signal directions."""
        assert SignalDirection.LONG.value == "long"
        assert SignalDirection.SHORT.value == "short"
        assert SignalDirection.FLAT.value == "flat"


class TestOrderTypes:
    """Test order-related enums."""

    def test_order_sides(self):
        """Test order sides."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_types(self):
        """Test order types."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"


class TestMarketDataEdgeCases:
    """Test edge cases for MarketDataEvent properties."""

    def test_market_data_mid_price_fallback(self):
        """Test mid price falls back to last when bid/ask are zero."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=0.0,
            ask=0.0,
            last=175.52
        )
        # When bid and ask are both 0, mid should fall back to last price
        assert event.mid == 175.52

    def test_market_data_spread_zero_bid_ask(self):
        """Test spread is zero when bid and ask are both zero."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=0.0,
            ask=0.0,
            last=175.52
        )
        # When bid and ask are both 0, spread should be 0
        assert event.spread == 0.0

    def test_market_data_mid_price_normal(self):
        """Test mid price calculation with normal bid/ask."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=100.0,
            ask=100.10,
            last=100.05
        )
        assert event.mid == pytest.approx(100.05)

    def test_market_data_spread_normal(self):
        """Test spread calculation with normal bid/ask."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=100.0,
            ask=100.10,
            last=100.05
        )
        assert event.spread == pytest.approx(0.10)

    def test_market_data_mid_only_bid_zero(self):
        """Test mid calculation when only bid is zero."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=0.0,
            ask=100.10,
            last=100.05
        )
        # When bid is 0 (invalid), mid falls back to last price
        assert event.mid == pytest.approx(100.05)

    def test_market_data_spread_only_bid_zero(self):
        """Test spread when only bid is zero."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=0.0,
            ask=100.10,
            last=100.05
        )
        # When bid is 0 (invalid), spread is 0 (can't calculate)
        assert event.spread == pytest.approx(0.0)

    def test_market_data_mid_only_ask_zero(self):
        """Test mid calculation when only ask is zero (Issue #26 edge case)."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=100.0,
            ask=0.0,
            last=100.05
        )
        # When ask is 0, mid should fall back to last
        assert event.mid == 100.05

    def test_market_data_spread_only_ask_zero(self):
        """Test spread when only ask is zero (Issue #26 edge case)."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=100.0,
            ask=0.0,
            last=100.05
        )
        # When ask is 0, spread should be 0
        assert event.spread == 0.0

    def test_market_data_mid_equal_bid_ask(self):
        """Test mid when bid equals ask (locked market, Issue #26 edge case)."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=100.0,
            ask=100.0,
            last=100.0
        )
        assert event.mid == pytest.approx(100.0)
        assert event.spread == pytest.approx(0.0)

    def test_market_data_mid_wide_spread(self):
        """Test mid with wide spread (Issue #26 edge case)."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=95.0,
            ask=105.0,
            last=100.0
        )
        assert event.mid == pytest.approx(100.0)
        assert event.spread == pytest.approx(10.0)

    def test_market_data_mid_negative_bid(self):
        """Test mid with negative bid (should not happen but edge case)."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=-1.0,
            ask=100.0,
            last=100.0
        )
        # Negative bid is treated as invalid (< 0 check fails)
        assert event.mid == 100.0  # Falls back to last

    def test_market_data_all_zero_prices(self):
        """Test when all prices are zero (Issue #26 edge case)."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=0.0,
            ask=0.0,
            last=0.0
        )
        assert event.mid == 0.0
        assert event.spread == 0.0

    def test_market_data_very_small_spread(self):
        """Test mid/spread with very small values (penny stocks, Issue #26)."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="PENNY",
            bid=0.01,
            ask=0.02,
            last=0.015
        )
        assert event.mid == pytest.approx(0.015)
        assert event.spread == pytest.approx(0.01)

    def test_market_data_large_values(self):
        """Test mid/spread with large price values (Issue #26 edge case)."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="BRKA",
            bid=500000.0,
            ask=500100.0,
            last=500050.0
        )
        assert event.mid == pytest.approx(500050.0)
        assert event.spread == pytest.approx(100.0)

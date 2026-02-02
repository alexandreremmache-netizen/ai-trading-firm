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
        """Test that events are immutable."""
        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=100.0,
            ask=100.05,
        )

        with pytest.raises(AttributeError):
            event.symbol = "MSFT"

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

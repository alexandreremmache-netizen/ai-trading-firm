"""
Tests for StopLossManager
=========================

Tests for automatic stop-loss and exit rule management.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from core.stop_loss_manager import StopLossManager, StopLossConfig, PositionEntry
from core.exit_rules import ExitRuleManager, StopLossRule, TakeProfitRule, ExitRuleType
from core.events import OrderSide


class TestStopLossConfig:
    """Tests for StopLossConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values (Phase 12 updated)."""
        config = StopLossConfig()

        assert config.use_atr_stops is True
        assert config.atr_multiplier == 2.5
        assert config.fixed_stop_loss_pct == 5.0
        assert config.trailing_stop_enabled is True
        assert config.trailing_activation_pct == 1.0  # Phase 12: was 2.0
        assert config.trailing_distance_pct == 1.5  # Phase 12: was 3.0
        assert config.take_profit_enabled is True
        assert config.take_profit_pct == 4.0  # Phase 12: was 10.0
        assert config.drawdown_close_threshold_pct == 15.0
        # Phase 12: Breakeven settings
        assert config.breakeven_enabled is True
        assert config.breakeven_activation_pct == 1.5
        assert config.breakeven_buffer_pct == 0.1
        assert config.max_holding_hours == 48.0  # Phase 12: was 0.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StopLossConfig(
            use_atr_stops=False,
            atr_multiplier=3.0,
            fixed_stop_loss_pct=7.5,
            trailing_stop_enabled=False,
        )

        assert config.use_atr_stops is False
        assert config.atr_multiplier == 3.0
        assert config.fixed_stop_loss_pct == 7.5
        assert config.trailing_stop_enabled is False


class TestPositionEntry:
    """Tests for PositionEntry tracking."""

    def test_long_position_creation(self):
        """Test creating a long position entry."""
        pos = PositionEntry(
            symbol="AAPL",
            entry_price=150.0,
            quantity=100,
            is_long=True,
            entry_time=datetime.now(timezone.utc),
        )

        assert pos.symbol == "AAPL"
        assert pos.entry_price == 150.0
        assert pos.quantity == 100
        assert pos.is_long is True
        assert pos.highest_price == 0.0  # Will be updated on first price update
        assert pos.stop_loss_price is None
        assert pos.take_profit_price is None

    def test_short_position_creation(self):
        """Test creating a short position entry."""
        pos = PositionEntry(
            symbol="TSLA",
            entry_price=200.0,
            quantity=50,
            is_long=False,
            entry_time=datetime.now(timezone.utc),
        )

        assert pos.is_long is False
        assert pos.lowest_price == float("inf")


class TestStopLossManager:
    """Tests for StopLossManager."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = MagicMock()
        bus.publish = AsyncMock()
        return bus

    @pytest.fixture
    def mock_audit_logger(self):
        """Create mock audit logger."""
        logger = MagicMock()
        return logger

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return StopLossConfig(
            use_atr_stops=False,  # Use fixed stops for predictable testing
            fixed_stop_loss_pct=5.0,
            trailing_stop_enabled=False,
            take_profit_enabled=True,
            take_profit_pct=10.0,
            trailing_take_profit=False,
        )

    @pytest.fixture
    def manager(self, config, mock_event_bus, mock_audit_logger):
        """Create StopLossManager instance."""
        return StopLossManager(
            config=config,
            event_bus=mock_event_bus,
            audit_logger=mock_audit_logger,
            broker=None,
        )

    def test_calculate_stop_loss_fixed(self, manager):
        """Test fixed percentage stop-loss calculation."""
        # Long position
        stop_price = manager._calculate_stop_loss(100.0, is_long=True, atr=None)
        assert stop_price == 95.0  # 100 - 5%

        # Short position
        stop_price = manager._calculate_stop_loss(100.0, is_long=False, atr=None)
        assert stop_price == 105.0  # 100 + 5%

    def test_calculate_stop_loss_atr(self):
        """Test ATR-based stop-loss calculation."""
        config = StopLossConfig(use_atr_stops=True, atr_multiplier=2.0)
        manager = StopLossManager(
            config=config,
            event_bus=MagicMock(),
            audit_logger=MagicMock(),
        )

        # Long position with ATR=5
        stop_price = manager._calculate_stop_loss(100.0, is_long=True, atr=5.0)
        assert stop_price == 90.0  # 100 - (5 * 2)

        # Short position with ATR=5
        stop_price = manager._calculate_stop_loss(100.0, is_long=False, atr=5.0)
        assert stop_price == 110.0  # 100 + (5 * 2)

    def test_calculate_take_profit(self, manager):
        """Test take-profit calculation."""
        # Long position
        tp_price = manager._calculate_take_profit(100.0, is_long=True)
        assert tp_price == 110.0  # 100 + 10%

        # Short position
        tp_price = manager._calculate_take_profit(100.0, is_long=False)
        assert tp_price == 90.0  # 100 - 10%

    def test_register_position(self, manager):
        """Test position registration."""
        manager.register_position(
            symbol="AAPL",
            entry_price=150.0,
            quantity=100,
            is_long=True,
        )

        assert "AAPL" in manager._positions
        pos = manager._positions["AAPL"]
        assert pos.entry_price == 150.0
        assert pos.quantity == 100
        assert pos.stop_loss_price == 142.5  # 150 - 5%
        assert pos.take_profit_price == 165.0  # 150 + 10%

    def test_register_position_with_override(self, manager):
        """Test position registration with overridden stop/take profit."""
        manager.register_position(
            symbol="MSFT",
            entry_price=300.0,
            quantity=50,
            is_long=True,
            stop_loss_price=280.0,  # Custom stop
            take_profit_price=330.0,  # Custom take profit
        )

        pos = manager._positions["MSFT"]
        # Note: The exit rules manager calculates its own stops based on percentage
        # The custom prices are stored in the position entry
        assert pos.stop_loss_price == 280.0
        assert pos.take_profit_price == 330.0

    def test_update_price(self, manager):
        """Test price update."""
        manager.update_price("AAPL", 155.0)
        assert manager._last_prices["AAPL"] == 155.0

    def test_close_position(self, manager):
        """Test position closing."""
        manager.register_position(
            symbol="GOOGL",
            entry_price=100.0,
            quantity=20,
            is_long=True,
        )

        assert "GOOGL" in manager._positions
        initial_count = manager._stats["positions_tracked"]

        manager.close_position("GOOGL")

        assert "GOOGL" not in manager._positions
        assert manager._stats["positions_tracked"] == initial_count - 1

    def test_get_position_status(self, manager):
        """Test getting position status."""
        manager.register_position(
            symbol="NVDA",
            entry_price=500.0,
            quantity=10,
            is_long=True,
        )
        manager.update_price("NVDA", 525.0)

        # Update internal position price
        manager._positions["NVDA"].current_price = 525.0

        status = manager.get_position_status("NVDA")

        assert status is not None
        assert status["symbol"] == "NVDA"
        assert status["entry_price"] == 500.0
        assert status["quantity"] == 10
        assert status["current_price"] == 525.0
        assert status["pnl_pct"] == pytest.approx(5.0, rel=0.01)

    def test_get_all_positions(self, manager):
        """Test getting all positions."""
        manager.register_position("AAPL", 150.0, 100, True)
        manager.register_position("MSFT", 300.0, 50, True)

        positions = manager.get_all_positions()
        symbols = [p["symbol"] for p in positions]

        assert len(positions) == 2
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_get_stats(self, manager):
        """Test statistics reporting."""
        stats = manager.get_stats()

        assert "positions_tracked" in stats
        assert "stops_triggered" in stats
        assert "trailing_stops_triggered" in stats
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_close_worst_positions(self, manager, mock_event_bus):
        """Test closing worst performing positions."""
        # Register multiple positions
        manager.register_position("WINNER", 100.0, 50, True)
        manager.register_position("LOSER1", 100.0, 50, True)
        manager.register_position("LOSER2", 100.0, 50, True)

        # Set current prices (LOSER1 worst, LOSER2 medium, WINNER best)
        manager._positions["WINNER"].current_price = 110.0  # +10%
        manager._positions["LOSER1"].current_price = 85.0  # -15%
        manager._positions["LOSER2"].current_price = 95.0  # -5%

        # Close 2 worst positions
        closed = await manager.close_worst_positions(n_positions=2, reason="test")

        # LOSER1 and LOSER2 should be closed
        assert len(closed) == 2
        assert "LOSER1" in closed
        assert "LOSER2" in closed
        assert "WINNER" not in closed

        # Verify decision events were published
        assert mock_event_bus.publish.call_count == 2


class TestExitRuleIntegration:
    """Tests for ExitRuleManager integration."""

    def test_stop_loss_triggers_at_threshold(self):
        """Test that stop loss triggers at percentage threshold."""
        manager = ExitRuleManager()
        manager.add_stop_loss(StopLossRule(
            symbol="TEST",
            percentage_threshold=5.0,
            trailing=False,
        ))
        manager.register_position("TEST", entry_price=100.0, quantity=100, is_long=True)

        # Price at -4% - should not trigger
        signals = manager.evaluate("TEST", current_price=96.0)
        assert len(signals) == 0

        # Price at -5% - should trigger
        signals = manager.evaluate("TEST", current_price=95.0)
        assert len(signals) == 1
        assert signals[0].rule_type == ExitRuleType.STOP_LOSS

    def test_trailing_stop_triggers_on_pullback(self):
        """Test that trailing stop triggers on pullback from high."""
        manager = ExitRuleManager()
        manager.add_stop_loss(StopLossRule(
            symbol="TEST",
            percentage_threshold=5.0,  # Not used for trailing
            trailing=True,
            trailing_distance_pct=3.0,  # 3% from peak
        ))
        manager.register_position("TEST", entry_price=100.0, quantity=100, is_long=True)

        # Price rises to 110 (new high)
        signals = manager.evaluate("TEST", current_price=110.0)
        assert len(signals) == 0

        # Price drops to 108 (1.8% from peak) - should not trigger
        signals = manager.evaluate("TEST", current_price=108.0)
        assert len(signals) == 0

        # Price drops to 106 (3.6% from peak) - should trigger
        signals = manager.evaluate("TEST", current_price=106.0)
        assert len(signals) == 1
        assert signals[0].rule_type == ExitRuleType.TRAILING_STOP_LOSS

    def test_take_profit_triggers(self):
        """Test that take profit triggers at target."""
        manager = ExitRuleManager()
        manager.add_take_profit(TakeProfitRule(
            symbol="TEST",
            percentage_threshold=10.0,
            trailing=False,
        ))
        manager.register_position("TEST", entry_price=100.0, quantity=100, is_long=True)

        # Price at +8% - should not trigger
        signals = manager.evaluate("TEST", current_price=108.0)
        assert len(signals) == 0

        # Price at +10% - should trigger
        signals = manager.evaluate("TEST", current_price=110.0)
        assert len(signals) == 1
        assert signals[0].rule_type == ExitRuleType.TAKE_PROFIT

    def test_short_position_stops(self):
        """Test stops work correctly for short positions."""
        manager = ExitRuleManager()
        manager.add_stop_loss(StopLossRule(
            symbol="SHORT",
            percentage_threshold=5.0,
            trailing=False,
        ))
        manager.register_position("SHORT", entry_price=100.0, quantity=100, is_long=False)

        # Price drops (profit for short) - should not trigger
        signals = manager.evaluate("SHORT", current_price=95.0)
        assert len(signals) == 0

        # Price rises 5% (loss for short) - should trigger
        signals = manager.evaluate("SHORT", current_price=105.0)
        assert len(signals) == 1


class TestDrawdownManagement:
    """Tests for drawdown-based position management."""

    @pytest.fixture
    def manager(self):
        """Create manager with drawdown config."""
        config = StopLossConfig(
            drawdown_close_threshold_pct=15.0,
            close_worst_n_positions=2,
        )
        return StopLossManager(
            config=config,
            event_bus=MagicMock(publish=AsyncMock()),
            audit_logger=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_drawdown_closes_worst_not_all(self, manager):
        """Test that drawdown management closes worst positions, not all."""
        # Register 5 positions
        for i, (symbol, pnl) in enumerate([
            ("GOOD1", 5.0),    # +5%
            ("GOOD2", 3.0),    # +3%
            ("FLAT", 0.0),     # 0%
            ("BAD1", -8.0),    # -8%
            ("BAD2", -12.0),   # -12%
        ]):
            manager.register_position(symbol, 100.0, 50, True)
            # Simulate PnL by setting current price
            manager._positions[symbol].current_price = 100.0 * (1 + pnl / 100)

        # Close 2 worst positions
        closed = await manager.close_worst_positions(n_positions=2, reason="drawdown")

        # Only worst 2 should be closed
        assert "BAD2" in closed
        assert "BAD1" in closed
        assert len(closed) == 2

        # Good positions should remain
        assert "GOOD1" in manager._positions
        assert "GOOD2" in manager._positions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

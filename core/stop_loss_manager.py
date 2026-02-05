"""
Stop-Loss Manager
=================

Centralized stop-loss and exit management for automatic position protection.

This manager:
1. Automatically places stop-loss orders after position entry
2. Manages trailing stops for profitable positions
3. Evaluates exit rules on each price update
4. Generates close orders instead of triggering system-wide shutdown

Key difference from kill-switch:
- Kill-switch: Halts ALL trading (emergency measure)
- StopLossManager: Closes INDIVIDUAL losing positions (normal risk management)

Integration points:
- Receives FillEvent to track new positions
- Receives MarketDataEvent to evaluate stops
- Publishes DecisionEvent to close positions
- Works with ExitRuleManager for rule evaluation

Per CLAUDE.md:
- Implements proper position-level risk management
- Supports ATR-based dynamic stops (2-3x ATR)
- No infinite loops - event-driven evaluation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from core.exit_rules import (
    ExitRuleManager,
    TakeProfitRule,
    StopLossRule,
    TimeExitRule,
    ExitSignal,
    ExitRuleType,
)
from core.events import (
    Event,
    EventType,
    FillEvent,
    MarketDataEvent,
    DecisionEvent,
    OrderSide,
    OrderType,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger
    from core.broker import IBBroker


logger = logging.getLogger(__name__)


@dataclass
class PositionEntry:
    """Tracks a position for stop-loss management."""
    symbol: str
    entry_price: float
    quantity: int
    is_long: bool
    entry_time: datetime
    atr: float = 0.0  # ATR at entry for dynamic stops
    highest_price: float = 0.0  # For trailing stops
    lowest_price: float = float("inf")  # For short trailing
    current_price: float = 0.0
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    trailing_stop_enabled: bool = True
    trailing_distance_pct: float = 3.0  # 3% trailing by default


@dataclass
class StopLossConfig:
    """Configuration for stop-loss management."""
    # ATR-based stop-loss settings
    use_atr_stops: bool = True
    atr_multiplier: float = 2.5  # 2.5x ATR default
    atr_period: int = 14

    # QUICK WIN #4: Adaptive ATR multiplier based on volatility regime
    high_vol_threshold: float = 1.5    # Current/historical ATR ratio for high vol
    low_vol_threshold: float = 0.7     # Current/historical ATR ratio for low vol
    high_vol_multiplier: float = 1.3   # Scale factor in high vol (2.5 * 1.3 = 3.25x)
    low_vol_multiplier: float = 0.8    # Scale factor in low vol (2.5 * 0.8 = 2.0x)

    # Fixed percentage stop-loss (fallback when ATR unavailable)
    fixed_stop_loss_pct: float = 5.0  # 5% default

    # Trailing stop settings
    trailing_stop_enabled: bool = True
    trailing_activation_pct: float = 2.0  # Activate after 2% profit
    trailing_distance_pct: float = 3.0  # Trail by 3%

    # Take profit settings
    take_profit_enabled: bool = True
    take_profit_pct: float = 10.0  # 10% profit target
    trailing_take_profit: bool = True

    # Time-based exit
    max_holding_hours: float = 0.0  # 0 = disabled

    # Drawdown response (closes positions instead of kill-switch)
    drawdown_close_threshold_pct: float = 15.0  # Close worst positions at 15% drawdown
    close_worst_n_positions: int = 2  # Close N worst positions

    # Monitoring interval
    check_interval_seconds: float = 0.5  # Check stops every 500ms


class StopLossManager:
    """
    Centralized stop-loss and exit management.

    CRITICAL: This manager handles position-level risk by closing
    individual losing positions, NOT by triggering system-wide shutdown.

    The kill-switch (RiskAgent) is reserved for EMERGENCY situations only:
    - Connectivity loss
    - System anomalies
    - Regulatory halts
    - Manual intervention

    For normal market losses, StopLossManager:
    1. Closes individual positions hitting stop-loss
    2. Manages trailing stops to lock in profits
    3. Prevents portfolio-level drawdown by closing worst performers

    Usage:
        manager = StopLossManager(config, event_bus, audit_logger, broker)
        await manager.start()

        # On new position fill
        manager.register_position(fill_event)

        # On each price update (called by market data handler)
        close_signals = manager.evaluate_stops(symbol, price)

    Architecture:
        FillEvent -> StopLossManager -> (registers position, places bracket orders)
        MarketDataEvent -> StopLossManager -> (evaluates stops)
        StopLossManager -> DecisionEvent -> RiskAgent -> Execution
    """

    def __init__(
        self,
        config: StopLossConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
        broker: IBBroker | None = None,
    ):
        self._config = config
        self._event_bus = event_bus
        self._audit_logger = audit_logger
        self._broker = broker

        # Exit rule manager for rule evaluation
        self._exit_manager = ExitRuleManager()

        # Position tracking
        self._positions: dict[str, PositionEntry] = {}

        # ATR cache for dynamic stops
        self._atr_cache: dict[str, float] = {}

        # Price cache for stop evaluation
        self._last_prices: dict[str, float] = {}

        # Stop monitoring task
        self._monitor_task: asyncio.Task | None = None
        self._running = False

        # Statistics
        self._stats = {
            "positions_tracked": 0,
            "stops_triggered": 0,
            "trailing_stops_triggered": 0,
            "take_profits_triggered": 0,
            "time_exits_triggered": 0,
            "drawdown_closes": 0,
        }

        logger.info(
            f"StopLossManager initialized: ATR={config.use_atr_stops} "
            f"(mult={config.atr_multiplier}), trailing={config.trailing_stop_enabled}, "
            f"drawdown_close={config.drawdown_close_threshold_pct}%"
        )

    async def start(self) -> None:
        """Start stop monitoring loop."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("StopLossManager started")

    async def stop(self) -> None:
        """Stop stop monitoring loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("StopLossManager stopped")

    async def _monitor_loop(self) -> None:
        """Background loop to check stops."""
        while self._running:
            try:
                await self._check_all_stops()
                await asyncio.sleep(self._config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in stop monitor loop: {e}")
                await asyncio.sleep(1.0)  # Back off on error

    async def _check_all_stops(self) -> None:
        """Check all positions against their stops."""
        for symbol, position in list(self._positions.items()):
            if symbol in self._last_prices:
                current_price = self._last_prices[symbol]
                await self._evaluate_position_stops(position, current_price)

    async def _evaluate_position_stops(
        self,
        position: PositionEntry,
        current_price: float
    ) -> None:
        """Evaluate stops for a single position."""
        # Update position tracking
        position.current_price = current_price

        if position.is_long:
            position.highest_price = max(position.highest_price, current_price)
        else:
            position.lowest_price = min(position.lowest_price, current_price)

        # Use exit rule manager for evaluation
        exit_signals = self._exit_manager.evaluate(position.symbol, current_price)

        for signal in exit_signals:
            await self._handle_exit_signal(signal)

    async def _handle_exit_signal(self, signal: ExitSignal) -> None:
        """Handle an exit signal by generating a close decision."""
        logger.info(
            f"EXIT SIGNAL: {signal.rule_type.value} for {signal.symbol} - "
            f"reason={signal.trigger_reason.value}, PnL={signal.position_pnl_pct:.2f}%"
        )

        # Update statistics
        self._stats["stops_triggered"] += 1
        if signal.rule_type in (ExitRuleType.TRAILING_STOP_LOSS, ExitRuleType.TRAILING_TAKE_PROFIT):
            self._stats["trailing_stops_triggered"] += 1
        elif signal.rule_type == ExitRuleType.TAKE_PROFIT:
            self._stats["take_profits_triggered"] += 1
        elif signal.rule_type == ExitRuleType.TIME_EXIT:
            self._stats["time_exits_triggered"] += 1

        # Get position details
        position = self._positions.get(signal.symbol)
        if not position:
            return

        # Create close decision
        close_side = OrderSide.SELL if position.is_long else OrderSide.BUY

        decision = DecisionEvent(
            source_agent="StopLossManager",
            symbol=signal.symbol,
            action=close_side,
            quantity=signal.sell_quantity,
            order_type=OrderType.MARKET,  # Market order for immediate exit
            rationale=f"Stop-loss trigger: {signal.trigger_reason.value} at {signal.trigger_price:.2f}",
            data_sources=("stop_loss_manager", "exit_rules"),
            conviction_score=1.0,  # High conviction - this is risk management
        )

        # Publish to event bus for processing through risk/compliance/execution
        await self._event_bus.publish(decision)

        # Audit log the stop trigger
        self._audit_logger.log_event(decision)

        # Update position tracking if partial close
        if signal.sell_percentage >= 100:
            # Full close - remove position
            self._exit_manager.close_position(signal.symbol)
            if signal.symbol in self._positions:
                del self._positions[signal.symbol]
                self._stats["positions_tracked"] -= 1
        else:
            # Partial close - update quantity
            remaining_qty = position.quantity - signal.sell_quantity
            self._exit_manager.update_position(signal.symbol, quantity=remaining_qty)
            position.quantity = remaining_qty

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        is_long: bool = True,
        atr: float | None = None,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> None:
        """
        Register a new position for stop-loss tracking.

        Called after a fill confirms position entry.

        Args:
            symbol: Instrument symbol
            entry_price: Fill price
            quantity: Position size
            is_long: True for long, False for short
            atr: ATR value at entry (for dynamic stops)
            stop_loss_price: Override stop price (optional)
            take_profit_price: Override take profit price (optional)
        """
        # Calculate stops
        if stop_loss_price is None:
            stop_loss_price = self._calculate_stop_loss(
                entry_price, is_long, atr
            )

        if take_profit_price is None and self._config.take_profit_enabled:
            take_profit_price = self._calculate_take_profit(
                entry_price, is_long
            )

        # Create position entry
        position = PositionEntry(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            is_long=is_long,
            entry_time=datetime.now(timezone.utc),
            atr=atr or 0.0,
            highest_price=entry_price,
            lowest_price=entry_price,
            current_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            trailing_stop_enabled=self._config.trailing_stop_enabled,
            trailing_distance_pct=self._config.trailing_distance_pct,
        )

        self._positions[symbol] = position
        self._stats["positions_tracked"] += 1

        # Register with exit rule manager
        self._exit_manager.register_position(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            is_long=is_long,
        )

        # Add stop-loss rule
        stop_loss_pct = abs((stop_loss_price - entry_price) / entry_price * 100)
        self._exit_manager.add_stop_loss(StopLossRule(
            symbol=symbol,
            percentage_threshold=stop_loss_pct,
            trailing=self._config.trailing_stop_enabled,
            trailing_distance_pct=self._config.trailing_distance_pct,
            sell_percentage=100.0,
            priority=2,
        ))

        # Add take-profit rule if enabled
        if take_profit_price and self._config.take_profit_enabled:
            take_profit_pct = abs((take_profit_price - entry_price) / entry_price * 100)
            self._exit_manager.add_take_profit(TakeProfitRule(
                symbol=symbol,
                percentage_threshold=take_profit_pct,
                trailing=self._config.trailing_take_profit,
                trailing_distance_pct=2.0,
                sell_percentage=100.0,
                priority=1,
            ))

        # Add time exit rule if enabled
        if self._config.max_holding_hours > 0:
            self._exit_manager.add_time_exit(TimeExitRule(
                symbol=symbol,
                max_holding_hours=self._config.max_holding_hours,
                sell_percentage=100.0,
                priority=0,
            ))

        target_str = f"${take_profit_price:.2f}" if take_profit_price else "N/A"
        logger.info(
            f"Position registered: {'LONG' if is_long else 'SHORT'} {quantity} {symbol} "
            f"@ ${entry_price:.2f}, stop=${stop_loss_price:.2f}, "
            f"target={target_str}"
        )

    # =========================================================================
    # QUICK WIN #4: Adaptive ATR multiplier based on volatility regime
    # =========================================================================
    def _calculate_adaptive_atr_multiplier(
        self,
        current_atr: float,
        historical_atr: float | None = None,
    ) -> float:
        """
        QUICK WIN #4: Adjust ATR multiplier based on volatility regime.

        High volatility: Wider stops (prevent whipsaws)
        Low volatility: Tighter stops (capture smaller moves)

        Args:
            current_atr: Current ATR value
            historical_atr: Historical ATR for comparison (optional)

        Returns:
            Adjusted ATR multiplier
        """
        base_multiplier = self._config.atr_multiplier

        if historical_atr is None or historical_atr <= 0:
            return base_multiplier

        vol_ratio = current_atr / historical_atr

        if vol_ratio > self._config.high_vol_threshold:
            # High volatility regime - wider stops
            adjusted = base_multiplier * self._config.high_vol_multiplier
            logger.debug(f"High vol regime (ratio={vol_ratio:.2f}): ATR mult {base_multiplier} -> {adjusted:.2f}")
            return adjusted
        elif vol_ratio < self._config.low_vol_threshold:
            # Low volatility regime - tighter stops
            adjusted = base_multiplier * self._config.low_vol_multiplier
            logger.debug(f"Low vol regime (ratio={vol_ratio:.2f}): ATR mult {base_multiplier} -> {adjusted:.2f}")
            return adjusted
        else:
            # Normal regime
            return base_multiplier

    def _calculate_stop_loss(
        self,
        entry_price: float,
        is_long: bool,
        atr: float | None,
        historical_atr: float | None = None,
    ) -> float:
        """Calculate stop-loss price with adaptive ATR multiplier."""
        if self._config.use_atr_stops and atr and atr > 0:
            # QUICK WIN #4: Use adaptive multiplier
            multiplier = self._calculate_adaptive_atr_multiplier(atr, historical_atr)
            stop_distance = atr * multiplier
        else:
            # Fixed percentage stop
            stop_distance = entry_price * (self._config.fixed_stop_loss_pct / 100)

        if is_long:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def _calculate_take_profit(
        self,
        entry_price: float,
        is_long: bool,
    ) -> float:
        """Calculate take-profit price."""
        profit_distance = entry_price * (self._config.take_profit_pct / 100)

        if is_long:
            return entry_price + profit_distance
        else:
            return entry_price - profit_distance

    def update_price(self, symbol: str, price: float) -> None:
        """
        Update price for stop evaluation.

        Called by market data handler.
        """
        self._last_prices[symbol] = price

    def update_atr(self, symbol: str, atr: float) -> None:
        """Update ATR cache for dynamic stops."""
        self._atr_cache[symbol] = atr

    def close_position(self, symbol: str) -> None:
        """Remove a position from tracking (after successful close)."""
        if symbol in self._positions:
            del self._positions[symbol]
            self._stats["positions_tracked"] -= 1
        self._exit_manager.close_position(symbol)

    async def close_worst_positions(
        self,
        n_positions: int = 2,
        reason: str = "drawdown_management"
    ) -> list[str]:
        """
        Close the N worst performing positions.

        Called by RiskAgent when drawdown exceeds threshold instead of
        triggering kill-switch. This is POSITION-LEVEL risk management.

        Args:
            n_positions: Number of worst positions to close
            reason: Reason for closing (for audit)

        Returns:
            List of symbols closed
        """
        if not self._positions:
            return []

        # Calculate PnL for all positions
        position_pnl = []
        for symbol, pos in self._positions.items():
            if pos.current_price > 0:
                if pos.is_long:
                    pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price * 100
                else:
                    pnl_pct = (pos.entry_price - pos.current_price) / pos.entry_price * 100
                position_pnl.append((symbol, pnl_pct, pos))

        # Sort by PnL ascending (worst first)
        position_pnl.sort(key=lambda x: x[1])

        # Close worst N positions
        closed_symbols = []
        for symbol, pnl_pct, pos in position_pnl[:n_positions]:
            logger.warning(
                f"DRAWDOWN MANAGEMENT: Closing {symbol} (PnL: {pnl_pct:.2f}%) - {reason}"
            )

            # Generate close decision
            close_side = OrderSide.SELL if pos.is_long else OrderSide.BUY

            decision = DecisionEvent(
                source_agent="StopLossManager",
                symbol=symbol,
                action=close_side,
                quantity=pos.quantity,
                order_type=OrderType.MARKET,
                rationale=f"Drawdown management: {reason}, PnL={pnl_pct:.2f}%",
                data_sources=("stop_loss_manager", "risk_management"),
                conviction_score=1.0,
            )

            await self._event_bus.publish(decision)
            self._audit_logger.log_event(decision)

            closed_symbols.append(symbol)
            self._stats["drawdown_closes"] += 1

        return closed_symbols

    def get_position_status(self, symbol: str) -> dict[str, Any] | None:
        """Get status of a tracked position."""
        position = self._positions.get(symbol)
        if not position:
            return None

        # Calculate PnL
        if position.current_price > 0:
            if position.is_long:
                pnl_pct = (position.current_price - position.entry_price) / position.entry_price * 100
            else:
                pnl_pct = (position.entry_price - position.current_price) / position.entry_price * 100
        else:
            pnl_pct = 0.0

        return {
            "symbol": symbol,
            "entry_price": position.entry_price,
            "quantity": position.quantity,
            "is_long": position.is_long,
            "current_price": position.current_price,
            "stop_loss_price": position.stop_loss_price,
            "take_profit_price": position.take_profit_price,
            "highest_price": position.highest_price,
            "lowest_price": position.lowest_price if position.lowest_price != float("inf") else None,
            "pnl_pct": pnl_pct,
            "trailing_enabled": position.trailing_stop_enabled,
            "entry_time": position.entry_time.isoformat(),
        }

    def get_all_positions(self) -> list[dict[str, Any]]:
        """Get status of all tracked positions."""
        return [
            self.get_position_status(symbol)
            for symbol in self._positions
            if self.get_position_status(symbol)
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            **self._stats,
            "exit_rule_stats": self._exit_manager.get_stats(),
            "config": {
                "use_atr_stops": self._config.use_atr_stops,
                "atr_multiplier": self._config.atr_multiplier,
                "fixed_stop_loss_pct": self._config.fixed_stop_loss_pct,
                "trailing_enabled": self._config.trailing_stop_enabled,
                "take_profit_pct": self._config.take_profit_pct,
            }
        }

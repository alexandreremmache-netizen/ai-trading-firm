"""
Exit Rules - TakeProfit and StopLoss
====================================

Dynamic exit rules with trailing support for position management.
Inspired by investing-algorithm-framework patterns.

Features:
- Fixed and trailing TakeProfit rules
- Fixed and trailing StopLoss rules
- Position-level and portfolio-level rules
- Time-based exits
- Rule chaining and priorities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ExitRuleType(Enum):
    """Type of exit rule."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIME_EXIT = "time_exit"
    TRAILING_TAKE_PROFIT = "trailing_take_profit"
    TRAILING_STOP_LOSS = "trailing_stop_loss"


class ExitTriggerReason(Enum):
    """Reason an exit was triggered."""
    TAKE_PROFIT_HIT = "take_profit_hit"
    STOP_LOSS_HIT = "stop_loss_hit"
    TRAILING_STOP_HIT = "trailing_stop_hit"
    TRAILING_PROFIT_HIT = "trailing_profit_hit"
    TIME_EXPIRED = "time_expired"
    MANUAL = "manual"


@dataclass
class TakeProfitRule:
    """
    Take profit rule for a position.

    Attributes:
        symbol: Instrument symbol
        percentage_threshold: Profit percentage to trigger (e.g., 10 = 10%)
        trailing: If True, profit tracks highest price and triggers on pullback
        trailing_distance_pct: For trailing, percentage below peak to trigger
        sell_percentage: Percentage of position to sell when triggered (0-100)
        priority: Higher priority rules are checked first
    """
    symbol: str
    percentage_threshold: float
    trailing: bool = False
    trailing_distance_pct: float = 2.0  # Trail by 2% below peak
    sell_percentage: float = 100.0  # Sell entire position
    priority: int = 1
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "type": "take_profit",
            "percentage_threshold": self.percentage_threshold,
            "trailing": self.trailing,
            "trailing_distance_pct": self.trailing_distance_pct,
            "sell_percentage": self.sell_percentage,
            "priority": self.priority,
            "enabled": self.enabled,
        }


@dataclass
class StopLossRule:
    """
    Stop loss rule for a position.

    Attributes:
        symbol: Instrument symbol
        percentage_threshold: Loss percentage to trigger (e.g., 5 = -5%)
        trailing: If True, stop tracks highest price (locks in gains)
        trailing_distance_pct: For trailing, percentage below peak to trigger
        sell_percentage: Percentage of position to sell when triggered (0-100)
        priority: Higher priority rules are checked first
    """
    symbol: str
    percentage_threshold: float
    trailing: bool = False
    trailing_distance_pct: float = 5.0  # Trail by 5% below peak
    sell_percentage: float = 100.0  # Sell entire position
    priority: int = 2  # Stop loss usually higher priority than take profit
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "type": "stop_loss",
            "percentage_threshold": self.percentage_threshold,
            "trailing": self.trailing,
            "trailing_distance_pct": self.trailing_distance_pct,
            "sell_percentage": self.sell_percentage,
            "priority": self.priority,
            "enabled": self.enabled,
        }


@dataclass
class TimeExitRule:
    """
    Time-based exit rule.

    Attributes:
        symbol: Instrument symbol
        max_holding_hours: Maximum hours to hold position
        sell_percentage: Percentage to sell when time expires
    """
    symbol: str
    max_holding_hours: float
    sell_percentage: float = 100.0
    priority: int = 0  # Lowest priority
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "type": "time_exit",
            "max_holding_hours": self.max_holding_hours,
            "sell_percentage": self.sell_percentage,
            "priority": self.priority,
            "enabled": self.enabled,
        }


@dataclass
class PositionState:
    """
    Tracks the state of a position for exit rule evaluation.

    Used internally by ExitRuleManager.
    """
    symbol: str
    entry_price: float
    quantity: int
    entry_time: datetime
    is_long: bool = True
    highest_price: float = 0.0  # For trailing rules
    lowest_price: float = float("inf")  # For short trailing
    current_price: float = 0.0
    pnl_pct: float = 0.0

    def update_price(self, price: float) -> None:
        """Update current price and track extremes."""
        self.current_price = price

        if self.is_long:
            self.highest_price = max(self.highest_price, price)
            self.pnl_pct = ((price - self.entry_price) / self.entry_price) * 100
        else:
            self.lowest_price = min(self.lowest_price, price)
            self.pnl_pct = ((self.entry_price - price) / self.entry_price) * 100

    @property
    def drawdown_from_peak_pct(self) -> float:
        """Calculate drawdown from highest price (for longs)."""
        if self.highest_price <= 0:
            return 0.0
        return ((self.highest_price - self.current_price) / self.highest_price) * 100

    @property
    def rally_from_low_pct(self) -> float:
        """Calculate rally from lowest price (for shorts)."""
        if self.lowest_price == float("inf") or self.lowest_price <= 0:
            return 0.0
        return ((self.current_price - self.lowest_price) / self.lowest_price) * 100


@dataclass
class ExitSignal:
    """
    Signal generated when an exit rule is triggered.

    Contains all information needed to execute the exit.
    """
    symbol: str
    rule_type: ExitRuleType
    trigger_reason: ExitTriggerReason
    sell_quantity: int
    sell_percentage: float
    trigger_price: float
    position_pnl_pct: float
    rule_details: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "rule_type": self.rule_type.value,
            "trigger_reason": self.trigger_reason.value,
            "sell_quantity": self.sell_quantity,
            "sell_percentage": self.sell_percentage,
            "trigger_price": self.trigger_price,
            "position_pnl_pct": self.position_pnl_pct,
            "timestamp": self.timestamp.isoformat(),
            "rule_details": self.rule_details,
        }


class ExitRuleManager:
    """
    Manages exit rules for positions.

    Tracks position states and evaluates rules on each price update.
    Generates ExitSignals when rules are triggered.

    Usage:
        manager = ExitRuleManager()

        # Add rules
        manager.add_take_profit(TakeProfitRule("AAPL", 10, trailing=True))
        manager.add_stop_loss(StopLossRule("AAPL", 5, trailing=False))

        # Register position
        manager.register_position("AAPL", entry_price=150.0, quantity=100)

        # On each price update
        exits = manager.evaluate("AAPL", current_price=165.0)
        for exit_signal in exits:
            # Execute the exit
            pass
    """

    def __init__(self):
        self._take_profits: dict[str, list[TakeProfitRule]] = {}
        self._stop_losses: dict[str, list[StopLossRule]] = {}
        self._time_exits: dict[str, list[TimeExitRule]] = {}
        self._positions: dict[str, PositionState] = {}

        # Statistics
        self._stats = {
            "total_triggers": 0,
            "take_profit_triggers": 0,
            "stop_loss_triggers": 0,
            "trailing_triggers": 0,
            "time_exit_triggers": 0,
        }

    def add_take_profit(self, rule: TakeProfitRule) -> None:
        """Add a take profit rule."""
        if rule.symbol not in self._take_profits:
            self._take_profits[rule.symbol] = []
        self._take_profits[rule.symbol].append(rule)
        # Sort by priority (higher first)
        self._take_profits[rule.symbol].sort(key=lambda r: -r.priority)
        logger.debug(
            f"Added TakeProfit rule for {rule.symbol}: "
            f"{rule.percentage_threshold}% {'(trailing)' if rule.trailing else ''}"
        )

    def add_stop_loss(self, rule: StopLossRule) -> None:
        """Add a stop loss rule."""
        if rule.symbol not in self._stop_losses:
            self._stop_losses[rule.symbol] = []
        self._stop_losses[rule.symbol].append(rule)
        # Sort by priority (higher first)
        self._stop_losses[rule.symbol].sort(key=lambda r: -r.priority)
        logger.debug(
            f"Added StopLoss rule for {rule.symbol}: "
            f"{rule.percentage_threshold}% {'(trailing)' if rule.trailing else ''}"
        )

    def add_time_exit(self, rule: TimeExitRule) -> None:
        """Add a time-based exit rule."""
        if rule.symbol not in self._time_exits:
            self._time_exits[rule.symbol] = []
        self._time_exits[rule.symbol].append(rule)
        logger.debug(f"Added TimeExit rule for {rule.symbol}: {rule.max_holding_hours}h")

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        is_long: bool = True,
        entry_time: datetime | None = None,
    ) -> None:
        """
        Register a new position for exit rule tracking.

        Args:
            symbol: Instrument symbol
            entry_price: Entry price
            quantity: Position size
            is_long: True for long, False for short
            entry_time: When position was opened (default: now)
        """
        self._positions[symbol] = PositionState(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=entry_time or datetime.now(timezone.utc),
            is_long=is_long,
            highest_price=entry_price,
            lowest_price=entry_price,
            current_price=entry_price,
        )
        logger.info(
            f"Registered position for {symbol}: "
            f"{'LONG' if is_long else 'SHORT'} {quantity} @ ${entry_price:.2f}"
        )

    def update_position(
        self,
        symbol: str,
        quantity: int | None = None,
        entry_price: float | None = None,
    ) -> None:
        """
        Update an existing position.

        Used when position size changes (partial fills, partial exits).
        """
        if symbol not in self._positions:
            return

        pos = self._positions[symbol]
        if quantity is not None:
            pos.quantity = quantity
        if entry_price is not None:
            pos.entry_price = entry_price

    def close_position(self, symbol: str) -> None:
        """Remove a position from tracking."""
        if symbol in self._positions:
            del self._positions[symbol]
            logger.info(f"Closed position tracking for {symbol}")

    def evaluate(self, symbol: str, current_price: float) -> list[ExitSignal]:
        """
        Evaluate all exit rules for a symbol at the current price.

        Args:
            symbol: Instrument symbol
            current_price: Current market price

        Returns:
            List of ExitSignals for triggered rules
        """
        if symbol not in self._positions:
            return []

        pos = self._positions[symbol]
        pos.update_price(current_price)

        signals = []

        # Check stop losses first (highest priority for risk management)
        signals.extend(self._evaluate_stop_losses(symbol, pos))

        # Check take profits
        signals.extend(self._evaluate_take_profits(symbol, pos))

        # Check time exits
        signals.extend(self._evaluate_time_exits(symbol, pos))

        return signals

    def _evaluate_stop_losses(
        self,
        symbol: str,
        pos: PositionState,
    ) -> list[ExitSignal]:
        """Evaluate stop loss rules."""
        signals = []
        rules = self._stop_losses.get(symbol, [])

        for rule in rules:
            if not rule.enabled:
                continue

            triggered = False
            trigger_reason = ExitTriggerReason.STOP_LOSS_HIT

            if rule.trailing:
                # Trailing stop: trigger if price drops from peak
                if pos.is_long:
                    if pos.drawdown_from_peak_pct >= rule.trailing_distance_pct:
                        triggered = True
                        trigger_reason = ExitTriggerReason.TRAILING_STOP_HIT
                else:  # Short position
                    if pos.rally_from_low_pct >= rule.trailing_distance_pct:
                        triggered = True
                        trigger_reason = ExitTriggerReason.TRAILING_STOP_HIT
            else:
                # Fixed stop: trigger if loss exceeds threshold
                if pos.pnl_pct <= -rule.percentage_threshold:
                    triggered = True

            if triggered:
                sell_qty = int(pos.quantity * rule.sell_percentage / 100)
                if sell_qty > 0:
                    signals.append(ExitSignal(
                        symbol=symbol,
                        rule_type=ExitRuleType.TRAILING_STOP_LOSS if rule.trailing else ExitRuleType.STOP_LOSS,
                        trigger_reason=trigger_reason,
                        sell_quantity=sell_qty,
                        sell_percentage=rule.sell_percentage,
                        trigger_price=pos.current_price,
                        position_pnl_pct=pos.pnl_pct,
                        rule_details=rule.to_dict(),
                    ))
                    self._stats["total_triggers"] += 1
                    self._stats["stop_loss_triggers"] += 1
                    if rule.trailing:
                        self._stats["trailing_triggers"] += 1
                    logger.info(
                        f"STOP LOSS triggered for {symbol}: "
                        f"PnL {pos.pnl_pct:.2f}%, selling {sell_qty} shares"
                    )

        return signals

    def _evaluate_take_profits(
        self,
        symbol: str,
        pos: PositionState,
    ) -> list[ExitSignal]:
        """Evaluate take profit rules."""
        signals = []
        rules = self._take_profits.get(symbol, [])

        for rule in rules:
            if not rule.enabled:
                continue

            triggered = False
            trigger_reason = ExitTriggerReason.TAKE_PROFIT_HIT

            if rule.trailing:
                # Trailing take profit: first reach threshold, then trail
                if pos.pnl_pct >= rule.percentage_threshold:
                    # Threshold reached, now check for pullback trigger
                    if pos.is_long and pos.drawdown_from_peak_pct >= rule.trailing_distance_pct:
                        triggered = True
                        trigger_reason = ExitTriggerReason.TRAILING_PROFIT_HIT
                    elif not pos.is_long and pos.rally_from_low_pct >= rule.trailing_distance_pct:
                        triggered = True
                        trigger_reason = ExitTriggerReason.TRAILING_PROFIT_HIT
            else:
                # Fixed take profit: trigger at threshold
                if pos.pnl_pct >= rule.percentage_threshold:
                    triggered = True

            if triggered:
                sell_qty = int(pos.quantity * rule.sell_percentage / 100)
                if sell_qty > 0:
                    signals.append(ExitSignal(
                        symbol=symbol,
                        rule_type=ExitRuleType.TRAILING_TAKE_PROFIT if rule.trailing else ExitRuleType.TAKE_PROFIT,
                        trigger_reason=trigger_reason,
                        sell_quantity=sell_qty,
                        sell_percentage=rule.sell_percentage,
                        trigger_price=pos.current_price,
                        position_pnl_pct=pos.pnl_pct,
                        rule_details=rule.to_dict(),
                    ))
                    self._stats["total_triggers"] += 1
                    self._stats["take_profit_triggers"] += 1
                    if rule.trailing:
                        self._stats["trailing_triggers"] += 1
                    logger.info(
                        f"TAKE PROFIT triggered for {symbol}: "
                        f"PnL {pos.pnl_pct:.2f}%, selling {sell_qty} shares"
                    )

        return signals

    def _evaluate_time_exits(
        self,
        symbol: str,
        pos: PositionState,
    ) -> list[ExitSignal]:
        """Evaluate time-based exit rules."""
        signals = []
        rules = self._time_exits.get(symbol, [])
        now = datetime.now(timezone.utc)

        for rule in rules:
            if not rule.enabled:
                continue

            holding_hours = (now - pos.entry_time).total_seconds() / 3600

            if holding_hours >= rule.max_holding_hours:
                sell_qty = int(pos.quantity * rule.sell_percentage / 100)
                if sell_qty > 0:
                    signals.append(ExitSignal(
                        symbol=symbol,
                        rule_type=ExitRuleType.TIME_EXIT,
                        trigger_reason=ExitTriggerReason.TIME_EXPIRED,
                        sell_quantity=sell_qty,
                        sell_percentage=rule.sell_percentage,
                        trigger_price=pos.current_price,
                        position_pnl_pct=pos.pnl_pct,
                        rule_details=rule.to_dict(),
                    ))
                    self._stats["total_triggers"] += 1
                    self._stats["time_exit_triggers"] += 1
                    logger.info(
                        f"TIME EXIT triggered for {symbol}: "
                        f"held {holding_hours:.1f}h, selling {sell_qty} shares"
                    )

        return signals

    def get_position_status(self, symbol: str) -> dict[str, Any] | None:
        """Get status of a tracked position."""
        if symbol not in self._positions:
            return None

        pos = self._positions[symbol]
        return {
            "symbol": symbol,
            "entry_price": pos.entry_price,
            "quantity": pos.quantity,
            "entry_time": pos.entry_time.isoformat(),
            "is_long": pos.is_long,
            "current_price": pos.current_price,
            "highest_price": pos.highest_price,
            "lowest_price": pos.lowest_price if pos.lowest_price != float("inf") else None,
            "pnl_pct": pos.pnl_pct,
            "drawdown_from_peak_pct": pos.drawdown_from_peak_pct,
            "holding_hours": (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600,
        }

    def get_rules_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Get all rules for a symbol."""
        return {
            "take_profits": [r.to_dict() for r in self._take_profits.get(symbol, [])],
            "stop_losses": [r.to_dict() for r in self._stop_losses.get(symbol, [])],
            "time_exits": [r.to_dict() for r in self._time_exits.get(symbol, [])],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            "positions_tracked": len(self._positions),
            "symbols_with_rules": len(
                set(self._take_profits.keys()) |
                set(self._stop_losses.keys()) |
                set(self._time_exits.keys())
            ),
            **self._stats,
        }

    def clear_rules(self, symbol: str | None = None) -> None:
        """Clear rules for a symbol or all symbols."""
        if symbol:
            self._take_profits.pop(symbol, None)
            self._stop_losses.pop(symbol, None)
            self._time_exits.pop(symbol, None)
        else:
            self._take_profits.clear()
            self._stop_losses.clear()
            self._time_exits.clear()

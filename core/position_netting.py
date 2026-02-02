"""
Position Netting
================

Aggregates and nets positions across multiple strategies to provide
a consolidated view of portfolio exposure.

Required for:
- Accurate risk calculation
- Margin optimization
- Cross-strategy position limits
- Portfolio-level exposure reporting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from core.contract_specs import ContractSpecsManager, get_currency_converter


logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class StrategyPosition:
    """Position held by a specific strategy."""
    strategy: str
    symbol: str
    quantity: int  # Positive for long, negative for short
    avg_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def side(self) -> PositionSide:
        if self.quantity > 0:
            return PositionSide.LONG
        elif self.quantity < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT


@dataclass
class NetPosition:
    """
    Net position aggregated across all strategies.

    Provides consolidated view of exposure regardless of which
    strategy originated the position.
    """
    symbol: str
    net_quantity: int  # Sum of all strategy quantities
    avg_entry_price: float  # Weighted average entry price
    market_value: float  # Current market value
    unrealized_pnl: float  # Total unrealized P&L
    contributing_strategies: list[str] = field(default_factory=list)
    strategy_breakdown: dict[str, int] = field(default_factory=dict)  # strategy -> quantity
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def side(self) -> PositionSide:
        if self.net_quantity > 0:
            return PositionSide.LONG
        elif self.net_quantity < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT

    @property
    def gross_quantity(self) -> int:
        """Total absolute quantity across strategies (ignores netting)."""
        return sum(abs(qty) for qty in self.strategy_breakdown.values())

    @property
    def netting_benefit(self) -> int:
        """Quantity reduced through netting (gross - net)."""
        return self.gross_quantity - abs(self.net_quantity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "net_quantity": self.net_quantity,
            "side": self.side.value,
            "avg_entry_price": self.avg_entry_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "gross_quantity": self.gross_quantity,
            "netting_benefit": self.netting_benefit,
            "contributing_strategies": self.contributing_strategies,
            "strategy_breakdown": self.strategy_breakdown,
        }


class PositionNetter:
    """
    Aggregates positions across strategies and provides net exposure.

    Features:
    - Net position calculation
    - Gross vs net exposure
    - Strategy attribution
    - P&L attribution
    - Position limit checking on net basis
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize position netter.

        Args:
            config: Configuration with:
                - use_fifo: Use FIFO for P&L (default: True)
                - track_history: Track position history (default: True)
        """
        self._config = config or {}
        self._use_fifo = self._config.get("use_fifo", True)
        self._track_history = self._config.get("track_history", True)

        # Strategy positions: {symbol: {strategy: StrategyPosition}}
        self._strategy_positions: dict[str, dict[str, StrategyPosition]] = {}

        # Market prices for P&L calculation
        self._market_prices: dict[str, float] = {}

        # Contract specs for notional value
        self._contract_specs = ContractSpecsManager()

        # Position history
        self._position_history: list[dict[str, Any]] = []

        logger.info("PositionNetter initialized")

    def update_position(
        self,
        strategy: str,
        symbol: str,
        quantity: int,
        avg_price: float
    ) -> None:
        """
        Update position for a strategy.

        Args:
            strategy: Strategy name
            symbol: Instrument symbol
            quantity: Position quantity (positive=long, negative=short, 0=flat)
            avg_price: Average entry price
        """
        if symbol not in self._strategy_positions:
            self._strategy_positions[symbol] = {}

        if quantity == 0:
            # Remove position
            if strategy in self._strategy_positions[symbol]:
                del self._strategy_positions[symbol][strategy]
                if not self._strategy_positions[symbol]:
                    del self._strategy_positions[symbol]
        else:
            # Update/create position
            market_price = self._market_prices.get(symbol, avg_price)
            spec = self._contract_specs.get_spec(symbol)
            multiplier = spec.multiplier if spec else 1.0

            market_value = abs(quantity) * market_price * multiplier
            unrealized_pnl = (market_price - avg_price) * quantity * multiplier

            position = StrategyPosition(
                strategy=strategy,
                symbol=symbol,
                quantity=quantity,
                avg_price=avg_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
            )

            self._strategy_positions[symbol][strategy] = position

        # Track history
        if self._track_history:
            self._position_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "strategy": strategy,
                "symbol": symbol,
                "quantity": quantity,
                "avg_price": avg_price,
            })

    def update_market_price(self, symbol: str, price: float) -> None:
        """Update market price for P&L calculation."""
        if price > 0:
            self._market_prices[symbol] = price
            # Recalculate P&L for positions in this symbol
            self._recalculate_pnl(symbol)

    def _recalculate_pnl(self, symbol: str) -> None:
        """Recalculate unrealized P&L for a symbol."""
        if symbol not in self._strategy_positions:
            return

        market_price = self._market_prices.get(symbol)
        if market_price is None:
            return

        spec = self._contract_specs.get_spec(symbol)
        multiplier = spec.multiplier if spec else 1.0

        for position in self._strategy_positions[symbol].values():
            position.market_value = abs(position.quantity) * market_price * multiplier
            position.unrealized_pnl = (market_price - position.avg_price) * position.quantity * multiplier

    def get_net_position(self, symbol: str) -> NetPosition | None:
        """
        Get net position for a symbol across all strategies.

        Args:
            symbol: Instrument symbol

        Returns:
            NetPosition or None if no positions
        """
        if symbol not in self._strategy_positions:
            return None

        positions = self._strategy_positions[symbol]
        if not positions:
            return None

        # Calculate net quantity
        net_quantity = sum(p.quantity for p in positions.values())

        # Calculate weighted average entry price
        total_value = 0.0
        total_abs_qty = 0
        for p in positions.values():
            if p.quantity != 0:
                total_value += p.avg_price * abs(p.quantity)
                total_abs_qty += abs(p.quantity)

        avg_entry = total_value / total_abs_qty if total_abs_qty > 0 else 0.0

        # Sum market value and P&L
        market_value = sum(p.market_value for p in positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in positions.values())

        # Build strategy breakdown
        strategy_breakdown = {p.strategy: p.quantity for p in positions.values()}
        contributing = [s for s, q in strategy_breakdown.items() if q != 0]

        return NetPosition(
            symbol=symbol,
            net_quantity=net_quantity,
            avg_entry_price=avg_entry,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            contributing_strategies=contributing,
            strategy_breakdown=strategy_breakdown,
        )

    def get_all_net_positions(self) -> dict[str, NetPosition]:
        """Get net positions for all symbols."""
        result = {}
        for symbol in self._strategy_positions:
            net = self.get_net_position(symbol)
            if net and net.net_quantity != 0:
                result[symbol] = net
        return result

    def get_strategy_positions(self, strategy: str) -> dict[str, StrategyPosition]:
        """Get all positions for a specific strategy."""
        result = {}
        for symbol, positions in self._strategy_positions.items():
            if strategy in positions:
                result[symbol] = positions[strategy]
        return result

    def get_gross_exposure(self) -> float:
        """Get total gross exposure (sum of absolute position values)."""
        total = 0.0
        for positions in self._strategy_positions.values():
            for p in positions.values():
                total += p.market_value
        return total

    def get_net_exposure(self) -> float:
        """Get total net exposure (net long - net short)."""
        total = 0.0
        for symbol in self._strategy_positions:
            net = self.get_net_position(symbol)
            if net:
                # Net value: positive for long, negative for short
                spec = self._contract_specs.get_spec(symbol)
                multiplier = spec.multiplier if spec else 1.0
                market_price = self._market_prices.get(symbol, 0)
                total += net.net_quantity * market_price * multiplier
        return total

    def get_long_exposure(self) -> float:
        """Get total long exposure."""
        total = 0.0
        for symbol in self._strategy_positions:
            net = self.get_net_position(symbol)
            if net and net.net_quantity > 0:
                total += net.market_value
        return total

    def get_short_exposure(self) -> float:
        """Get total short exposure (as positive number)."""
        total = 0.0
        for symbol in self._strategy_positions:
            net = self.get_net_position(symbol)
            if net and net.net_quantity < 0:
                total += abs(net.market_value)
        return total

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        total = 0.0
        for positions in self._strategy_positions.values():
            for p in positions.values():
                total += p.unrealized_pnl
        return total

    def check_position_limit(
        self,
        symbol: str,
        new_quantity: int,
        strategy: str,
        max_net_position: int
    ) -> tuple[bool, str]:
        """
        Check if adding a position would exceed limits on net basis.

        Args:
            symbol: Instrument symbol
            new_quantity: Proposed new quantity for this strategy
            strategy: Strategy name
            max_net_position: Maximum allowed net position

        Returns:
            Tuple of (allowed, reason)
        """
        # Calculate what net position would be after this trade
        current_net = self.get_net_position(symbol)
        current_qty = 0
        if current_net:
            # Get current quantity for this strategy
            current_strategy_qty = current_net.strategy_breakdown.get(strategy, 0)
            # Calculate change
            qty_change = new_quantity - current_strategy_qty
            # New net = old net + change
            proposed_net = current_net.net_quantity + qty_change
        else:
            proposed_net = new_quantity

        if abs(proposed_net) > max_net_position:
            return False, f"Would exceed net position limit: {abs(proposed_net)} > {max_net_position}"

        return True, "OK"

    def get_netting_summary(self) -> dict[str, Any]:
        """Get summary of netting benefits."""
        total_gross = 0
        total_net = 0
        netting_details = []

        for symbol in self._strategy_positions:
            net = self.get_net_position(symbol)
            if net:
                total_gross += net.gross_quantity
                total_net += abs(net.net_quantity)
                if net.netting_benefit > 0:
                    netting_details.append({
                        "symbol": symbol,
                        "gross": net.gross_quantity,
                        "net": abs(net.net_quantity),
                        "benefit": net.netting_benefit,
                    })

        return {
            "total_gross_quantity": total_gross,
            "total_net_quantity": total_net,
            "total_netting_benefit": total_gross - total_net,
            "netting_ratio": total_net / total_gross if total_gross > 0 else 1.0,
            "symbols_with_netting": netting_details,
        }

    def get_status(self) -> dict[str, Any]:
        """Get netter status for monitoring."""
        all_net = self.get_all_net_positions()

        return {
            "symbols_tracked": len(self._strategy_positions),
            "net_positions": len(all_net),
            "gross_exposure": self.get_gross_exposure(),
            "net_exposure": self.get_net_exposure(),
            "long_exposure": self.get_long_exposure(),
            "short_exposure": self.get_short_exposure(),
            "total_unrealized_pnl": self.get_total_unrealized_pnl(),
            "netting_summary": self.get_netting_summary(),
        }

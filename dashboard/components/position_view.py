"""
Position View
=============

Portfolio position tracking and display for the trading system dashboard.

Features:
- Real-time position tracking with market value updates
- Portfolio summary with P&L calculations
- Position grouping by asset type, sector, and strategy
- Position history for chart visualization
- Concentration and weight analysis
- WebSocket-ready export to dict
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, TYPE_CHECKING

from core.events import FillEvent, OrderSide

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class AssetType(Enum):
    """Asset type classification."""
    EQUITY = "equity"
    ETF = "etf"
    FUTURES = "futures"
    FOREX = "forex"
    OPTIONS = "options"
    UNKNOWN = "unknown"


# Default sector mapping for common instruments
# Can be overridden via configuration
DEFAULT_SECTOR_MAP: dict[str, str] = {
    # Equities - Technology
    "AAPL": "technology",
    "MSFT": "technology",
    "GOOGL": "technology",
    "META": "technology",
    "NVDA": "technology",
    # Equities - Consumer Discretionary
    "AMZN": "consumer_discretionary",
    "TSLA": "consumer_discretionary",
    # Financials
    "JPM": "financials",
    "V": "financials",
    # Healthcare
    "JNJ": "healthcare",
    # ETFs - Index
    "SPY": "index",
    "QQQ": "index",
    "IWM": "index",
    "DIA": "index",
    # ETFs - Commodities
    "GLD": "commodities",
    "SLV": "commodities",
    # ETFs - Fixed Income
    "TLT": "fixed_income",
    # ETFs - Sector
    "XLF": "financials",
    "XLE": "energy",
    "VXX": "volatility",
    # Futures - Index
    "ES": "index_futures",
    "NQ": "index_futures",
    "YM": "index_futures",
    "RTY": "index_futures",
    "MES": "index_futures",
    "MNQ": "index_futures",
    "MYM": "index_futures",
    "M2K": "index_futures",
    # Futures - Energy
    "CL": "energy",
    "MCL": "energy",
    "NG": "energy",
    # Futures - Metals
    "GC": "precious_metals",
    "MGC": "precious_metals",
    "SI": "precious_metals",
    "SIL": "precious_metals",
    # Forex
    "EUR": "forex_major",
    "GBP": "forex_major",
    "JPY": "forex_major",
    "CHF": "forex_major",
    "AUD": "forex_major",
    "CAD": "forex_major",
    "EURUSD": "forex_major",
    "GBPUSD": "forex_major",
    "USDJPY": "forex_major",
    "USDCHF": "forex_major",
    "AUDUSD": "forex_major",
    "USDCAD": "forex_major",
}

# Asset type detection by symbol prefix/suffix patterns
FUTURES_SYMBOLS = {
    "ES", "NQ", "YM", "RTY", "CL", "GC", "SI", "NG",
    "MES", "MNQ", "MYM", "M2K", "MCL", "MGC", "SIL",
    "ZB", "ZN", "ZF", "ZT", "ZC", "ZW", "ZS", "ZM", "ZL",
}

ETF_SYMBOLS = {
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT",
    "XLF", "XLE", "XLK", "XLV", "XLP", "XLI", "XLB",
    "XLU", "XLRE", "XLC", "VXX", "UVXY", "TQQQ", "SQQQ",
}

FOREX_SYMBOLS = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
    "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
}


@dataclass
class PositionRecord:
    """
    Record of a single portfolio position.

    Tracks position details including cost basis, market value, and P&L.
    """
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    weight_pct: float = 0.0
    # Classification
    asset_type: AssetType = AssetType.UNKNOWN
    sector: str = "unknown"
    strategy: str = "unknown"
    # Metadata
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    first_trade_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Calculate derived fields after initialization."""
        self._recalculate()

    def _recalculate(self) -> None:
        """Recalculate derived values."""
        self.market_value = self.quantity * self.current_price
        cost_basis = self.quantity * self.avg_cost
        self.unrealized_pnl = self.market_value - cost_basis
        if cost_basis != 0:
            self.unrealized_pnl_pct = (self.unrealized_pnl / abs(cost_basis)) * 100
        else:
            self.unrealized_pnl_pct = 0.0

    def update_price(self, price: float) -> None:
        """Update current price and recalculate P&L."""
        self.current_price = price
        self.last_update = datetime.now(timezone.utc)
        self._recalculate()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": round(self.avg_cost, 4),
            "current_price": round(self.current_price, 4),
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 2),
            "weight_pct": round(self.weight_pct, 2),
            "asset_type": self.asset_type.value,
            "sector": self.sector,
            "strategy": self.strategy,
            "last_update": self.last_update.isoformat(),
            "first_trade_time": self.first_trade_time.isoformat(),
        }


@dataclass
class PortfolioSummary:
    """
    Summary of portfolio state.

    Provides aggregate metrics for the entire portfolio.
    """
    total_value: float = 0.0
    cash: float = 0.0
    invested: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    positions_count: int = 0
    # Additional metrics
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    # Concentration metrics
    largest_position_pct: float = 0.0
    top_5_concentration_pct: float = 0.0
    herfindahl_index: float = 0.0
    # Timestamps
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "total_value": round(self.total_value, 2),
            "cash": round(self.cash, 2),
            "invested": round(self.invested, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct, 2),
            "positions_count": self.positions_count,
            "long_exposure": round(self.long_exposure, 2),
            "short_exposure": round(self.short_exposure, 2),
            "net_exposure": round(self.net_exposure, 2),
            "gross_exposure": round(self.gross_exposure, 2),
            "largest_position_pct": round(self.largest_position_pct, 2),
            "top_5_concentration_pct": round(self.top_5_concentration_pct, 2),
            "herfindahl_index": round(self.herfindahl_index, 4),
            "last_update": self.last_update.isoformat(),
        }


@dataclass
class PositionHistoryPoint:
    """Single point in position history for charting."""
    timestamp: datetime
    symbol: str
    quantity: int
    market_value: float
    unrealized_pnl: float
    weight_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for charting."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "quantity": self.quantity,
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "weight_pct": round(self.weight_pct, 2),
        }


class PositionView:
    """
    Portfolio position tracking and display.

    Manages position data, calculates portfolio metrics, and provides
    data for dashboard visualization.

    Usage:
        view = PositionView(initial_cash=1_000_000)

        # Process fills from trading
        view.process_fill(fill_event)

        # Update prices from market data
        view.update_price("AAPL", 175.50)

        # Get all positions
        positions = view.get_all_positions()

        # Get portfolio summary
        summary = view.get_portfolio_summary()

        # Get positions grouped by sector
        by_sector = view.get_positions_by_sector()

        # Export for WebSocket streaming
        data = view.to_dict()
    """

    # Maximum history points per symbol
    MAX_HISTORY_POINTS = 1000

    # Snapshot interval for history (seconds)
    HISTORY_SNAPSHOT_INTERVAL = 60

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        sector_map: dict[str, str] | None = None,
        strategy_map: dict[str, str] | None = None,
    ):
        """
        Initialize PositionView.

        Args:
            initial_cash: Starting cash balance
            sector_map: Custom symbol-to-sector mapping (merged with defaults)
            strategy_map: Symbol-to-strategy mapping
        """
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: dict[str, PositionRecord] = {}

        # Sector mapping (default + custom)
        self._sector_map = dict(DEFAULT_SECTOR_MAP)
        if sector_map:
            self._sector_map.update(sector_map)

        # Strategy mapping
        self._strategy_map = strategy_map or {}

        # Position history for charts (symbol -> deque of history points)
        self._position_history: dict[str, deque[PositionHistoryPoint]] = {}
        self._last_snapshot_time: datetime | None = None

        # Daily P&L tracking
        self._day_start_value: float | None = None
        self._day_start_date: datetime | None = None

        # Total realized P&L
        self._realized_pnl = 0.0

        # Fill tracking for audit
        self._total_fills = 0
        self._total_volume = 0
        self._total_commissions = 0.0

        logger.info(f"PositionView initialized with cash={initial_cash}")

    def process_fill(self, fill: FillEvent) -> None:
        """
        Process a fill event to update positions.

        Args:
            fill: Fill event from execution
        """
        symbol = fill.symbol
        quantity = fill.filled_quantity
        price = fill.fill_price
        commission = fill.commission

        # Adjust quantity for side
        if fill.side == OrderSide.SELL:
            quantity = -quantity

        self._total_fills += 1
        self._total_volume += abs(fill.filled_quantity)
        self._total_commissions += commission

        # Update cash for commission
        self._cash -= commission

        if symbol in self._positions:
            self._update_existing_position(symbol, quantity, price)
        else:
            self._create_new_position(symbol, quantity, price)

        logger.debug(
            f"Processed fill: {symbol} {fill.side.value} {fill.filled_quantity} @ {price}"
        )

    def _create_new_position(self, symbol: str, quantity: int, price: float) -> None:
        """Create a new position."""
        if quantity == 0:
            return

        # Update cash
        self._cash -= quantity * price

        # Classify the position
        asset_type = self._detect_asset_type(symbol)
        sector = self._sector_map.get(symbol, "unknown")
        strategy = self._strategy_map.get(symbol, "unknown")

        position = PositionRecord(
            symbol=symbol,
            quantity=quantity,
            avg_cost=price,
            current_price=price,
            asset_type=asset_type,
            sector=sector,
            strategy=strategy,
        )

        self._positions[symbol] = position
        self._recalculate_weights()

        # Initialize history for this symbol
        if symbol not in self._position_history:
            self._position_history[symbol] = deque(maxlen=self.MAX_HISTORY_POINTS)

    def _update_existing_position(
        self, symbol: str, quantity: int, price: float
    ) -> None:
        """Update an existing position with a new fill."""
        position = self._positions[symbol]
        old_quantity = position.quantity
        new_quantity = old_quantity + quantity

        # Update cash
        self._cash -= quantity * price

        if new_quantity == 0:
            # Position closed - calculate realized P&L
            realized = (price - position.avg_cost) * abs(old_quantity)
            if old_quantity < 0:
                realized = -realized
            self._realized_pnl += realized
            del self._positions[symbol]
            logger.info(f"Position closed: {symbol}, realized P&L: {realized:.2f}")
        elif (old_quantity > 0 and quantity > 0) or (old_quantity < 0 and quantity < 0):
            # Adding to position - update average cost
            total_cost = (old_quantity * position.avg_cost) + (quantity * price)
            position.avg_cost = total_cost / new_quantity
            position.quantity = new_quantity
            position.current_price = price
            position._recalculate()
        elif abs(new_quantity) < abs(old_quantity):
            # Reducing position - keep avg cost, record realized P&L
            closed_quantity = abs(quantity)
            realized = (price - position.avg_cost) * closed_quantity
            if old_quantity < 0:
                realized = -realized
            self._realized_pnl += realized
            position.quantity = new_quantity
            position.current_price = price
            position._recalculate()
        else:
            # Reversing position (going from long to short or vice versa)
            # Close old position first
            closed_quantity = abs(old_quantity)
            realized = (price - position.avg_cost) * closed_quantity
            if old_quantity < 0:
                realized = -realized
            self._realized_pnl += realized
            # New position
            position.quantity = new_quantity
            position.avg_cost = price
            position.current_price = price
            position._recalculate()

        self._recalculate_weights()

    def update_position(
        self,
        symbol: str,
        quantity: int,
        avg_cost: float,
        current_price: float | None = None,
    ) -> None:
        """
        Directly update or create a position (for initialization or sync).

        Args:
            symbol: Instrument symbol
            quantity: Position quantity (negative for short)
            avg_cost: Average cost basis
            current_price: Current market price (defaults to avg_cost)
        """
        if current_price is None:
            current_price = avg_cost

        if quantity == 0:
            # Remove position if it exists
            if symbol in self._positions:
                del self._positions[symbol]
                self._recalculate_weights()
            return

        asset_type = self._detect_asset_type(symbol)
        sector = self._sector_map.get(symbol, "unknown")
        strategy = self._strategy_map.get(symbol, "unknown")

        if symbol in self._positions:
            position = self._positions[symbol]
            position.quantity = quantity
            position.avg_cost = avg_cost
            position.current_price = current_price
            position.asset_type = asset_type
            position.sector = sector
            position.strategy = strategy
            position._recalculate()
        else:
            position = PositionRecord(
                symbol=symbol,
                quantity=quantity,
                avg_cost=avg_cost,
                current_price=current_price,
                asset_type=asset_type,
                sector=sector,
                strategy=strategy,
            )
            self._positions[symbol] = position

            # Initialize history
            if symbol not in self._position_history:
                self._position_history[symbol] = deque(maxlen=self.MAX_HISTORY_POINTS)

        self._recalculate_weights()
        logger.debug(f"Updated position: {symbol} qty={quantity} avg={avg_cost}")

    def update_price(self, symbol: str, price: float) -> None:
        """
        Update the current price for a position.

        Args:
            symbol: Instrument symbol
            price: Current market price
        """
        if symbol not in self._positions:
            return

        self._positions[symbol].update_price(price)
        self._recalculate_weights()
        self._maybe_snapshot_history()

    def update_prices(self, prices: dict[str, float]) -> None:
        """
        Batch update prices for multiple positions.

        Args:
            prices: Dictionary of symbol -> price
        """
        updated = False
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol].update_price(price)
                updated = True

        if updated:
            self._recalculate_weights()
            self._maybe_snapshot_history()

    def set_cash(self, cash: float) -> None:
        """
        Set the cash balance directly.

        Args:
            cash: New cash balance
        """
        self._cash = cash

    def get_all_positions(self) -> list[PositionRecord]:
        """
        Get all current positions.

        Returns:
            List of PositionRecord objects
        """
        return list(self._positions.values())

    def get_position(self, symbol: str) -> PositionRecord | None:
        """
        Get a specific position.

        Args:
            symbol: Instrument symbol

        Returns:
            PositionRecord or None if not found
        """
        return self._positions.get(symbol)

    def get_portfolio_summary(self) -> PortfolioSummary:
        """
        Calculate and return portfolio summary.

        Returns:
            PortfolioSummary with current metrics
        """
        now = datetime.now(timezone.utc)

        # Calculate total invested (sum of market values)
        invested = sum(p.market_value for p in self._positions.values())
        total_value = self._cash + invested

        # Calculate total unrealized P&L
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        total_pnl = total_unrealized + self._realized_pnl
        total_pnl_pct = (total_pnl / self._initial_cash * 100) if self._initial_cash > 0 else 0.0

        # Daily P&L tracking
        self._update_day_tracking(total_value, now)
        daily_pnl = total_value - (self._day_start_value or self._initial_cash)
        day_start = self._day_start_value or self._initial_cash
        daily_pnl_pct = (daily_pnl / day_start * 100) if day_start > 0 else 0.0

        # Exposure calculations
        long_exposure = sum(
            p.market_value for p in self._positions.values()
            if p.quantity > 0
        )
        short_exposure = abs(sum(
            p.market_value for p in self._positions.values()
            if p.quantity < 0
        ))
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure

        # Concentration metrics
        largest_position_pct = 0.0
        top_5_concentration_pct = 0.0
        herfindahl_index = 0.0

        if self._positions and total_value > 0:
            weights = sorted(
                [abs(p.market_value) / total_value * 100 for p in self._positions.values()],
                reverse=True
            )
            largest_position_pct = weights[0] if weights else 0.0
            top_5_concentration_pct = sum(weights[:5])
            # Herfindahl index (sum of squared weights)
            herfindahl_index = sum((w / 100) ** 2 for w in weights)

        return PortfolioSummary(
            total_value=total_value,
            cash=self._cash,
            invested=invested,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            positions_count=len(self._positions),
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            largest_position_pct=largest_position_pct,
            top_5_concentration_pct=top_5_concentration_pct,
            herfindahl_index=herfindahl_index,
            last_update=now,
        )

    def get_positions_by_sector(self) -> dict[str, list[PositionRecord]]:
        """
        Get positions grouped by sector.

        Returns:
            Dictionary of sector -> list of positions
        """
        by_sector: dict[str, list[PositionRecord]] = {}
        for position in self._positions.values():
            sector = position.sector
            if sector not in by_sector:
                by_sector[sector] = []
            by_sector[sector].append(position)
        return by_sector

    def get_positions_by_asset_type(self) -> dict[str, list[PositionRecord]]:
        """
        Get positions grouped by asset type.

        Returns:
            Dictionary of asset_type -> list of positions
        """
        by_type: dict[str, list[PositionRecord]] = {}
        for position in self._positions.values():
            asset_type = position.asset_type.value
            if asset_type not in by_type:
                by_type[asset_type] = []
            by_type[asset_type].append(position)
        return by_type

    def get_positions_by_strategy(self) -> dict[str, list[PositionRecord]]:
        """
        Get positions grouped by strategy.

        Returns:
            Dictionary of strategy -> list of positions
        """
        by_strategy: dict[str, list[PositionRecord]] = {}
        for position in self._positions.values():
            strategy = position.strategy
            if strategy not in by_strategy:
                by_strategy[strategy] = []
            by_strategy[strategy].append(position)
        return by_strategy

    def get_sector_weights(self) -> dict[str, float]:
        """
        Calculate portfolio weights by sector.

        Returns:
            Dictionary of sector -> weight percentage
        """
        total_value = self._cash + sum(p.market_value for p in self._positions.values())
        if total_value <= 0:
            return {}

        weights: dict[str, float] = {}
        for position in self._positions.values():
            sector = position.sector
            weight = abs(position.market_value) / total_value * 100
            weights[sector] = weights.get(sector, 0.0) + weight
        return weights

    def get_asset_type_weights(self) -> dict[str, float]:
        """
        Calculate portfolio weights by asset type.

        Returns:
            Dictionary of asset_type -> weight percentage
        """
        total_value = self._cash + sum(p.market_value for p in self._positions.values())
        if total_value <= 0:
            return {}

        weights: dict[str, float] = {}
        for position in self._positions.values():
            asset_type = position.asset_type.value
            weight = abs(position.market_value) / total_value * 100
            weights[asset_type] = weights.get(asset_type, 0.0) + weight
        return weights

    def get_concentration_metrics(self) -> dict[str, Any]:
        """
        Calculate portfolio concentration metrics.

        Returns:
            Dictionary with concentration analysis
        """
        summary = self.get_portfolio_summary()
        sector_weights = self.get_sector_weights()
        asset_weights = self.get_asset_type_weights()

        # Find most concentrated sector
        max_sector = max(sector_weights.items(), key=lambda x: x[1]) if sector_weights else ("none", 0.0)
        max_asset = max(asset_weights.items(), key=lambda x: x[1]) if asset_weights else ("none", 0.0)

        return {
            "positions_count": summary.positions_count,
            "largest_position_pct": summary.largest_position_pct,
            "top_5_concentration_pct": summary.top_5_concentration_pct,
            "herfindahl_index": summary.herfindahl_index,
            "most_concentrated_sector": max_sector[0],
            "most_concentrated_sector_pct": round(max_sector[1], 2),
            "most_concentrated_asset_type": max_asset[0],
            "most_concentrated_asset_type_pct": round(max_asset[1], 2),
            "sector_weights": {k: round(v, 2) for k, v in sector_weights.items()},
            "asset_type_weights": {k: round(v, 2) for k, v in asset_weights.items()},
        }

    def get_position_history(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[PositionHistoryPoint]:
        """
        Get position history for charting.

        Args:
            symbol: Specific symbol (all if None)
            limit: Maximum points to return

        Returns:
            List of history points
        """
        if symbol:
            history = list(self._position_history.get(symbol, []))
            history.reverse()
            return history[:limit]

        # Aggregate all history
        all_history: list[PositionHistoryPoint] = []
        for history_deque in self._position_history.values():
            all_history.extend(history_deque)

        # Sort by timestamp descending
        all_history.sort(key=lambda x: x.timestamp, reverse=True)
        return all_history[:limit]

    def get_portfolio_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get portfolio value history for charting.

        Returns aggregated portfolio value over time.

        Args:
            limit: Maximum points to return

        Returns:
            List of portfolio history points
        """
        # Collect all unique timestamps
        timestamps: set[datetime] = set()
        for history_deque in self._position_history.values():
            for point in history_deque:
                timestamps.add(point.timestamp)

        if not timestamps:
            return []

        # Build portfolio value at each timestamp
        history: list[dict[str, Any]] = []
        for ts in sorted(timestamps, reverse=True)[:limit]:
            total_value = self._cash
            total_pnl = 0.0
            for symbol, history_deque in self._position_history.items():
                for point in history_deque:
                    if point.timestamp == ts:
                        total_value += point.market_value
                        total_pnl += point.unrealized_pnl
                        break

            history.append({
                "timestamp": ts.isoformat(),
                "total_value": round(total_value, 2),
                "unrealized_pnl": round(total_pnl, 2),
            })

        return history

    def to_dict(self) -> dict[str, Any]:
        """
        Export complete view state to dictionary for WebSocket streaming.

        Returns:
            Complete state as dict
        """
        summary = self.get_portfolio_summary()
        positions = self.get_all_positions()

        return {
            "summary": summary.to_dict(),
            "positions": [p.to_dict() for p in positions],
            "by_sector": {
                sector: [p.to_dict() for p in positions_list]
                for sector, positions_list in self.get_positions_by_sector().items()
            },
            "by_asset_type": {
                asset_type: [p.to_dict() for p in positions_list]
                for asset_type, positions_list in self.get_positions_by_asset_type().items()
            },
            "by_strategy": {
                strategy: [p.to_dict() for p in positions_list]
                for strategy, positions_list in self.get_positions_by_strategy().items()
            },
            "concentration": self.get_concentration_metrics(),
            "sector_weights": {k: round(v, 2) for k, v in self.get_sector_weights().items()},
            "asset_type_weights": {k: round(v, 2) for k, v in self.get_asset_type_weights().items()},
            "statistics": {
                "total_fills": self._total_fills,
                "total_volume": self._total_volume,
                "total_commissions": round(self._total_commissions, 2),
                "realized_pnl": round(self._realized_pnl, 2),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _detect_asset_type(self, symbol: str) -> AssetType:
        """Detect asset type based on symbol."""
        # Check exact matches first
        if symbol in FUTURES_SYMBOLS:
            return AssetType.FUTURES
        if symbol in ETF_SYMBOLS:
            return AssetType.ETF
        if symbol in FOREX_SYMBOLS:
            return AssetType.FOREX

        # Check patterns
        symbol_upper = symbol.upper()

        # Forex patterns (pairs like EURUSD, EUR/USD)
        if "/" in symbol or (len(symbol) == 6 and symbol_upper[:3] in {"EUR", "GBP", "USD", "JPY", "CHF", "AUD", "CAD", "NZD"}):
            return AssetType.FOREX

        # Options patterns (usually have dates/strikes)
        if any(c in symbol for c in ["C", "P"]) and any(c.isdigit() for c in symbol):
            # Could be option - simplified check
            pass

        # Default to equity for unknown symbols
        return AssetType.EQUITY

    def _recalculate_weights(self) -> None:
        """Recalculate all position weights."""
        total_value = self._cash + sum(p.market_value for p in self._positions.values())

        if total_value <= 0:
            for position in self._positions.values():
                position.weight_pct = 0.0
            return

        for position in self._positions.values():
            position.weight_pct = (position.market_value / total_value) * 100

    def _update_day_tracking(self, current_value: float, now: datetime) -> None:
        """Update daily P&L tracking, reset at start of new day."""
        today = now.date()

        if self._day_start_date is None or self._day_start_date != today:
            self._day_start_value = current_value
            self._day_start_date = today

    def _maybe_snapshot_history(self) -> None:
        """Take a history snapshot if enough time has passed."""
        now = datetime.now(timezone.utc)

        if self._last_snapshot_time is not None:
            elapsed = (now - self._last_snapshot_time).total_seconds()
            if elapsed < self.HISTORY_SNAPSHOT_INTERVAL:
                return

        self._last_snapshot_time = now

        for symbol, position in self._positions.items():
            if symbol not in self._position_history:
                self._position_history[symbol] = deque(maxlen=self.MAX_HISTORY_POINTS)

            point = PositionHistoryPoint(
                timestamp=now,
                symbol=symbol,
                quantity=position.quantity,
                market_value=position.market_value,
                unrealized_pnl=position.unrealized_pnl,
                weight_pct=position.weight_pct,
            )
            self._position_history[symbol].append(point)

    def clear(self) -> None:
        """Clear all positions and reset to initial state."""
        self._positions.clear()
        self._position_history.clear()
        self._cash = self._initial_cash
        self._realized_pnl = 0.0
        self._total_fills = 0
        self._total_volume = 0
        self._total_commissions = 0.0
        self._day_start_value = None
        self._day_start_date = None
        self._last_snapshot_time = None
        logger.info("PositionView cleared and reset")

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def realized_pnl(self) -> float:
        """Total realized P&L."""
        return self._realized_pnl

    @property
    def positions_count(self) -> int:
        """Number of open positions."""
        return len(self._positions)

    @property
    def symbols(self) -> list[str]:
        """List of symbols with open positions."""
        return list(self._positions.keys())

"""
Market Data View
================

Real-time market data display component for the trading system dashboard.

Supports all instrument types from the trading system:
- Equities: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, V, JNJ
- ETFs: SPY, QQQ, IWM, DIA, GLD, SLV, TLT, XLF, XLE, VXX
- E-mini Futures: ES, NQ, YM, RTY, CL, GC, SI
- Micro Futures: MES, MNQ, MYM, M2K, MCL, MGC, SIL
- Forex: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD

Features:
- Real-time quote updates with bid/ask/last/volume
- OHLC bar aggregation for charts (1m, 5m, 15m, 1h, 1d)
- Price alerts when thresholds are crossed
- Market overview with indices, forex, futures, and VIX
- Thread-safe with asyncio locks
- WebSocket-ready export for real-time streaming
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.events import MarketDataEvent


logger = logging.getLogger(__name__)


class InstrumentType(Enum):
    """Classification of instrument types."""
    EQUITY = "equity"
    ETF = "etf"
    FUTURES = "futures"
    MICRO_FUTURES = "micro_futures"
    FOREX = "forex"
    INDEX = "index"
    UNKNOWN = "unknown"


class AlertType(Enum):
    """Types of price alerts."""
    ABOVE = "above"           # Price crosses above threshold
    BELOW = "below"           # Price crosses below threshold
    PERCENT_CHANGE = "pct"    # Percent change exceeds threshold
    SPREAD_WIDE = "spread"    # Bid-ask spread exceeds threshold


class TimeFrame(Enum):
    """OHLC aggregation timeframes."""
    M1 = "1m"       # 1 minute
    M5 = "5m"       # 5 minutes
    M15 = "15m"     # 15 minutes
    H1 = "1h"       # 1 hour
    D1 = "1d"       # 1 day


# Timeframe to seconds mapping
TIMEFRAME_SECONDS: dict[TimeFrame, int] = {
    TimeFrame.M1: 60,
    TimeFrame.M5: 300,
    TimeFrame.M15: 900,
    TimeFrame.H1: 3600,
    TimeFrame.D1: 86400,
}


# Default instrument universe from config
DEFAULT_EQUITIES = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]
DEFAULT_ETFS = ["SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "XLF", "XLE", "VXX"]
DEFAULT_FUTURES = ["ES", "NQ", "YM", "RTY", "CL", "GC", "SI"]
DEFAULT_MICRO_FUTURES = ["MES", "MNQ", "MYM", "M2K", "MCL", "MGC", "SIL"]
DEFAULT_FOREX = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"]

# Instrument type detection
INSTRUMENT_TYPE_MAP: dict[str, InstrumentType] = {
    # Equities
    **{s: InstrumentType.EQUITY for s in DEFAULT_EQUITIES},
    # ETFs
    **{s: InstrumentType.ETF for s in DEFAULT_ETFS},
    # Futures
    **{s: InstrumentType.FUTURES for s in DEFAULT_FUTURES},
    # Micro Futures
    **{s: InstrumentType.MICRO_FUTURES for s in DEFAULT_MICRO_FUTURES},
    # Forex
    **{s: InstrumentType.FOREX for s in DEFAULT_FOREX},
    # Indices
    "VIX": InstrumentType.INDEX,
}


@dataclass
class QuoteData:
    """
    Real-time quote data for a single instrument.

    Contains current market prices and statistics for display.
    """
    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    change: float = 0.0           # Absolute change from open/previous close
    change_pct: float = 0.0       # Percentage change
    high: float = 0.0             # Session high
    low: float = 0.0              # Session low
    open: float = 0.0             # Session open
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Additional metadata
    instrument_type: InstrumentType = InstrumentType.UNKNOWN
    exchange: str = ""
    bid_size: int = 0
    ask_size: int = 0
    prev_close: float = 0.0

    @property
    def mid(self) -> float:
        """Calculate mid price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid = self.mid
        if mid > 0:
            return (self.spread / mid) * 100
        return 0.0

    @property
    def is_stale(self) -> bool:
        """Check if quote is older than 60 seconds."""
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > 60.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "symbol": self.symbol,
            "bid": round(self.bid, 4),
            "ask": round(self.ask, 4),
            "last": round(self.last, 4),
            "mid": round(self.mid, 4),
            "spread": round(self.spread, 4),
            "spread_pct": round(self.spread_pct, 4),
            "volume": self.volume,
            "change": round(self.change, 4),
            "change_pct": round(self.change_pct, 2),
            "high": round(self.high, 4),
            "low": round(self.low, 4),
            "open": round(self.open, 4),
            "prev_close": round(self.prev_close, 4),
            "timestamp": self.timestamp.isoformat(),
            "instrument_type": self.instrument_type.value,
            "exchange": self.exchange,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "is_stale": self.is_stale,
        }


@dataclass
class OHLCBar:
    """
    OHLC (Open-High-Low-Close) bar for charting.

    Represents price action over a specific time period.
    """
    symbol: str
    timestamp: datetime               # Bar start time
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    timeframe: TimeFrame = TimeFrame.M1
    tick_count: int = 0              # Number of ticks in this bar
    is_complete: bool = False        # True if bar period has ended

    def update(self, price: float, volume: int = 0) -> None:
        """
        Update bar with a new tick.

        Args:
            price: New price tick
            volume: Volume for this tick
        """
        if self.tick_count == 0:
            # First tick sets OHLC
            self.open = price
            self.high = price
            self.low = price
            self.close = price
        else:
            # Update high/low/close
            self.high = max(self.high, price)
            self.low = min(self.low, price)
            self.close = price

        self.volume += volume
        self.tick_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for charting libraries."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "time": int(self.timestamp.timestamp()),  # Unix timestamp for charts
            "open": round(self.open, 4),
            "high": round(self.high, 4),
            "low": round(self.low, 4),
            "close": round(self.close, 4),
            "volume": self.volume,
            "timeframe": self.timeframe.value,
            "tick_count": self.tick_count,
            "is_complete": self.is_complete,
        }


@dataclass
class MarketOverview:
    """
    Market overview data for dashboard summary.

    Provides snapshot of major indices, forex, futures, and VIX.
    """
    indices: dict[str, QuoteData] = field(default_factory=dict)      # SPY, QQQ, DIA
    forex: dict[str, QuoteData] = field(default_factory=dict)        # EURUSD, GBPUSD
    futures: dict[str, QuoteData] = field(default_factory=dict)      # ES, NQ, CL, GC
    vix: QuoteData | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "indices": {k: v.to_dict() for k, v in self.indices.items()},
            "forex": {k: v.to_dict() for k, v in self.forex.items()},
            "futures": {k: v.to_dict() for k, v in self.futures.items()},
            "vix": self.vix.to_dict() if self.vix else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PriceAlert:
    """
    Price alert configuration and state.

    Triggers when price crosses defined thresholds.
    """
    alert_id: str
    symbol: str
    alert_type: AlertType
    threshold: float
    enabled: bool = True
    triggered: bool = False
    triggered_at: datetime | None = None
    triggered_price: float | None = None
    # For percent change alerts
    reference_price: float | None = None

    def check(self, quote: QuoteData) -> bool:
        """
        Check if alert should trigger based on current quote.

        Args:
            quote: Current quote data

        Returns:
            True if alert triggered, False otherwise
        """
        if not self.enabled or self.triggered:
            return False

        price = quote.last

        if self.alert_type == AlertType.ABOVE:
            if price >= self.threshold:
                self._trigger(price)
                return True

        elif self.alert_type == AlertType.BELOW:
            if price <= self.threshold:
                self._trigger(price)
                return True

        elif self.alert_type == AlertType.PERCENT_CHANGE:
            ref = self.reference_price or quote.open or quote.prev_close
            if ref > 0:
                change_pct = abs((price - ref) / ref) * 100
                if change_pct >= self.threshold:
                    self._trigger(price)
                    return True

        elif self.alert_type == AlertType.SPREAD_WIDE:
            if quote.spread_pct >= self.threshold:
                self._trigger(price)
                return True

        return False

    def _trigger(self, price: float) -> None:
        """Mark alert as triggered."""
        self.triggered = True
        self.triggered_at = datetime.now(timezone.utc)
        self.triggered_price = price

    def reset(self) -> None:
        """Reset alert to check again."""
        self.triggered = False
        self.triggered_at = None
        self.triggered_price = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "alert_type": self.alert_type.value,
            "threshold": self.threshold,
            "enabled": self.enabled,
            "triggered": self.triggered,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "triggered_price": self.triggered_price,
            "reference_price": self.reference_price,
        }


@dataclass
class AlertNotification:
    """Notification for a triggered alert."""
    alert: PriceAlert
    quote: QuoteData
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert": self.alert.to_dict(),
            "quote": self.quote.to_dict(),
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


class MarketDataView:
    """
    Real-time market data display component.

    Manages quote updates, OHLC aggregation, price alerts, and market overview
    for the trading system dashboard.

    Usage:
        view = MarketDataView()

        # Update with market data events
        await view.update_quote(market_data_event)

        # Get current quote
        quote = view.get_quote("AAPL")

        # Get batch of quotes
        quotes = view.get_quotes_batch(["AAPL", "MSFT", "GOOGL"])

        # Get OHLC history for charts
        bars = view.get_ohlc_history("AAPL", TimeFrame.M5, limit=100)

        # Get market overview
        overview = view.get_market_overview()

        # Set price alerts
        view.add_alert("AAPL", AlertType.ABOVE, 200.0)

        # Export for WebSocket streaming
        data = view.to_dict()

    Thread Safety:
        All public methods acquire an asyncio lock before modifying state,
        making the view safe for concurrent use.
    """

    # Default limits
    DEFAULT_OHLC_HISTORY_SIZE = 500    # Max bars per timeframe per symbol
    DEFAULT_QUOTE_HISTORY_SIZE = 100   # Max quote snapshots per symbol

    # Alert notification callback type
    AlertCallback = Callable[[AlertNotification], None]

    def __init__(
        self,
        ohlc_history_size: int = DEFAULT_OHLC_HISTORY_SIZE,
        quote_history_size: int = DEFAULT_QUOTE_HISTORY_SIZE,
    ):
        """
        Initialize the market data view.

        Args:
            ohlc_history_size: Maximum OHLC bars to retain per timeframe
            quote_history_size: Maximum quote snapshots to retain
        """
        self._ohlc_history_size = ohlc_history_size
        self._quote_history_size = quote_history_size

        # Current quotes by symbol
        self._quotes: dict[str, QuoteData] = {}

        # Quote history for tracking
        self._quote_history: dict[str, deque[QuoteData]] = {}

        # OHLC bars by symbol and timeframe
        # Structure: {symbol: {timeframe: deque[OHLCBar]}}
        self._ohlc_bars: dict[str, dict[TimeFrame, deque[OHLCBar]]] = {}

        # Current (in-progress) bars by symbol and timeframe
        self._current_bars: dict[str, dict[TimeFrame, OHLCBar]] = {}

        # Price alerts by symbol
        self._alerts: dict[str, list[PriceAlert]] = {}

        # Alert notification callbacks
        self._alert_callbacks: list[MarketDataView.AlertCallback] = []

        # Triggered alerts history
        self._triggered_alerts: deque[AlertNotification] = deque(maxlen=100)

        # Update statistics
        self._update_count = 0
        self._last_update_time: datetime | None = None

        # Session tracking for daily open/close
        self._session_opens: dict[str, float] = {}
        self._session_date: datetime | None = None

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info(
            f"MarketDataView initialized: ohlc_history={ohlc_history_size}, "
            f"quote_history={quote_history_size}"
        )

    async def update_quote(self, event: MarketDataEvent) -> QuoteData:
        """
        Update quote data from a market data event.

        This is the main entry point for processing market data.
        Updates the quote, OHLC bars, and checks price alerts.

        Args:
            event: MarketDataEvent from the market data manager

        Returns:
            Updated QuoteData
        """
        async with self._lock:
            symbol = event.symbol
            now = event.timestamp

            # Detect instrument type
            instrument_type = INSTRUMENT_TYPE_MAP.get(symbol, InstrumentType.UNKNOWN)

            # Get or create quote
            if symbol in self._quotes:
                prev_quote = self._quotes[symbol]
                prev_close = prev_quote.prev_close or prev_quote.last
            else:
                prev_close = 0.0

            # Check for new session (reset open)
            current_date = now.date()
            if self._session_date != current_date:
                self._session_date = current_date
                self._session_opens.clear()

            # Track session open
            if symbol not in self._session_opens and event.last > 0:
                self._session_opens[symbol] = event.open_price if event.open_price > 0 else event.last

            session_open = self._session_opens.get(symbol, event.open_price or event.last)

            # Calculate change
            reference_price = session_open if session_open > 0 else (prev_close if prev_close > 0 else event.last)
            change = event.last - reference_price if reference_price > 0 else 0.0
            change_pct = (change / reference_price * 100) if reference_price > 0 else 0.0

            # Create updated quote
            quote = QuoteData(
                symbol=symbol,
                bid=event.bid,
                ask=event.ask,
                last=event.last,
                volume=event.volume,
                change=change,
                change_pct=change_pct,
                high=max(event.high, self._quotes.get(symbol, QuoteData(symbol)).high) if event.high > 0 else self._quotes.get(symbol, QuoteData(symbol)).high,
                low=min(event.low, self._quotes.get(symbol, QuoteData(symbol)).low) if event.low > 0 and self._quotes.get(symbol, QuoteData(symbol)).low > 0 else (event.low if event.low > 0 else self._quotes.get(symbol, QuoteData(symbol)).low),
                open=session_open,
                timestamp=now,
                instrument_type=instrument_type,
                exchange=event.exchange,
                bid_size=event.bid_size,
                ask_size=event.ask_size,
                prev_close=prev_close,
            )

            # Store quote
            self._quotes[symbol] = quote

            # Add to history
            if symbol not in self._quote_history:
                self._quote_history[symbol] = deque(maxlen=self._quote_history_size)
            self._quote_history[symbol].append(quote)

            # Update OHLC bars
            self._update_ohlc_bars(symbol, event.last, event.volume, now)

            # Check price alerts
            self._check_alerts(symbol, quote)

            # Update statistics
            self._update_count += 1
            self._last_update_time = now

            return quote

    def _update_ohlc_bars(
        self,
        symbol: str,
        price: float,
        volume: int,
        timestamp: datetime,
    ) -> None:
        """
        Update OHLC bars for all timeframes.

        Args:
            symbol: Instrument symbol
            price: Current price
            volume: Current volume
            timestamp: Current timestamp
        """
        if price <= 0:
            return

        # Initialize symbol structures if needed
        if symbol not in self._ohlc_bars:
            self._ohlc_bars[symbol] = {
                tf: deque(maxlen=self._ohlc_history_size)
                for tf in TimeFrame
            }

        if symbol not in self._current_bars:
            self._current_bars[symbol] = {}

        # Update each timeframe
        for timeframe in TimeFrame:
            tf_seconds = TIMEFRAME_SECONDS[timeframe]

            # Calculate bar start time (aligned to timeframe)
            bar_start = datetime.fromtimestamp(
                (timestamp.timestamp() // tf_seconds) * tf_seconds,
                tz=timezone.utc
            )

            # Get or create current bar
            current_bar = self._current_bars[symbol].get(timeframe)

            if current_bar is None or current_bar.timestamp != bar_start:
                # Complete previous bar if exists
                if current_bar is not None:
                    current_bar.is_complete = True
                    self._ohlc_bars[symbol][timeframe].append(current_bar)

                # Create new bar
                current_bar = OHLCBar(
                    symbol=symbol,
                    timestamp=bar_start,
                    timeframe=timeframe,
                )
                self._current_bars[symbol][timeframe] = current_bar

            # Update current bar with tick
            current_bar.update(price, volume)

    def _check_alerts(self, symbol: str, quote: QuoteData) -> None:
        """
        Check price alerts for a symbol.

        Args:
            symbol: Instrument symbol
            quote: Current quote data
        """
        if symbol not in self._alerts:
            return

        for alert in self._alerts[symbol]:
            if alert.check(quote):
                # Create notification
                message = self._format_alert_message(alert, quote)
                notification = AlertNotification(
                    alert=alert,
                    quote=quote,
                    message=message,
                )

                # Store in history
                self._triggered_alerts.append(notification)

                # Call callbacks
                for callback in self._alert_callbacks:
                    try:
                        callback(notification)
                    except Exception as e:
                        logger.exception(f"Alert callback error: {e}")

                logger.info(f"Price alert triggered: {message}")

    def _format_alert_message(self, alert: PriceAlert, quote: QuoteData) -> str:
        """Format alert notification message."""
        if alert.alert_type == AlertType.ABOVE:
            return f"{alert.symbol} crossed above {alert.threshold:.2f} at {quote.last:.2f}"
        elif alert.alert_type == AlertType.BELOW:
            return f"{alert.symbol} crossed below {alert.threshold:.2f} at {quote.last:.2f}"
        elif alert.alert_type == AlertType.PERCENT_CHANGE:
            return f"{alert.symbol} changed {quote.change_pct:.2f}% (threshold: {alert.threshold:.2f}%)"
        elif alert.alert_type == AlertType.SPREAD_WIDE:
            return f"{alert.symbol} spread widened to {quote.spread_pct:.2f}% (threshold: {alert.threshold:.2f}%)"
        return f"Alert triggered for {alert.symbol}"

    def get_quote(self, symbol: str) -> QuoteData | None:
        """
        Get current quote for a symbol.

        Args:
            symbol: Instrument symbol

        Returns:
            QuoteData or None if not available
        """
        return self._quotes.get(symbol)

    def get_quotes_batch(self, symbols: list[str]) -> dict[str, QuoteData]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of instrument symbols

        Returns:
            Dictionary mapping symbols to their quotes
        """
        return {
            symbol: quote
            for symbol in symbols
            if (quote := self._quotes.get(symbol)) is not None
        }

    def get_all_quotes(self) -> dict[str, QuoteData]:
        """
        Get all current quotes.

        Returns:
            Dictionary mapping all symbols to their quotes
        """
        return dict(self._quotes)

    def get_ohlc_history(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.M5,
        limit: int | None = None,
    ) -> list[OHLCBar]:
        """
        Get OHLC bar history for a symbol.

        Args:
            symbol: Instrument symbol
            timeframe: Bar timeframe
            limit: Maximum number of bars (None for all)

        Returns:
            List of OHLCBar objects (oldest first)
        """
        if symbol not in self._ohlc_bars:
            return []

        bars = list(self._ohlc_bars[symbol].get(timeframe, []))

        # Include current bar
        if symbol in self._current_bars and timeframe in self._current_bars[symbol]:
            current = self._current_bars[symbol][timeframe]
            if current.tick_count > 0:
                bars.append(current)

        if limit is not None:
            bars = bars[-limit:]

        return bars

    def get_ohlc_history_dict(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.M5,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get OHLC history as list of dictionaries for charting.

        Args:
            symbol: Instrument symbol
            timeframe: Bar timeframe
            limit: Maximum number of bars

        Returns:
            List of bar dictionaries
        """
        bars = self.get_ohlc_history(symbol, timeframe, limit)
        return [bar.to_dict() for bar in bars]

    def get_market_overview(self) -> MarketOverview:
        """
        Get market overview with major indices, forex, futures, and VIX.

        Returns:
            MarketOverview with current data
        """
        overview = MarketOverview(
            timestamp=datetime.now(timezone.utc),
        )

        # Indices (major ETFs as proxies)
        for symbol in ["SPY", "QQQ", "DIA", "IWM"]:
            if symbol in self._quotes:
                overview.indices[symbol] = self._quotes[symbol]

        # Forex
        for symbol in DEFAULT_FOREX:
            if symbol in self._quotes:
                overview.forex[symbol] = self._quotes[symbol]

        # Futures
        for symbol in ["ES", "NQ", "CL", "GC"]:
            if symbol in self._quotes:
                overview.futures[symbol] = self._quotes[symbol]

        # VIX
        if "VIX" in self._quotes:
            overview.vix = self._quotes["VIX"]
        elif "VXX" in self._quotes:
            # Use VXX as VIX proxy if VIX not available
            overview.vix = self._quotes["VXX"]

        return overview

    def add_alert(
        self,
        symbol: str,
        alert_type: AlertType,
        threshold: float,
        reference_price: float | None = None,
        alert_id: str | None = None,
    ) -> PriceAlert:
        """
        Add a price alert for a symbol.

        Args:
            symbol: Instrument symbol
            alert_type: Type of alert
            threshold: Alert threshold value
            reference_price: Reference price for percent change alerts
            alert_id: Optional custom alert ID

        Returns:
            Created PriceAlert
        """
        if alert_id is None:
            alert_id = f"{symbol}_{alert_type.value}_{threshold}_{datetime.now(timezone.utc).timestamp()}"

        alert = PriceAlert(
            alert_id=alert_id,
            symbol=symbol,
            alert_type=alert_type,
            threshold=threshold,
            reference_price=reference_price,
        )

        if symbol not in self._alerts:
            self._alerts[symbol] = []

        self._alerts[symbol].append(alert)

        logger.info(f"Added price alert: {symbol} {alert_type.value} {threshold}")

        return alert

    def remove_alert(self, alert_id: str) -> bool:
        """
        Remove a price alert.

        Args:
            alert_id: Alert ID to remove

        Returns:
            True if removed, False if not found
        """
        for symbol, alerts in self._alerts.items():
            for alert in alerts:
                if alert.alert_id == alert_id:
                    alerts.remove(alert)
                    logger.info(f"Removed price alert: {alert_id}")
                    return True
        return False

    def get_alerts(self, symbol: str | None = None) -> list[PriceAlert]:
        """
        Get price alerts.

        Args:
            symbol: Filter by symbol (None for all)

        Returns:
            List of PriceAlert objects
        """
        if symbol is not None:
            return self._alerts.get(symbol, [])

        all_alerts = []
        for alerts in self._alerts.values():
            all_alerts.extend(alerts)
        return all_alerts

    def get_triggered_alerts(self, limit: int = 50) -> list[AlertNotification]:
        """
        Get recently triggered alerts.

        Args:
            limit: Maximum number of alerts

        Returns:
            List of AlertNotification objects (newest first)
        """
        alerts = list(self._triggered_alerts)
        alerts.reverse()
        return alerts[:limit]

    def add_alert_callback(self, callback: AlertCallback) -> None:
        """
        Register a callback for alert notifications.

        Args:
            callback: Function to call when alert triggers
        """
        self._alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: AlertCallback) -> bool:
        """
        Remove an alert callback.

        Args:
            callback: Callback function to remove

        Returns:
            True if removed, False if not found
        """
        try:
            self._alert_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def get_quote_history(
        self,
        symbol: str,
        limit: int | None = None,
    ) -> list[QuoteData]:
        """
        Get quote history for a symbol.

        Args:
            symbol: Instrument symbol
            limit: Maximum number of quotes

        Returns:
            List of QuoteData (oldest first)
        """
        if symbol not in self._quote_history:
            return []

        history = list(self._quote_history[symbol])
        if limit is not None:
            history = history[-limit:]
        return history

    def get_quotes_by_type(
        self,
        instrument_type: InstrumentType,
    ) -> dict[str, QuoteData]:
        """
        Get all quotes for a specific instrument type.

        Args:
            instrument_type: Type of instruments to get

        Returns:
            Dictionary mapping symbols to quotes
        """
        return {
            symbol: quote
            for symbol, quote in self._quotes.items()
            if quote.instrument_type == instrument_type
        }

    def get_price_table(
        self,
        symbols: list[str] | None = None,
        sort_by: str = "symbol",
        ascending: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get formatted price table for display.

        Args:
            symbols: Symbols to include (None for all)
            sort_by: Column to sort by (symbol, last, change_pct, volume)
            ascending: Sort direction

        Returns:
            List of quote dictionaries for table display
        """
        if symbols is None:
            quotes = list(self._quotes.values())
        else:
            quotes = [
                self._quotes[s] for s in symbols
                if s in self._quotes
            ]

        # Sort
        sort_key = {
            "symbol": lambda q: q.symbol,
            "last": lambda q: q.last,
            "change": lambda q: q.change,
            "change_pct": lambda q: q.change_pct,
            "volume": lambda q: q.volume,
            "spread": lambda q: q.spread_pct,
        }.get(sort_by, lambda q: q.symbol)

        quotes.sort(key=sort_key, reverse=not ascending)

        return [q.to_dict() for q in quotes]

    def get_chart_data(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.M5,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Get data formatted for charting libraries.

        Args:
            symbol: Instrument symbol
            timeframe: Bar timeframe
            limit: Number of bars

        Returns:
            Dictionary with OHLC arrays for charting
        """
        bars = self.get_ohlc_history(symbol, timeframe, limit)

        return {
            "symbol": symbol,
            "timeframe": timeframe.value,
            "timestamps": [bar.timestamp.isoformat() for bar in bars],
            "times": [int(bar.timestamp.timestamp()) for bar in bars],
            "open": [bar.open for bar in bars],
            "high": [bar.high for bar in bars],
            "low": [bar.low for bar in bars],
            "close": [bar.close for bar in bars],
            "volume": [bar.volume for bar in bars],
        }

    def clear_symbol(self, symbol: str) -> None:
        """
        Clear all data for a symbol.

        Args:
            symbol: Symbol to clear
        """
        self._quotes.pop(symbol, None)
        self._quote_history.pop(symbol, None)
        self._ohlc_bars.pop(symbol, None)
        self._current_bars.pop(symbol, None)
        self._alerts.pop(symbol, None)
        self._session_opens.pop(symbol, None)

    def clear_all(self) -> None:
        """Clear all market data."""
        self._quotes.clear()
        self._quote_history.clear()
        self._ohlc_bars.clear()
        self._current_bars.clear()
        self._alerts.clear()
        self._triggered_alerts.clear()
        self._session_opens.clear()
        self._update_count = 0
        self._last_update_time = None

    def to_dict(self) -> dict[str, Any]:
        """
        Export view state to dictionary for WebSocket streaming.

        Returns:
            Complete view state as dict
        """
        now = datetime.now(timezone.utc)

        # Get overview
        overview = self.get_market_overview()

        # Get quotes by type
        quotes_by_type = {
            it.value: {
                symbol: quote.to_dict()
                for symbol, quote in self._quotes.items()
                if quote.instrument_type == it
            }
            for it in InstrumentType
        }

        # Active alerts
        active_alerts = [
            alert.to_dict()
            for alerts in self._alerts.values()
            for alert in alerts
            if alert.enabled and not alert.triggered
        ]

        # Recent triggered alerts
        triggered_alerts = [
            notification.to_dict()
            for notification in list(self._triggered_alerts)[-10:]
        ]

        return {
            "quotes": {symbol: quote.to_dict() for symbol, quote in self._quotes.items()},
            "quotes_by_type": quotes_by_type,
            "overview": overview.to_dict(),
            "active_alerts": active_alerts,
            "triggered_alerts": triggered_alerts,
            "statistics": {
                "update_count": self._update_count,
                "last_update": self._last_update_time.isoformat() if self._last_update_time else None,
                "symbols_tracked": len(self._quotes),
                "alerts_active": len(active_alerts),
                "alerts_triggered": len(self._triggered_alerts),
            },
            "timestamp": now.isoformat(),
        }

    async def to_dict_async(self) -> dict[str, Any]:
        """
        Async version of to_dict with proper locking.

        Returns:
            Complete view state as dict
        """
        async with self._lock:
            return self.to_dict()

    @property
    def update_count(self) -> int:
        """Total number of quote updates processed."""
        return self._update_count

    @property
    def symbols_tracked(self) -> int:
        """Number of symbols being tracked."""
        return len(self._quotes)

    @property
    def last_update_time(self) -> datetime | None:
        """Timestamp of last quote update."""
        return self._last_update_time

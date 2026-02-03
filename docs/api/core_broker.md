# broker

**Path**: `C:\Users\Alexa\ai-trading-firm\core\broker.py`

## Overview

Interactive Brokers Integration
===============================

Exclusive broker interface for market data, portfolio state, and execution.
Paper trading is the default mode (port 7497).

IMPORTANT: Ensure TWS or IB Gateway is running before connecting.
- TWS Paper: port 7497
- TWS Live: port 7496
- Gateway Paper: port 4002
- Gateway Live: port 4001

## Classes

### ConnectionState

**Inherits from**: Enum

Broker connection state.

### BrokerConfig

Interactive Brokers connection configuration.

### MarketDataStaleness

Tracks staleness status for market data.

### Position

Current position in a symbol.

### PortfolioState

Current portfolio state.

### OrderStatus

Order status tracking.

### IBRateLimiter

IB API Rate Limiter (P0-1 fix).

Interactive Brokers enforces rate limits:
- Max 60 requests per 10 minutes (600 seconds) for market data
- No duplicate requests within 15 seconds
- Exceeding limits causes temporary bans

This class implements a sliding window rate limiter to prevent
hitting IB's rate limits and causing connection issues.

#### Methods

##### `def __init__(self)`

##### `def can_make_request(self, request_key: ) -> tuple[bool, str]`

Check if a request can be made within rate limits.

Args:
    request_key: Optional key for duplicate detection (e.g., "mktdata:AAPL")

Returns:
    Tuple of (can_request, reason_if_not)

##### `def record_request(self, request_key: ) -> None`

Record that a request was made.

##### `def get_remaining_requests(self) -> int`

Get number of requests remaining in current window.

##### `def get_wait_time(self, request_key: ) -> float`

Get seconds to wait before next request is allowed.

### IBBroker

Interactive Brokers integration using ib_insync.

Responsibilities:
- Connect to TWS/Gateway
- Stream real-time market data
- Query portfolio state
- Execute orders (paper trading by default)

This is the ONLY interface to the market.
All market access goes through this class.

#### Methods

##### `def __init__(self, config: BrokerConfig)`

##### `def is_connected(self) -> bool`

Check if connected to IB.

##### `def connection_state(self) -> ConnectionState`

Get current connection state.

##### `def account_id(self) -> str`

Get the connected account ID.

##### `def circuit_breaker(self) -> CircuitBreaker`

Get the circuit breaker for this broker (#S6).

##### `def rate_limiter(self) -> IBRateLimiter`

Get the IB API rate limiter (P0-1).

##### `async def connect(self) -> bool`

Connect to Interactive Brokers TWS or Gateway.

Returns True if connected successfully.

IMPORTANT: TWS or IB Gateway must be running with API enabled.
Configure in TWS: Edit > Global Configuration > API > Settings
- Enable ActiveX and Socket Clients
- Socket port: 7497 (paper) or 7496 (live)
- Allow connections from localhost

##### `async def disconnect(self) -> None`

Disconnect from Interactive Brokers.

##### `def get_reconciliation_status(self) -> dict`

Get the last order reconciliation status.

##### `def on_disconnect(self, callback: Callable[, None]) -> None`

Register callback for disconnection events.

##### `def on_reconnect(self, callback: Callable[, None]) -> None`

Register callback for reconnection events.

##### `def enable_auto_reconnect(self) -> None`

Enable automatic reconnection.

##### `def disable_auto_reconnect(self) -> None`

Disable automatic reconnection.

##### `def get_connection_stats(self) -> dict[str, Any]`

Get connection statistics.

##### `def on_market_data(self, callback: Callable[, None]) -> None`

Register callback for market data updates.

##### `def on_fill(self, callback: Callable[, None]) -> None`

Register callback for order fills.

##### `async def subscribe_market_data(self, symbol: str, exchange: str, currency: str, sec_type: str) -> bool`

Subscribe to real-time market data for a symbol.

Args:
    symbol: Ticker symbol (e.g., "AAPL", "MSFT")
    exchange: Exchange (default SMART for IB routing)
    currency: Currency (default USD)
    sec_type: Security type (STK, OPT, FUT, CASH)

Returns True if subscribed successfully.

##### `async def unsubscribe_market_data(self, symbol: str) -> None`

Unsubscribe from market data for a symbol.

##### `async def get_portfolio_state(self) -> PortfolioState`

Get current portfolio state from IB.

Returns positions, cash, P&L, etc.

##### `async def place_order(self, order_event: OrderEvent)`

Place an order with Interactive Brokers.

Args:
    order_event: The order to place

Returns:
    Broker order ID if successful, None otherwise

##### `async def cancel_order(self, broker_order_id: int) -> bool`

Cancel an order.

Returns True if cancellation was submitted successfully.

##### `async def get_historical_data(self, symbol: str, duration: str, bar_size: str, what_to_show: str, exchange: str, currency: str) -> list[dict]`

Get historical data from IB.

Args:
    symbol: Ticker symbol
    duration: Duration string (e.g., "1 D", "1 W", "1 M", "1 Y")
    bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour", "1 day")
    what_to_show: Data type (TRADES, MIDPOINT, BID, ASK)
    exchange: Exchange (default SMART)
    currency: Currency (default USD)

Returns:
    List of bar dictionaries with OHLCV data

##### `async def get_realtime_bars(self, symbol: str, bar_size: int, what_to_show: str) -> None`

Subscribe to real-time bars (5-second bars).

Args:
    symbol: Ticker symbol
    bar_size: Bar size in seconds (5 for real-time bars)
    what_to_show: Data type (TRADES, MIDPOINT, BID, ASK)

##### `def get_open_orders(self) -> list[dict]`

Get all open orders.

##### `def get_executions(self) -> list[dict]`

Get today's executions.

##### `async def request_market_data_type(self, market_data_type: int) -> None`

Set market data type.

Args:
    market_data_type:
        1 = Live (requires market data subscription)
        2 = Frozen (last available)
        3 = Delayed (15-20 min delay, free)
        4 = Delayed Frozen

##### `def set_contract_specs_manager(self, manager) -> None`

Set the contract specifications manager for margin and multiplier lookups.

##### `def check_data_staleness(self, symbol: str) -> MarketDataStaleness`

Check staleness of market data for a symbol.

Args:
    symbol: The symbol to check

Returns:
    MarketDataStaleness with detailed status

##### `def check_all_data_staleness(self) -> dict[str, MarketDataStaleness]`

Check staleness of all subscribed market data.

Returns:
    Dict mapping symbol to MarketDataStaleness

##### `def get_stale_symbols(self) -> list[str]`

Get list of symbols with stale data.

Returns:
    List of symbols where data is stale (warning level)

##### `def get_critical_stale_symbols(self) -> list[str]`

Get list of symbols with critically stale data.

Returns:
    List of symbols where data is critically stale

##### `def is_data_fresh(self, symbol: str) -> bool`

Quick check if data is fresh (not stale at all).

Args:
    symbol: The symbol to check

Returns:
    True if data is fresh, False otherwise

##### `def is_data_usable(self, symbol: str) -> bool`

Check if data is usable (not critically stale).

Args:
    symbol: The symbol to check

Returns:
    True if data can be used for trading decisions

##### `def on_staleness_alert(self, callback: Callable[, None]) -> None`

Register callback for staleness alerts.

##### `def get_data_age(self, symbol: str) -> float`

Get age of market data in seconds.

Args:
    symbol: The symbol to check

Returns:
    Age in seconds, or float('inf') if no data

##### `def get_staleness_summary(self) -> dict`

Get summary of all market data staleness.

Returns:
    Dict with staleness statistics

## Constants

- `PAPER_PORTS`
- `LIVE_PORTS`

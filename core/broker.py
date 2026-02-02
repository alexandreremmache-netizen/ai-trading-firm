"""
Interactive Brokers Integration
===============================

Exclusive broker interface for market data, portfolio state, and execution.
Paper trading is the default mode (port 7497).

IMPORTANT: Ensure TWS or IB Gateway is running before connecting.
- TWS Paper: port 7497
- TWS Live: port 7496
- Gateway Paper: port 4002
- Gateway Live: port 4001
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Any
from enum import Enum

import nest_asyncio
from ib_insync import (
    IB,
    Contract,
    Stock,
    Option,
    Future,
    Forex,
    Order,
    Trade,
    Ticker,
    MarketOrder,
    LimitOrder,
    StopOrder,
    StopLimitOrder,
    PortfolioItem,
    AccountValue,
    Fill,
    util,
)

from core.events import (
    MarketDataEvent,
    FillEvent,
    OrderEvent,
    OrderSide,
    OrderType,
)
from core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
)


# Allow nested event loops (required for ib_insync in some environments)
nest_asyncio.apply()

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Broker connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class BrokerConfig:
    """Interactive Brokers connection configuration."""
    host: str = "127.0.0.1"
    port: int = 7497  # Paper trading port (TWS)
    client_id: int = 1
    timeout_seconds: float = 30.0
    readonly: bool = False
    account: str = ""  # Leave empty for default account
    # Reconnection settings
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    initial_reconnect_delay_seconds: float = 1.0
    max_reconnect_delay_seconds: float = 60.0
    reconnect_backoff_multiplier: float = 2.0
    # Market data staleness settings
    staleness_warning_seconds: float = 5.0    # Warn if data older than 5 seconds
    staleness_critical_seconds: float = 30.0  # Reject if data older than 30 seconds
    staleness_check_enabled: bool = True


@dataclass
class MarketDataStaleness:
    """Tracks staleness status for market data."""
    symbol: str
    last_update: datetime
    age_seconds: float
    is_stale: bool
    is_critical: bool
    has_data: bool


@dataclass
class Position:
    """Current position in a symbol."""
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    exchange: str = "SMART"
    currency: str = "USD"


@dataclass
class PortfolioState:
    """Current portfolio state."""
    timestamp: datetime
    net_liquidation: float
    total_cash: float
    buying_power: float
    positions: dict[str, Position]
    daily_pnl: float
    account_id: str = ""


@dataclass
class OrderStatus:
    """Order status tracking."""
    order_id: int
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    status: str = "pending"
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class IBBroker:
    """
    Interactive Brokers integration using ib_insync.

    Responsibilities:
    - Connect to TWS/Gateway
    - Stream real-time market data
    - Query portfolio state
    - Execute orders (paper trading by default)

    This is the ONLY interface to the market.
    All market access goes through this class.
    """

    def __init__(self, config: BrokerConfig):
        self._config = config
        self._connection_state = ConnectionState.DISCONNECTED
        self._ib = IB()
        self._subscriptions: dict[str, Ticker] = {}
        self._contracts: dict[str, Contract] = {}
        self._market_data_callbacks: list[Callable[[MarketDataEvent], None]] = []
        self._fill_callbacks: list[Callable[[FillEvent], None]] = []
        self._last_portfolio_state: PortfolioState | None = None
        self._order_tracking: dict[int, OrderStatus] = {}
        self._account_id: str = ""

        # Optional components
        self._contract_specs_manager = None

        # Market data staleness tracking
        self._last_data_update: dict[str, datetime] = {}  # symbol -> last update time
        self._staleness_callbacks: list[Callable[[MarketDataStaleness], None]] = []

        # Reconnection state
        self._reconnect_attempt = 0
        self._reconnect_task: asyncio.Task | None = None
        self._should_reconnect = True
        self._last_disconnect_time: datetime | None = None
        self._disconnect_callbacks: list[Callable[[], None]] = []
        self._reconnect_callbacks: list[Callable[[], None]] = []

        # Register IB event handlers (use lambda to ensure proper binding)
        self._ib.connectedEvent += lambda: self._on_connected()
        self._ib.disconnectedEvent += lambda: self._on_disconnected()
        self._ib.errorEvent += lambda reqId, errorCode, errorString, contract: self._on_error(reqId, errorCode, errorString, contract)
        self._ib.orderStatusEvent += lambda trade: self._on_order_status(trade)
        self._ib.execDetailsEvent += lambda trade, fill: self._on_exec_details(trade, fill)

        # Circuit breaker for broker operations (#S6)
        self._circuit_breaker = CircuitBreaker(
            name="broker",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                failure_rate_threshold=0.5,
                min_calls_for_rate=10,
                reset_timeout_seconds=30.0,
                half_open_max_calls=3,
                success_threshold=2,
                call_timeout_seconds=config.timeout_seconds,
                failure_exceptions=(
                    asyncio.TimeoutError,
                    ConnectionRefusedError,
                    ConnectionError,
                    OSError,
                ),
            ),
        )
        self._circuit_breaker.on_state_change(self._on_circuit_state_change)

    @property
    def is_connected(self) -> bool:
        """Check if connected to IB."""
        return self._connection_state == ConnectionState.CONNECTED and self._ib.isConnected()

    @property
    def connection_state(self) -> ConnectionState:
        """Get current connection state."""
        return self._connection_state

    @property
    def account_id(self) -> str:
        """Get the connected account ID."""
        return self._account_id

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get the circuit breaker for this broker (#S6)."""
        return self._circuit_breaker

    def _on_circuit_state_change(
        self, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Handle circuit breaker state changes (#S6)."""
        if new_state == CircuitState.OPEN:
            logger.critical(
                f"CIRCUIT BREAKER OPEN: Broker operations suspended. "
                f"Will retry in {self._circuit_breaker._config.reset_timeout_seconds}s"
            )
        elif new_state == CircuitState.HALF_OPEN:
            logger.warning(
                "CIRCUIT BREAKER HALF_OPEN: Testing broker connectivity..."
            )
        elif new_state == CircuitState.CLOSED:
            logger.info(
                "CIRCUIT BREAKER CLOSED: Broker operations resumed"
            )

    async def connect(self) -> bool:
        """
        Connect to Interactive Brokers TWS or Gateway.

        Returns True if connected successfully.

        IMPORTANT: TWS or IB Gateway must be running with API enabled.
        Configure in TWS: Edit > Global Configuration > API > Settings
        - Enable ActiveX and Socket Clients
        - Socket port: 7497 (paper) or 7496 (live)
        - Allow connections from localhost
        """
        if self.is_connected:
            logger.info("Already connected to IB")
            return True

        self._connection_state = ConnectionState.CONNECTING

        try:
            logger.info(
                f"Connecting to IB at {self._config.host}:{self._config.port} "
                f"(client_id={self._config.client_id}, readonly={self._config.readonly})"
            )

            # Connect using ib_insync
            await self._ib.connectAsync(
                host=self._config.host,
                port=self._config.port,
                clientId=self._config.client_id,
                timeout=self._config.timeout_seconds,
                readonly=self._config.readonly,
            )

            # Get account ID
            accounts = self._ib.managedAccounts()
            if accounts:
                self._account_id = self._config.account or accounts[0]
                logger.info(f"Using account: {self._account_id}")

            self._connection_state = ConnectionState.CONNECTED
            logger.info(
                f"Connected to Interactive Brokers "
                f"(server version: {self._ib.client.serverVersion()})"
            )

            return True

        except asyncio.TimeoutError:
            logger.error(
                f"Connection timeout - is TWS/Gateway running on "
                f"{self._config.host}:{self._config.port}?"
            )
            self._connection_state = ConnectionState.ERROR
            return False

        except ConnectionRefusedError:
            logger.error(
                f"Connection refused - ensure TWS/Gateway is running and "
                f"API connections are enabled on port {self._config.port}"
            )
            self._connection_state = ConnectionState.ERROR
            return False

        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self._connection_state = ConnectionState.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        # Stop reconnection attempts
        self._should_reconnect = False

        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        if self._ib.isConnected():
            # Cancel all market data subscriptions
            for ticker in self._subscriptions.values():
                self._ib.cancelMktData(ticker.contract)

            self._subscriptions.clear()
            self._contracts.clear()

            self._ib.disconnect()
            self._connection_state = ConnectionState.DISCONNECTED
            logger.info("Disconnected from Interactive Brokers")

    async def _reconnect_with_backoff(self) -> None:
        """
        Attempt to reconnect with exponential backoff.

        Uses configurable delays and max attempts.
        """
        delay = self._config.initial_reconnect_delay_seconds

        while (
            self._should_reconnect
            and self._reconnect_attempt < self._config.max_reconnect_attempts
            and not self.is_connected
        ):
            self._reconnect_attempt += 1

            logger.info(
                f"Reconnection attempt {self._reconnect_attempt}/{self._config.max_reconnect_attempts} "
                f"in {delay:.1f}s..."
            )

            await asyncio.sleep(delay)

            if self.is_connected:
                logger.info("Already reconnected, stopping reconnection attempts")
                break

            try:
                success = await self.connect()
                if success:
                    logger.info(
                        f"Reconnected successfully after {self._reconnect_attempt} attempts"
                    )
                    # Re-subscribe to market data
                    await self._resubscribe_market_data()
                    break
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")

            # Exponential backoff
            delay = min(
                delay * self._config.reconnect_backoff_multiplier,
                self._config.max_reconnect_delay_seconds
            )

        if not self.is_connected and self._reconnect_attempt >= self._config.max_reconnect_attempts:
            logger.error(
                f"Failed to reconnect after {self._config.max_reconnect_attempts} attempts. "
                f"Manual intervention required."
            )
            self._connection_state = ConnectionState.ERROR

    async def _resubscribe_market_data(self) -> None:
        """Re-subscribe to market data after reconnection."""
        # Store subscription keys before clearing
        subscription_keys = list(self._subscriptions.keys())
        self._subscriptions.clear()
        self._contracts.clear()

        logger.info(f"Re-subscribing to {len(subscription_keys)} market data feeds...")

        for key in subscription_keys:
            parts = key.split(":")
            if len(parts) >= 3:
                symbol, exchange, currency = parts[0], parts[1], parts[2]
                try:
                    await self.subscribe_market_data(symbol, exchange, currency)
                except Exception as e:
                    logger.error(f"Failed to re-subscribe to {symbol}: {e}")

    async def _reconcile_orders_on_reconnect(self) -> None:
        """
        Reconcile order state on broker reconnection (Issue #I2).

        Compares local order tracking with broker's actual order state
        and resolves any discrepancies.
        """
        if not self.is_connected:
            return

        logger.info("Starting order reconciliation after reconnection...")

        try:
            # Get all open orders from broker
            broker_orders = self._ib.openTrades()

            # Build set of broker order IDs
            broker_order_ids = {trade.order.orderId for trade in broker_orders}

            # Track reconciliation results
            reconciled = 0
            discrepancies = 0
            orphaned_local = 0
            orphaned_broker = 0

            # Check local tracked orders against broker state
            for order_id, local_status in list(self._order_tracking.items()):
                if order_id in broker_order_ids:
                    # Order exists in broker - verify status
                    broker_trade = next(
                        t for t in broker_orders if t.order.orderId == order_id
                    )
                    broker_status = broker_trade.orderStatus.status
                    broker_filled = int(broker_trade.orderStatus.filled)
                    broker_avg_price = broker_trade.orderStatus.avgFillPrice

                    if local_status.status != broker_status:
                        logger.warning(
                            f"Order {order_id} status mismatch: "
                            f"local={local_status.status}, broker={broker_status}"
                        )
                        local_status.status = broker_status
                        discrepancies += 1

                    if local_status.filled_quantity != broker_filled:
                        logger.warning(
                            f"Order {order_id} fill mismatch: "
                            f"local={local_status.filled_quantity}, broker={broker_filled}"
                        )
                        local_status.filled_quantity = broker_filled
                        local_status.avg_fill_price = broker_avg_price
                        discrepancies += 1

                    local_status.last_update = datetime.now(timezone.utc)
                    reconciled += 1
                else:
                    # Order in local tracking but not in broker
                    # This could mean it was filled/cancelled while disconnected
                    if local_status.status not in ("Filled", "Cancelled", "Inactive"):
                        logger.warning(
                            f"Order {order_id} ({local_status.symbol}) exists locally "
                            f"but not in broker - marking as unknown"
                        )
                        local_status.status = "Unknown"
                        local_status.last_update = datetime.now(timezone.utc)
                        orphaned_local += 1

            # Check for broker orders not in local tracking
            for trade in broker_orders:
                order_id = trade.order.orderId
                if order_id not in self._order_tracking:
                    # Order exists in broker but not locally
                    logger.warning(
                        f"Order {order_id} ({trade.contract.symbol}) exists in broker "
                        f"but not in local tracking - adding to tracking"
                    )

                    # Add to local tracking
                    self._order_tracking[order_id] = OrderStatus(
                        order_id=order_id,
                        client_order_id=str(trade.order.orderId),
                        symbol=trade.contract.symbol,
                        side=OrderSide.BUY if trade.order.action == "BUY" else OrderSide.SELL,
                        quantity=int(trade.order.totalQuantity),
                        filled_quantity=int(trade.orderStatus.filled),
                        avg_fill_price=trade.orderStatus.avgFillPrice,
                        status=trade.orderStatus.status,
                        last_update=datetime.now(timezone.utc),
                    )
                    orphaned_broker += 1

            logger.info(
                f"Order reconciliation complete: "
                f"reconciled={reconciled}, discrepancies={discrepancies}, "
                f"orphaned_local={orphaned_local}, orphaned_broker={orphaned_broker}"
            )

            # Emit reconciliation event for monitoring
            self._last_reconciliation = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reconciled": reconciled,
                "discrepancies": discrepancies,
                "orphaned_local": orphaned_local,
                "orphaned_broker": orphaned_broker,
            }

        except Exception as e:
            logger.error(f"Order reconciliation failed: {e}")

    def get_reconciliation_status(self) -> dict:
        """Get the last order reconciliation status."""
        return getattr(self, '_last_reconciliation', {})

    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """Register callback for disconnection events."""
        self._disconnect_callbacks.append(callback)

    def on_reconnect(self, callback: Callable[[], None]) -> None:
        """Register callback for reconnection events."""
        self._reconnect_callbacks.append(callback)

    def enable_auto_reconnect(self) -> None:
        """Enable automatic reconnection."""
        self._should_reconnect = True
        self._reconnect_attempt = 0
        logger.info("Auto-reconnect enabled")

    def disable_auto_reconnect(self) -> None:
        """Disable automatic reconnection."""
        self._should_reconnect = False
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
        logger.info("Auto-reconnect disabled")

    def get_connection_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        return {
            "is_connected": self.is_connected,
            "connection_state": self._connection_state.value,
            "auto_reconnect": self._config.auto_reconnect,
            "reconnect_attempts": self._reconnect_attempt,
            "max_reconnect_attempts": self._config.max_reconnect_attempts,
            "last_disconnect": (
                self._last_disconnect_time.isoformat()
                if self._last_disconnect_time else None
            ),
            "subscriptions_count": len(self._subscriptions),
            "circuit_breaker": self._circuit_breaker.get_stats().to_dict(),
        }

    def on_market_data(self, callback: Callable[[MarketDataEvent], None]) -> None:
        """Register callback for market data updates."""
        self._market_data_callbacks.append(callback)

    def on_fill(self, callback: Callable[[FillEvent], None]) -> None:
        """Register callback for order fills."""
        self._fill_callbacks.append(callback)

    def _create_contract(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
        sec_type: str = "STK",
    ) -> Contract:
        """Create an IB contract for a symbol."""
        if sec_type == "STK":
            return Stock(symbol, exchange, currency)
        elif sec_type == "OPT":
            return Option(symbol, exchange=exchange, currency=currency)
        elif sec_type == "FUT":
            # For futures, we'll handle front month in subscribe_market_data
            return Future(symbol, exchange=exchange, currency=currency)
        elif sec_type == "CASH":
            # Forex pairs - convert single currency to pair (assume vs USD)
            if len(symbol) == 3:
                # Convert EUR -> EURUSD, GBP -> GBPUSD, etc.
                if symbol in ("EUR", "GBP", "AUD", "NZD"):
                    pair = f"{symbol}USD"
                elif symbol in ("JPY", "CHF", "CAD"):
                    pair = f"USD{symbol}"
                else:
                    pair = f"{symbol}USD"
                return Forex(pair)
            return Forex(symbol)
        else:
            # Generic contract
            contract = Contract()
            contract.symbol = symbol
            contract.exchange = exchange
            contract.currency = currency
            contract.secType = sec_type
            return contract

    async def _get_front_month_future(
        self,
        symbol: str,
        exchange: str,
        currency: str,
    ) -> Contract | None:
        """Get the front month (nearest expiration) futures contract."""
        try:
            # Create a generic future contract to query
            fut = Future(symbol, exchange=exchange, currency=currency)

            # Get all available contracts
            details = await self._ib.reqContractDetailsAsync(fut)

            if not details:
                logger.warning(f"No futures contracts found for {symbol}")
                return None

            # Sort by expiration and get the front month (nearest expiration that's tradeable)
            from datetime import datetime as dt
            today = dt.now().strftime("%Y%m%d")

            valid_contracts = []
            for d in details:
                exp = d.contract.lastTradeDateOrContractMonth
                if exp >= today:  # Only future expirations
                    valid_contracts.append(d.contract)

            if not valid_contracts:
                logger.warning(f"No valid futures contracts for {symbol}")
                return None

            # Sort by expiration date and get the front month
            valid_contracts.sort(key=lambda c: c.lastTradeDateOrContractMonth)
            front_month = valid_contracts[0]

            logger.info(f"Selected front month for {symbol}: {front_month.localSymbol} (exp: {front_month.lastTradeDateOrContractMonth})")
            return front_month

        except Exception as e:
            logger.error(f"Error getting front month future for {symbol}: {e}")
            return None

    async def subscribe_market_data(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
        sec_type: str = "STK",
    ) -> bool:
        """
        Subscribe to real-time market data for a symbol.

        Args:
            symbol: Ticker symbol (e.g., "AAPL", "MSFT")
            exchange: Exchange (default SMART for IB routing)
            currency: Currency (default USD)
            sec_type: Security type (STK, OPT, FUT, CASH)

        Returns True if subscribed successfully.
        """
        if not self.is_connected:
            logger.error("Cannot subscribe: not connected to IB")
            return False

        subscription_key = f"{symbol}:{exchange}:{currency}"

        if subscription_key in self._subscriptions:
            logger.debug(f"Already subscribed to {subscription_key}")
            return True

        try:
            # Handle futures specially - need to get front month
            if sec_type == "FUT":
                contract = await self._get_front_month_future(symbol, exchange, currency)
                if not contract:
                    logger.error(f"Failed to get front month future for {symbol}")
                    return False
            else:
                # Create contract for other types
                contract = self._create_contract(symbol, exchange, currency, sec_type)

                # Qualify the contract (get full details from IB)
                qualified = await self._ib.qualifyContractsAsync(contract)
                if not qualified:
                    logger.error(f"Failed to qualify contract for {symbol}")
                    return False

                contract = qualified[0]

            self._contracts[subscription_key] = contract

            # Request market data
            ticker = self._ib.reqMktData(
                contract,
                genericTickList="",  # Use default tick types
                snapshot=False,  # Stream continuously
                regulatorySnapshot=False,
            )

            # Register update handler
            ticker.updateEvent += lambda t: self._on_ticker_update(t, subscription_key)

            self._subscriptions[subscription_key] = ticker

            logger.info(f"Subscribed to market data: {subscription_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False

    async def unsubscribe_market_data(self, symbol: str) -> None:
        """Unsubscribe from market data for a symbol."""
        keys_to_remove = [k for k in self._subscriptions if k.startswith(f"{symbol}:")]

        for key in keys_to_remove:
            ticker = self._subscriptions.get(key)
            if ticker:
                self._ib.cancelMktData(ticker.contract)
                del self._subscriptions[key]
                logger.info(f"Unsubscribed from market data: {key}")

            if key in self._contracts:
                del self._contracts[key]

    async def get_portfolio_state(self) -> PortfolioState:
        """
        Get current portfolio state from IB.

        Returns positions, cash, P&L, etc.
        """
        if not self.is_connected:
            logger.warning("Not connected, returning cached portfolio state")
            if self._last_portfolio_state:
                return self._last_portfolio_state
            return PortfolioState(
                timestamp=datetime.now(timezone.utc),
                net_liquidation=0.0,
                total_cash=0.0,
                buying_power=0.0,
                positions={},
                daily_pnl=0.0,
            )

        try:
            # Use accountSummary for simpler API
            await asyncio.sleep(0.2)  # Allow for data sync

            # Parse account values from already-synced data
            account_values = {}
            for av in self._ib.accountValues():
                account_values[av.tag] = av.value

            net_liquidation = float(account_values.get("NetLiquidation", 0))
            total_cash = float(account_values.get("TotalCashValue", 0))
            buying_power = float(account_values.get("BuyingPower", 0))
            daily_pnl = float(account_values.get("DailyPnL", 0))

            # Parse positions
            positions = {}
            for item in self._ib.portfolio():
                pos = Position(
                    symbol=item.contract.symbol,
                    quantity=int(item.position),
                    avg_cost=item.averageCost,
                    market_value=item.marketValue,
                    unrealized_pnl=item.unrealizedPNL,
                    realized_pnl=item.realizedPNL,
                    exchange=item.contract.exchange,
                    currency=item.contract.currency,
                )
                positions[item.contract.symbol] = pos

            self._last_portfolio_state = PortfolioState(
                timestamp=datetime.now(timezone.utc),
                net_liquidation=net_liquidation,
                total_cash=total_cash,
                buying_power=buying_power,
                positions=positions,
                daily_pnl=daily_pnl,
                account_id=self._account_id,
            )

            return self._last_portfolio_state

        except Exception as e:
            logger.error(f"Failed to get portfolio state: {e}")
            if self._last_portfolio_state:
                return self._last_portfolio_state
            raise

    async def place_order(self, order_event: OrderEvent) -> int | None:
        """
        Place an order with Interactive Brokers.

        Args:
            order_event: The order to place

        Returns:
            Broker order ID if successful, None otherwise
        """
        if not self.is_connected:
            logger.error("Cannot place order: not connected to IB")
            return None

        try:
            # Create contract
            contract = self._create_contract(
                order_event.symbol,
                exchange="SMART",
                currency="USD",
                sec_type="STK",
            )

            # Qualify contract
            qualified = await self._ib.qualifyContractsAsync(contract)
            if not qualified:
                logger.error(f"Failed to qualify contract for {order_event.symbol}")
                return None

            contract = qualified[0]

            # Determine order action
            action = "BUY" if order_event.side == OrderSide.BUY else "SELL"

            # Create IB order based on type
            if order_event.order_type == OrderType.MARKET:
                ib_order = MarketOrder(action, order_event.quantity)

            elif order_event.order_type == OrderType.LIMIT:
                if order_event.limit_price is None:
                    logger.error("Limit order requires limit_price")
                    return None
                ib_order = LimitOrder(action, order_event.quantity, order_event.limit_price)

            elif order_event.order_type == OrderType.STOP:
                if order_event.stop_price is None:
                    logger.error("Stop order requires stop_price")
                    return None
                ib_order = StopOrder(action, order_event.quantity, order_event.stop_price)

            elif order_event.order_type == OrderType.STOP_LIMIT:
                if order_event.limit_price is None or order_event.stop_price is None:
                    logger.error("Stop-limit order requires both limit_price and stop_price")
                    return None
                ib_order = StopLimitOrder(
                    action,
                    order_event.quantity,
                    order_event.limit_price,
                    order_event.stop_price,
                )
            else:
                logger.error(f"Unsupported order type: {order_event.order_type}")
                return None

            # Set time in force (#E7)
            # Import TimeInForce if needed
            from core.events import TimeInForce
            tif_map = {
                TimeInForce.DAY: "DAY",
                TimeInForce.GTC: "GTC",
                TimeInForce.IOC: "IOC",
                TimeInForce.FOK: "FOK",
                TimeInForce.GTD: "GTD",
                TimeInForce.OPG: "OPG",
                TimeInForce.MOC: "MOC",
            }
            ib_order.tif = tif_map.get(order_event.time_in_force, "DAY")

            # Log IOC/FOK orders specially as they have specific behavior
            if order_event.time_in_force in [TimeInForce.IOC, TimeInForce.FOK]:
                logger.info(
                    f"Order with {order_event.time_in_force.value} time-in-force: "
                    f"will {'partially fill or cancel' if order_event.time_in_force == TimeInForce.IOC else 'fully fill or cancel'}"
                )

            # Place the order
            trade = self._ib.placeOrder(contract, ib_order)

            # Track order
            order_status = OrderStatus(
                order_id=trade.order.orderId,
                client_order_id=order_event.event_id,
                symbol=order_event.symbol,
                side=order_event.side,
                quantity=order_event.quantity,
                status="submitted",
            )
            self._order_tracking[trade.order.orderId] = order_status

            logger.info(
                f"Order placed: {action} {order_event.quantity} {order_event.symbol} "
                f"@ {order_event.limit_price or 'MKT'} "
                f"(order_id={trade.order.orderId})"
            )

            return trade.order.orderId

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    async def cancel_order(self, broker_order_id: int) -> bool:
        """
        Cancel an order.

        Returns True if cancellation was submitted successfully.
        """
        if not self.is_connected:
            logger.error("Cannot cancel order: not connected to IB")
            return False

        try:
            # Find the trade
            for trade in self._ib.trades():
                if trade.order.orderId == broker_order_id:
                    self._ib.cancelOrder(trade.order)
                    logger.info(f"Order cancellation submitted: {broker_order_id}")
                    return True

            logger.warning(f"Order not found for cancellation: {broker_order_id}")
            return False

        except Exception as e:
            logger.error(f"Failed to cancel order {broker_order_id}: {e}")
            return False

    async def get_historical_data(
        self,
        symbol: str,
        duration: str = "1 D",
        bar_size: str = "1 min",
        what_to_show: str = "TRADES",
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> list[dict]:
        """
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
        """
        if not self.is_connected:
            logger.error("Cannot get historical data: not connected to IB")
            return []

        try:
            # Create and qualify contract
            contract = Stock(symbol, exchange, currency)
            qualified = await self._ib.qualifyContractsAsync(contract)
            if not qualified:
                logger.error(f"Failed to qualify contract for {symbol}")
                return []

            contract = qualified[0]

            # Request historical data
            bars = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",  # Current time
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,  # Regular trading hours only
                formatDate=1,
            )

            # Convert to dictionaries
            result = []
            for bar in bars:
                result.append({
                    "date": bar.date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "average": bar.average,
                    "bar_count": bar.barCount,
                })

            logger.info(f"Retrieved {len(result)} bars for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []

    async def get_realtime_bars(
        self,
        symbol: str,
        bar_size: int = 5,
        what_to_show: str = "TRADES",
    ) -> None:
        """
        Subscribe to real-time bars (5-second bars).

        Args:
            symbol: Ticker symbol
            bar_size: Bar size in seconds (5 for real-time bars)
            what_to_show: Data type (TRADES, MIDPOINT, BID, ASK)
        """
        if not self.is_connected:
            logger.error("Cannot subscribe to real-time bars: not connected")
            return

        try:
            contract = Stock(symbol, "SMART", "USD")
            qualified = await self._ib.qualifyContractsAsync(contract)
            if not qualified:
                return

            contract = qualified[0]

            bars = self._ib.reqRealTimeBars(
                contract,
                barSize=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
            )

            logger.info(f"Subscribed to real-time bars for {symbol}")

        except Exception as e:
            logger.error(f"Failed to subscribe to real-time bars: {e}")

    # ========== Event Handlers ==========

    def _on_connected(self) -> None:
        """Handle connection event."""
        self._connection_state = ConnectionState.CONNECTED
        self._reconnect_attempt = 0  # Reset reconnect counter
        logger.info("IB connection established")

        # Schedule order reconciliation (Issue #I2)
        asyncio.create_task(self._reconcile_orders_on_reconnect())

        # Notify reconnect callbacks
        for callback in self._reconnect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Reconnect callback error: {e}")

    def _on_disconnected(self) -> None:
        """Handle disconnection event."""
        self._connection_state = ConnectionState.DISCONNECTED
        self._last_disconnect_time = datetime.now(timezone.utc)
        logger.warning("IB connection lost")

        # Notify callbacks
        for callback in self._disconnect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Disconnect callback error: {e}")

        # Start reconnection if enabled
        if self._config.auto_reconnect and self._should_reconnect:
            if self._reconnect_task is None or self._reconnect_task.done():
                self._reconnect_task = asyncio.create_task(self._reconnect_with_backoff())

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract) -> None:
        """Handle error event from IB."""
        # Common non-critical errors
        if errorCode in (2104, 2106, 2158):  # Market data farm connected/disconnected
            logger.debug(f"IB info [{errorCode}]: {errorString}")
        elif errorCode == 2103:  # Market data farm connection broken
            logger.warning(f"IB market data connection issue: {errorString}")
        elif errorCode == 10197:  # No market data during competing session
            logger.warning(f"Market data unavailable: {errorString}")
        else:
            logger.error(f"IB error [{errorCode}] reqId={reqId}: {errorString}")

    def _on_order_status(self, trade: Trade) -> None:
        """Handle order status update."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status

        if order_id in self._order_tracking:
            tracking = self._order_tracking[order_id]
            tracking.status = status
            tracking.filled_quantity = int(trade.orderStatus.filled)
            tracking.avg_fill_price = trade.orderStatus.avgFillPrice
            tracking.last_update = datetime.now(timezone.utc)

        logger.info(
            f"Order status update: {order_id} -> {status} "
            f"(filled: {trade.orderStatus.filled}/{trade.order.totalQuantity})"
        )

    def _on_exec_details(self, trade: Trade, fill: Fill) -> None:
        """Handle execution/fill details."""
        try:
            # Create fill event
            fill_event = FillEvent(
                source_agent="broker",
                order_id=str(trade.order.orderId),
                broker_order_id=trade.order.orderId,
                symbol=fill.contract.symbol,
                side=OrderSide.BUY if fill.execution.side == "BOT" else OrderSide.SELL,
                filled_quantity=int(fill.execution.shares),
                fill_price=fill.execution.price,
                commission=fill.commissionReport.commission if fill.commissionReport else 0.0,
                exchange=fill.execution.exchange,
            )

            # Notify callbacks
            for callback in self._fill_callbacks:
                try:
                    callback(fill_event)
                except Exception as e:
                    logger.error(f"Fill callback error: {e}")

            logger.info(
                f"Fill: {fill.execution.side} {fill.execution.shares} "
                f"{fill.contract.symbol} @ {fill.execution.price}"
            )

        except Exception as e:
            logger.error(f"Error processing fill: {e}")

    def _safe_int(self, value: Any) -> int:
        """Safely convert value to int, handling NaN and None."""
        import math
        if value is None:
            return 0
        try:
            if math.isnan(value):
                return 0
            return int(value)
        except (ValueError, TypeError):
            return 0

    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float, handling NaN and None."""
        import math
        if value is None:
            return 0.0
        try:
            if math.isnan(value):
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _on_ticker_update(self, ticker: Ticker, subscription_key: str) -> None:
        """Handle ticker update from IB."""
        try:
            # Extract symbol from subscription key
            symbol = subscription_key.split(":")[0]

            # Track last update time for staleness detection
            self._last_data_update[symbol] = datetime.now(timezone.utc)

            # Create market data event (with safe conversions for NaN values)
            event = MarketDataEvent(
                source_agent="broker",
                symbol=symbol,
                exchange=ticker.contract.exchange if ticker.contract else "SMART",
                bid=self._safe_float(ticker.bid) if ticker.bid and self._safe_float(ticker.bid) > 0 else 0.0,
                ask=self._safe_float(ticker.ask) if ticker.ask and self._safe_float(ticker.ask) > 0 else 0.0,
                last=self._safe_float(ticker.last) if ticker.last and self._safe_float(ticker.last) > 0 else 0.0,
                volume=self._safe_int(ticker.volume),
                bid_size=self._safe_int(ticker.bidSize),
                ask_size=self._safe_int(ticker.askSize),
                high=self._safe_float(ticker.high),
                low=self._safe_float(ticker.low),
                open_price=self._safe_float(ticker.open),
                close=self._safe_float(ticker.close),
            )

            # Notify callbacks
            for callback in self._market_data_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Market data callback error: {e}")

        except Exception as e:
            logger.error(f"Error processing ticker update: {e}")

    # ========== Utility Methods ==========

    def get_open_orders(self) -> list[dict]:
        """Get all open orders."""
        orders = []
        for trade in self._ib.openTrades():
            orders.append({
                "order_id": trade.order.orderId,
                "symbol": trade.contract.symbol,
                "action": trade.order.action,
                "quantity": trade.order.totalQuantity,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
                "status": trade.orderStatus.status,
                "order_type": trade.order.orderType,
                "limit_price": trade.order.lmtPrice,
                "stop_price": trade.order.auxPrice,
            })
        return orders

    def get_executions(self) -> list[dict]:
        """Get today's executions."""
        executions = []
        for fill in self._ib.fills():
            executions.append({
                "exec_id": fill.execution.execId,
                "symbol": fill.contract.symbol,
                "side": fill.execution.side,
                "quantity": fill.execution.shares,
                "price": fill.execution.price,
                "time": fill.execution.time,
                "exchange": fill.execution.exchange,
                "commission": fill.commissionReport.commission if fill.commissionReport else 0,
            })
        return executions

    async def request_market_data_type(self, market_data_type: int = 1) -> None:
        """
        Set market data type.

        Args:
            market_data_type:
                1 = Live (requires market data subscription)
                2 = Frozen (last available)
                3 = Delayed (15-20 min delay, free)
                4 = Delayed Frozen
        """
        self._ib.reqMarketDataType(market_data_type)
        logger.info(f"Market data type set to {market_data_type}")

    def set_contract_specs_manager(self, manager) -> None:
        """Set the contract specifications manager for margin and multiplier lookups."""
        self._contract_specs_manager = manager
        logger.info("Contract specs manager attached to broker")

    # ========== Market Data Staleness Detection ==========

    def check_data_staleness(self, symbol: str) -> MarketDataStaleness:
        """
        Check staleness of market data for a symbol.

        Args:
            symbol: The symbol to check

        Returns:
            MarketDataStaleness with detailed status
        """
        now = datetime.now(timezone.utc)
        last_update = self._last_data_update.get(symbol)

        if last_update is None:
            return MarketDataStaleness(
                symbol=symbol,
                last_update=now,
                age_seconds=float('inf'),
                is_stale=True,
                is_critical=True,
                has_data=False,
            )

        age_seconds = (now - last_update).total_seconds()
        is_stale = age_seconds > self._config.staleness_warning_seconds
        is_critical = age_seconds > self._config.staleness_critical_seconds

        return MarketDataStaleness(
            symbol=symbol,
            last_update=last_update,
            age_seconds=age_seconds,
            is_stale=is_stale,
            is_critical=is_critical,
            has_data=True,
        )

    def check_all_data_staleness(self) -> dict[str, MarketDataStaleness]:
        """
        Check staleness of all subscribed market data.

        Returns:
            Dict mapping symbol to MarketDataStaleness
        """
        result = {}
        for subscription_key in self._subscriptions.keys():
            symbol = subscription_key.split(":")[0]
            if symbol not in result:
                result[symbol] = self.check_data_staleness(symbol)
        return result

    def get_stale_symbols(self) -> list[str]:
        """
        Get list of symbols with stale data.

        Returns:
            List of symbols where data is stale (warning level)
        """
        stale = []
        for subscription_key in self._subscriptions.keys():
            symbol = subscription_key.split(":")[0]
            if symbol not in stale:
                staleness = self.check_data_staleness(symbol)
                if staleness.is_stale:
                    stale.append(symbol)
        return stale

    def get_critical_stale_symbols(self) -> list[str]:
        """
        Get list of symbols with critically stale data.

        Returns:
            List of symbols where data is critically stale
        """
        critical = []
        for subscription_key in self._subscriptions.keys():
            symbol = subscription_key.split(":")[0]
            if symbol not in critical:
                staleness = self.check_data_staleness(symbol)
                if staleness.is_critical:
                    critical.append(symbol)
        return critical

    def is_data_fresh(self, symbol: str) -> bool:
        """
        Quick check if data is fresh (not stale at all).

        Args:
            symbol: The symbol to check

        Returns:
            True if data is fresh, False otherwise
        """
        staleness = self.check_data_staleness(symbol)
        return staleness.has_data and not staleness.is_stale

    def is_data_usable(self, symbol: str) -> bool:
        """
        Check if data is usable (not critically stale).

        Args:
            symbol: The symbol to check

        Returns:
            True if data can be used for trading decisions
        """
        if not self._config.staleness_check_enabled:
            return True

        staleness = self.check_data_staleness(symbol)
        if staleness.is_critical:
            logger.warning(
                f"Market data for {symbol} is critically stale "
                f"(age={staleness.age_seconds:.1f}s, threshold={self._config.staleness_critical_seconds}s)"
            )
            return False
        if staleness.is_stale:
            logger.info(
                f"Market data for {symbol} is stale "
                f"(age={staleness.age_seconds:.1f}s, threshold={self._config.staleness_warning_seconds}s)"
            )
        return True

    def on_staleness_alert(self, callback: Callable[[MarketDataStaleness], None]) -> None:
        """Register callback for staleness alerts."""
        self._staleness_callbacks.append(callback)

    def get_data_age(self, symbol: str) -> float:
        """
        Get age of market data in seconds.

        Args:
            symbol: The symbol to check

        Returns:
            Age in seconds, or float('inf') if no data
        """
        last_update = self._last_data_update.get(symbol)
        if last_update is None:
            return float('inf')
        return (datetime.now(timezone.utc) - last_update).total_seconds()

    def get_staleness_summary(self) -> dict:
        """
        Get summary of all market data staleness.

        Returns:
            Dict with staleness statistics
        """
        all_staleness = self.check_all_data_staleness()

        fresh_count = sum(1 for s in all_staleness.values() if s.has_data and not s.is_stale)
        stale_count = sum(1 for s in all_staleness.values() if s.is_stale and not s.is_critical)
        critical_count = sum(1 for s in all_staleness.values() if s.is_critical)
        no_data_count = sum(1 for s in all_staleness.values() if not s.has_data)

        ages = [s.age_seconds for s in all_staleness.values() if s.has_data and s.age_seconds != float('inf')]

        return {
            "total_symbols": len(all_staleness),
            "fresh": fresh_count,
            "stale_warning": stale_count,
            "stale_critical": critical_count,
            "no_data": no_data_count,
            "avg_age_seconds": sum(ages) / len(ages) if ages else 0,
            "max_age_seconds": max(ages) if ages else 0,
            "stale_symbols": self.get_stale_symbols(),
            "critical_symbols": self.get_critical_stale_symbols(),
        }

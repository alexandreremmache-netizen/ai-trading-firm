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
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque
from functools import partial, wraps
from typing import Callable, Any, TypeVar
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
    # P0-1/P0-2: Environment safety settings
    environment: str = "paper"  # "paper" or "live" - MUST match port configuration
    # P2: Order status polling fallback
    order_polling_enabled: bool = True
    order_polling_interval_seconds: float = 5.0
    order_polling_timeout_seconds: float = 60.0  # Stop polling after this time
    # P2: Connection quality metrics
    connection_quality_window_seconds: float = 60.0
    # P2: Session recovery
    session_recovery_enabled: bool = True


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


# ============================================================================
# P2: Connection Quality Metrics
# ============================================================================

@dataclass
class ConnectionQualityMetrics:
    """Metrics for tracking connection quality to broker."""
    latency_samples: list[float] = field(default_factory=list)
    reconnect_count: int = 0
    total_disconnects: int = 0
    last_disconnect_time: datetime | None = None
    last_reconnect_time: datetime | None = None
    connection_uptime_seconds: float = 0.0
    connection_start_time: datetime | None = None
    message_count: int = 0
    error_count: int = 0
    pacing_violations: int = 0

    def record_latency(self, latency_ms: float, max_samples: int = 100) -> None:
        """Record a latency sample."""
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > max_samples:
            self.latency_samples.pop(0)

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)

    @property
    def max_latency_ms(self) -> float:
        """Get maximum latency."""
        return max(self.latency_samples) if self.latency_samples else 0.0

    @property
    def min_latency_ms(self) -> float:
        """Get minimum latency."""
        return min(self.latency_samples) if self.latency_samples else 0.0

    @property
    def uptime_percentage(self) -> float:
        """Calculate connection uptime percentage."""
        if self.connection_start_time is None:
            return 0.0
        total_time = (datetime.now(timezone.utc) - self.connection_start_time).total_seconds()
        if total_time == 0:
            return 100.0
        return (self.connection_uptime_seconds / total_time) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "reconnect_count": self.reconnect_count,
            "total_disconnects": self.total_disconnects,
            "uptime_percentage": round(self.uptime_percentage, 2),
            "connection_uptime_seconds": round(self.connection_uptime_seconds, 1),
            "message_count": self.message_count,
            "error_count": self.error_count,
            "pacing_violations": self.pacing_violations,
            "last_disconnect": (
                self.last_disconnect_time.isoformat()
                if self.last_disconnect_time else None
            ),
            "last_reconnect": (
                self.last_reconnect_time.isoformat()
                if self.last_reconnect_time else None
            ),
        }


# ============================================================================
# P2: Session Recovery State
# ============================================================================

@dataclass
class SessionState:
    """State for session recovery after reconnection."""
    subscribed_symbols: list[str] = field(default_factory=list)
    pending_orders: dict[int, dict] = field(default_factory=dict)
    last_known_positions: dict[str, dict] = field(default_factory=dict)
    last_portfolio_state: dict | None = None
    session_start_time: datetime | None = None
    last_save_time: datetime | None = None

    def save_subscription(self, symbol: str, exchange: str, currency: str) -> None:
        """Save a subscription for recovery."""
        key = f"{symbol}:{exchange}:{currency}"
        if key not in self.subscribed_symbols:
            self.subscribed_symbols.append(key)

    def remove_subscription(self, symbol: str) -> None:
        """Remove a subscription."""
        self.subscribed_symbols = [
            s for s in self.subscribed_symbols
            if not s.startswith(f"{symbol}:")
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary for persistence."""
        return {
            "subscribed_symbols": self.subscribed_symbols,
            "pending_orders": self.pending_orders,
            "last_known_positions": self.last_known_positions,
            "session_start_time": (
                self.session_start_time.isoformat()
                if self.session_start_time else None
            ),
            "last_save_time": (
                self.last_save_time.isoformat()
                if self.last_save_time else None
            ),
        }


class IBRateLimiter:
    """
    IB API Rate Limiter (P0-1 fix).

    Interactive Brokers enforces rate limits:
    - Max 60 requests per 10 minutes (600 seconds) for market data
    - No duplicate requests within 15 seconds
    - Exceeding limits causes temporary bans

    This class implements a sliding window rate limiter to prevent
    hitting IB's rate limits and causing connection issues.

    P0-10: Added exponential backoff support for pacing violations.
    """

    # IB rate limit constants
    MAX_REQUESTS = 60
    WINDOW_SECONDS = 600  # 10 minutes
    MIN_REQUEST_INTERVAL = 15.0  # Minimum seconds between identical requests

    # Exponential backoff settings (P0-10)
    INITIAL_BACKOFF_SECONDS = 60.0    # 1 minute initial backoff
    MAX_BACKOFF_SECONDS = 300.0       # 5 minute max backoff
    BACKOFF_MULTIPLIER = 2.0          # Double each time
    BACKOFF_RESET_SECONDS = 600.0     # Reset backoff after 10 minutes of success

    def __init__(self):
        # Sliding window of request timestamps
        self._request_times: deque[datetime] = deque()
        # Track last request time per request key (for duplicate detection)
        self._last_request: dict[str, datetime] = {}
        # P0-10: Exponential backoff state
        self._violation_count: int = 0
        self._last_violation_time: datetime | None = None
        self._backoff_until: datetime | None = None

    def _clean_old_requests(self) -> None:
        """Remove requests older than the window."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.WINDOW_SECONDS)
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()

    def record_pacing_violation(self) -> float:
        """
        Record a pacing violation and calculate backoff time (P0-10).

        Call this when IB returns error code 162 (pacing violation).

        Returns:
            Backoff time in seconds
        """
        now = datetime.now(timezone.utc)

        # Reset violation count if enough time has passed
        if self._last_violation_time:
            elapsed = (now - self._last_violation_time).total_seconds()
            if elapsed > self.BACKOFF_RESET_SECONDS:
                self._violation_count = 0

        self._violation_count += 1
        self._last_violation_time = now

        # Calculate exponential backoff
        backoff = min(
            self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_MULTIPLIER ** (self._violation_count - 1)),
            self.MAX_BACKOFF_SECONDS
        )

        self._backoff_until = now + timedelta(seconds=backoff)
        logger.warning(
            f"Pacing violation #{self._violation_count}: backing off for {backoff:.0f}s "
            f"(until {self._backoff_until.isoformat()})"
        )

        return backoff

    def is_in_backoff(self) -> tuple[bool, float]:
        """
        Check if currently in backoff period (P0-10).

        Returns:
            Tuple of (is_in_backoff, remaining_seconds)
        """
        if self._backoff_until is None:
            return False, 0.0

        now = datetime.now(timezone.utc)
        if now >= self._backoff_until:
            return False, 0.0

        remaining = (self._backoff_until - now).total_seconds()
        return True, remaining

    def can_make_request(self, request_key: str | None = None) -> tuple[bool, str]:
        """
        Check if a request can be made within rate limits.

        Args:
            request_key: Optional key for duplicate detection (e.g., "mktdata:AAPL")

        Returns:
            Tuple of (can_request, reason_if_not)
        """
        # P0-10: Check backoff first
        in_backoff, backoff_remaining = self.is_in_backoff()
        if in_backoff:
            return False, f"In backoff period due to pacing violation. Wait {backoff_remaining:.1f}s"

        self._clean_old_requests()
        now = datetime.now(timezone.utc)

        # Check duplicate request interval
        if request_key and request_key in self._last_request:
            elapsed = (now - self._last_request[request_key]).total_seconds()
            if elapsed < self.MIN_REQUEST_INTERVAL:
                wait_time = self.MIN_REQUEST_INTERVAL - elapsed
                return False, f"Duplicate request too soon. Wait {wait_time:.1f}s"

        # Check overall rate limit
        if len(self._request_times) >= self.MAX_REQUESTS:
            oldest = self._request_times[0]
            wait_time = (oldest + timedelta(seconds=self.WINDOW_SECONDS) - now).total_seconds()
            return False, f"Rate limit reached ({self.MAX_REQUESTS}/{self.WINDOW_SECONDS}s). Wait {wait_time:.1f}s"

        return True, ""

    def record_request(self, request_key: str | None = None) -> None:
        """Record that a request was made."""
        now = datetime.now(timezone.utc)
        self._request_times.append(now)
        if request_key:
            self._last_request[request_key] = now

    def get_remaining_requests(self) -> int:
        """Get number of requests remaining in current window."""
        self._clean_old_requests()
        return max(0, self.MAX_REQUESTS - len(self._request_times))

    def get_wait_time(self, request_key: str | None = None) -> float:
        """Get seconds to wait before next request is allowed."""
        # P0-10: Check backoff first
        in_backoff, backoff_remaining = self.is_in_backoff()
        if in_backoff:
            return backoff_remaining

        can_request, _ = self.can_make_request(request_key)
        if can_request:
            return 0.0

        self._clean_old_requests()
        now = datetime.now(timezone.utc)

        # Check duplicate interval
        if request_key and request_key in self._last_request:
            elapsed = (now - self._last_request[request_key]).total_seconds()
            if elapsed < self.MIN_REQUEST_INTERVAL:
                return self.MIN_REQUEST_INTERVAL - elapsed

        # Check rate limit
        if len(self._request_times) >= self.MAX_REQUESTS:
            oldest = self._request_times[0]
            return (oldest + timedelta(seconds=self.WINDOW_SECONDS) - now).total_seconds()

        return 0.0

    def get_stats(self) -> dict:
        """Get rate limiter statistics (P0-10)."""
        self._clean_old_requests()
        in_backoff, backoff_remaining = self.is_in_backoff()
        return {
            "requests_in_window": len(self._request_times),
            "remaining_requests": self.get_remaining_requests(),
            "violation_count": self._violation_count,
            "in_backoff": in_backoff,
            "backoff_remaining_seconds": backoff_remaining,
            "last_violation": (
                self._last_violation_time.isoformat()
                if self._last_violation_time else None
            ),
        }


# Paper vs Live port mapping for validation
PAPER_PORTS = {7497, 4002}  # TWS Paper, Gateway Paper
LIVE_PORTS = {7496, 4001}   # TWS Live, Gateway Live


# =============================================================================
# P3: Retry Handler with Jitter and Budget Tracking
# =============================================================================

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior with jitter support (P3 fix)."""
    # Basic retry settings
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0

    # P3: Jitter settings to prevent thundering herd
    jitter_mode: str = "full"  # "none", "full", "equal", "decorrelated"
    jitter_factor: float = 0.5  # Max jitter as fraction of delay (for "equal" mode)

    # P3: Retry budget tracking
    budget_window_seconds: float = 60.0  # Time window for budget tracking
    budget_max_retries: int = 10  # Max retries allowed in budget window
    budget_enabled: bool = True

    # Exceptions that should be retried
    retryable_exceptions: tuple[type, ...] = (
        asyncio.TimeoutError,
        ConnectionError,
        ConnectionRefusedError,
        OSError,
    )

    # Exceptions that should NOT be retried (even if they match retryable)
    non_retryable_exceptions: tuple[type, ...] = (
        ValueError,
        TypeError,
        KeyError,
    )


@dataclass
class RetryBudgetStats:
    """Statistics for retry budget tracking (P3 fix)."""
    total_retries: int = 0
    retries_in_window: int = 0
    budget_exhausted_count: int = 0
    last_retry_time: datetime | None = None
    window_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RetryHandler:
    """
    Retry handler with jitter and budget tracking (P3 fix).

    Features:
    - Exponential backoff with configurable multiplier
    - Jitter to prevent thundering herd problem
    - Retry budget to prevent retry storms
    - Circuit breaker integration

    Jitter Modes:
    - "none": No jitter, use exact delay
    - "full": Random delay between 0 and calculated delay
    - "equal": Half calculated delay + random half
    - "decorrelated": Decorrelated jitter (best for distributed systems)

    Usage:
        handler = RetryHandler(config)

        # As decorator
        @handler.with_retry
        async def call_broker():
            ...

        # Or manually
        result = await handler.execute_with_retry(call_broker)
    """

    def __init__(
        self,
        config: RetryConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self._config = config or RetryConfig()
        self._circuit_breaker = circuit_breaker

        # P3: Budget tracking
        self._budget_stats = RetryBudgetStats()
        self._retry_timestamps: deque[datetime] = deque()

        # Decorrelated jitter state
        self._last_delay = self._config.initial_delay_seconds

    def _calculate_delay_with_jitter(self, attempt: int) -> float:
        """
        Calculate delay with jitter applied (P3 fix).

        Args:
            attempt: Current retry attempt (0-indexed)

        Returns:
            Delay in seconds with jitter applied
        """
        # Base exponential backoff
        base_delay = self._config.initial_delay_seconds * (
            self._config.backoff_multiplier ** attempt
        )
        base_delay = min(base_delay, self._config.max_delay_seconds)

        jitter_mode = self._config.jitter_mode.lower()

        if jitter_mode == "none":
            return base_delay

        elif jitter_mode == "full":
            # Full jitter: random value between 0 and base_delay
            return random.uniform(0, base_delay)

        elif jitter_mode == "equal":
            # Equal jitter: base/2 + random(0, base/2)
            half = base_delay / 2
            jitter = random.uniform(0, half * self._config.jitter_factor * 2)
            return half + jitter

        elif jitter_mode == "decorrelated":
            # Decorrelated jitter: better for distributed systems
            # sleep = min(cap, random(base, sleep * 3))
            self._last_delay = min(
                self._config.max_delay_seconds,
                random.uniform(
                    self._config.initial_delay_seconds,
                    self._last_delay * 3
                )
            )
            return self._last_delay

        else:
            logger.warning(f"Unknown jitter mode '{jitter_mode}', using full jitter")
            return random.uniform(0, base_delay)

    def _check_retry_budget(self) -> tuple[bool, str]:
        """
        Check if retry budget allows another retry (P3 fix).

        Returns:
            Tuple of (allowed, reason_if_not)
        """
        if not self._config.budget_enabled:
            return True, ""

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._config.budget_window_seconds)

        # Clean old timestamps
        while self._retry_timestamps and self._retry_timestamps[0] < cutoff:
            self._retry_timestamps.popleft()

        self._budget_stats.retries_in_window = len(self._retry_timestamps)

        if self._budget_stats.retries_in_window >= self._config.budget_max_retries:
            self._budget_stats.budget_exhausted_count += 1
            return False, (
                f"Retry budget exhausted: {self._budget_stats.retries_in_window}/"
                f"{self._config.budget_max_retries} retries in "
                f"{self._config.budget_window_seconds}s window"
            )

        return True, ""

    def _record_retry(self) -> None:
        """Record a retry attempt for budget tracking."""
        now = datetime.now(timezone.utc)
        self._retry_timestamps.append(now)
        self._budget_stats.total_retries += 1
        self._budget_stats.last_retry_time = now

    def _is_retryable(self, exc: Exception) -> bool:
        """Check if an exception should be retried."""
        # Check non-retryable first (they take precedence)
        if isinstance(exc, self._config.non_retryable_exceptions):
            return False

        # Check if it's a retryable exception
        return isinstance(exc, self._config.retryable_exceptions)

    async def execute_with_retry(
        self,
        func: Callable[..., Any],
        *args,
        operation_name: str = "",
        **kwargs,
    ) -> Any:
        """
        Execute a function with retry logic (P3 fix).

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            operation_name: Optional name for logging
            **kwargs: Keyword arguments for func

        Returns:
            Result of the function

        Raises:
            Exception: The last exception if all retries exhausted
        """
        operation = operation_name or func.__name__
        last_exception: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            # Check retry budget (except for first attempt)
            if attempt > 0:
                budget_ok, budget_reason = self._check_retry_budget()
                if not budget_ok:
                    logger.warning(
                        f"Retry budget exhausted for {operation}: {budget_reason}"
                    )
                    if last_exception:
                        raise last_exception
                    raise RuntimeError(budget_reason)

            # Check circuit breaker if configured
            if self._circuit_breaker and self._circuit_breaker.is_open:
                logger.warning(
                    f"Circuit breaker open, skipping retry for {operation}"
                )
                if last_exception:
                    raise last_exception
                raise CircuitOpenError(
                    self._circuit_breaker.name,
                    self._circuit_breaker._config.reset_timeout_seconds
                )

            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Reset decorrelated jitter state on success
                self._last_delay = self._config.initial_delay_seconds

                return result

            except Exception as e:
                last_exception = e

                # Check if this exception is retryable
                if not self._is_retryable(e):
                    logger.debug(
                        f"Non-retryable exception for {operation}: {type(e).__name__}"
                    )
                    raise

                # Check if we have more attempts
                if attempt >= self._config.max_retries:
                    logger.error(
                        f"All {self._config.max_retries + 1} attempts exhausted "
                        f"for {operation}: {e}"
                    )
                    raise

                # Record retry for budget tracking
                self._record_retry()

                # Calculate delay with jitter
                delay = self._calculate_delay_with_jitter(attempt)

                logger.warning(
                    f"Retry {attempt + 1}/{self._config.max_retries} for {operation} "
                    f"after {delay:.2f}s (error: {type(e).__name__}: {e})"
                )

                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Unexpected state in retry handler for {operation}")

    def with_retry(
        self,
        func: Callable[..., Any] | None = None,
        *,
        operation_name: str = "",
    ):
        """
        Decorator to add retry logic to a function (P3 fix).

        Usage:
            @handler.with_retry
            async def my_func():
                ...

            @handler.with_retry(operation_name="custom_op")
            async def my_func():
                ...
        """
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(fn)
            async def wrapper(*args, **kwargs):
                return await self.execute_with_retry(
                    fn, *args, operation_name=operation_name or fn.__name__, **kwargs
                )
            return wrapper

        if func is not None:
            return decorator(func)
        return decorator

    def get_stats(self) -> dict[str, Any]:
        """Get retry handler statistics."""
        self._check_retry_budget()  # Update budget stats
        return {
            "total_retries": self._budget_stats.total_retries,
            "retries_in_window": self._budget_stats.retries_in_window,
            "budget_max": self._config.budget_max_retries,
            "budget_window_seconds": self._config.budget_window_seconds,
            "budget_exhausted_count": self._budget_stats.budget_exhausted_count,
            "last_retry_time": (
                self._budget_stats.last_retry_time.isoformat()
                if self._budget_stats.last_retry_time else None
            ),
            "circuit_breaker_state": (
                self._circuit_breaker.state.value
                if self._circuit_breaker else None
            ),
        }

    def reset_budget(self) -> None:
        """Reset the retry budget (e.g., after a successful operation)."""
        self._retry_timestamps.clear()
        self._budget_stats.retries_in_window = 0
        logger.debug("Retry budget reset")


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

        # P0-3: Store event handler references for proper cleanup (prevents memory leak)
        # Lambda closures capture 'self' and are never garbage collected if not removed
        self._ticker_callbacks: dict[str, Callable] = {}  # subscription_key -> callback

        # P0-5: Lock for thread-safe access to _subscriptions during reconnection
        # Prevents race condition when reconnect clears subscriptions while handlers access them
        self._subscriptions_lock = asyncio.Lock()

        # P0-3: Register IB event handlers using bound methods (not lambdas)
        # Store references so we can unregister them on cleanup
        self._ib.connectedEvent += self._on_connected
        self._ib.disconnectedEvent += self._on_disconnected
        self._ib.errorEvent += self._on_error
        self._ib.orderStatusEvent += self._on_order_status
        self._ib.execDetailsEvent += self._on_exec_details

        # P0-1: IB API rate limiter to prevent hitting IB's rate limits
        self._rate_limiter = IBRateLimiter()

        # ERR-001: Idempotency tracking for order placement
        # Prevents duplicate orders when retry happens after timeout but order was placed
        self._processed_orders: dict[str, int] = {}  # idempotency_key -> broker_order_id

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

        # P2: Connection quality metrics
        self._connection_quality = ConnectionQualityMetrics()

        # P2: Session recovery state
        self._session_state = SessionState()

        # P2: Order status polling task
        self._order_polling_task: asyncio.Task | None = None
        self._orders_being_polled: dict[int, datetime] = {}  # order_id -> start_time

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

    @property
    def rate_limiter(self) -> IBRateLimiter:
        """Get the IB API rate limiter (P0-1)."""
        return self._rate_limiter

    def _validate_paper_vs_live_config(self) -> None:
        """
        Validate that environment setting matches port configuration (P0-2).

        CRITICAL SAFETY CHECK: Prevents accidentally trading live when
        expecting paper trading, or vice versa.

        Raises:
            ValueError: If environment doesn't match port
        """
        port = self._config.port
        env = self._config.environment.lower()

        is_paper_port = port in PAPER_PORTS
        is_live_port = port in LIVE_PORTS

        if env == "paper":
            if is_live_port:
                raise ValueError(
                    f"CRITICAL SAFETY ERROR: Environment is 'paper' but port {port} "
                    f"is a LIVE trading port! This could result in real trades. "
                    f"Either change port to {7497} (TWS Paper) or {4002} (Gateway Paper), "
                    f"or set environment='live' if you truly intend to trade live."
                )
            if not is_paper_port:
                logger.warning(
                    f"Port {port} is not a standard IB port. "
                    f"Standard paper ports are: {PAPER_PORTS}"
                )
        elif env == "live":
            if is_paper_port:
                raise ValueError(
                    f"CRITICAL CONFIGURATION ERROR: Environment is 'live' but port {port} "
                    f"is a PAPER trading port. Change port to {7496} (TWS Live) or "
                    f"{4001} (Gateway Live) for live trading."
                )
            if not is_live_port:
                logger.warning(
                    f"Port {port} is not a standard IB port. "
                    f"Standard live ports are: {LIVE_PORTS}"
                )
            # Extra warning for live trading
            logger.warning(
                "⚠️  LIVE TRADING MODE ENABLED - Real money will be used! ⚠️"
            )
        else:
            raise ValueError(
                f"Invalid environment: '{env}'. Must be 'paper' or 'live'."
            )

    def _validate_paper_account(self) -> bool:
        """
        Validate that account ID matches expected paper/live configuration (P0-2).

        Returns:
            True if account appears to match environment, False otherwise.
            Returns True for live trading (can't distinguish live accounts).

        Note:
            Paper accounts typically start with 'D' (e.g., 'DU1234567').
            This is a heuristic check, not guaranteed.
        """
        if not self._account_id:
            return True  # Can't validate without account ID

        is_demo_account = self._account_id.upper().startswith("D")
        expected_paper = self._config.environment.lower() == "paper"

        if expected_paper and not is_demo_account:
            logger.critical(
                f"⚠️  CRITICAL WARNING: Environment is 'paper' but account "
                f"'{self._account_id}' does not appear to be a demo account! "
                f"Demo accounts typically start with 'D'. "
                f"Please verify you are connected to paper trading."
            )
            return False

        if not expected_paper and is_demo_account:
            logger.warning(
                f"Environment is 'live' but account '{self._account_id}' "
                f"appears to be a demo account (starts with 'D')."
            )
            return False

        return True

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

    async def connect(
        self,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ) -> bool:
        """
        Connect to Interactive Brokers TWS or Gateway.

        Args:
            max_retries: Maximum number of connection attempts (default: 3).
                         ERR-002 fix: Prevents startup failure on transient network issues.
            backoff_base: Base for exponential backoff between retries (default: 2.0).
                          Delay = backoff_base ** attempt_number seconds.

        Returns True if connected successfully.

        IMPORTANT: TWS or IB Gateway must be running with API enabled.
        Configure in TWS: Edit > Global Configuration > API > Settings
        - Enable ActiveX and Socket Clients
        - Socket port: 7497 (paper) or 7496 (live)
        - Allow connections from localhost
        """
        # P0-2: Validate paper/live configuration BEFORE connecting
        # This prevents accidentally connecting to live trading
        self._validate_paper_vs_live_config()

        if self.is_connected:
            logger.info("Already connected to IB")
            return True

        # ERR-002: Retry with exponential backoff for transient network issues
        last_error: Exception | None = None
        for attempt in range(max_retries):
            if attempt > 0:
                delay = backoff_base ** attempt
                logger.info(
                    f"Connection attempt {attempt + 1}/{max_retries} "
                    f"after {delay:.1f}s backoff..."
                )
                await asyncio.sleep(delay)

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

                # P0-8: Validate API is actually enabled and functional
                # connectAsync can succeed but API may not be enabled in TWS settings
                if not self._ib.isConnected():
                    logger.error(
                        "Connection appeared successful but API is not responsive. "
                        "Ensure API is enabled in TWS: Edit > Global Configuration > API > Settings"
                    )
                    self._connection_state = ConnectionState.ERROR
                    return False

                # Verify we can actually communicate with the API
                server_version = self._ib.client.serverVersion()
                if server_version < 100:  # Minimum reasonable version
                    logger.warning(
                        f"IB Gateway/TWS version may be outdated (server version: {server_version}). "
                        f"Some features may not work correctly."
                    )

                # Get account ID
                accounts = self._ib.managedAccounts()
                if not accounts:
                    logger.error(
                        "No managed accounts returned. API may not be fully enabled or "
                        "account permissions may be restricted."
                    )
                    self._connection_state = ConnectionState.ERROR
                    return False

                self._account_id = self._config.account or accounts[0]
                logger.info(f"Using account: {self._account_id}")

                # P0-2: Validate account matches expected environment
                self._validate_paper_account()

                # P0-8: Test API functionality with a real request
                try:
                    # Give IB a moment to sync initial data
                    await asyncio.sleep(0.5)
                    # Request current time from server - this is a true API test
                    server_time = await asyncio.wait_for(
                        self._ib.reqCurrentTimeAsync(),
                        timeout=5.0
                    )
                    if server_time:
                        logger.debug(f"API functionality verified (server time: {server_time})")
                    else:
                        logger.warning("API returned empty time response")
                except asyncio.TimeoutError:
                    logger.error("API validation failed - server not responding to requests")
                    self._connection_state = ConnectionState.ERROR
                    return False
                except Exception as e:
                    logger.warning(f"API validation warning: {e}")

                self._connection_state = ConnectionState.CONNECTED
                logger.info(
                    f"Connected to Interactive Brokers "
                    f"(server version: {server_version})"
                )

                return True

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    f"Connection attempt {attempt + 1}/{max_retries} timed out - "
                    f"is TWS/Gateway running on {self._config.host}:{self._config.port}?"
                )
                self._connection_state = ConnectionState.ERROR
                # Continue to next retry attempt

            except ConnectionRefusedError as e:
                last_error = e
                logger.warning(
                    f"Connection attempt {attempt + 1}/{max_retries} refused - "
                    f"ensure TWS/Gateway is running and API connections are enabled "
                    f"on port {self._config.port}"
                )
                self._connection_state = ConnectionState.ERROR
                # Continue to next retry attempt

            except OSError as e:
                # Socket-level errors (network issues, port in use, etc.)
                last_error = e
                logger.warning(
                    f"Connection attempt {attempt + 1}/{max_retries} socket error: {e}"
                )
                self._connection_state = ConnectionState.ERROR

            except Exception as e:
                # Broad exception catch is intentional here - IB can raise various
                # undocumented exceptions during connection (socket errors, SSL errors,
                # protocol errors). We want to gracefully handle any failure and
                # set the connection state appropriately rather than crash.
                # P3: Use logger.exception to preserve traceback for debugging
                last_error = e
                logger.exception(
                    f"Connection attempt {attempt + 1}/{max_retries} failed with "
                    f"{type(e).__name__}"
                )
                self._connection_state = ConnectionState.ERROR
                # Continue to next retry attempt

        # ERR-002: All retries exhausted
        logger.error(
            f"Failed to connect to IB after {max_retries} attempts. "
            f"Last error: {last_error}"
        )
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
            # P0-3: Remove ticker callbacks before canceling subscriptions
            for key, ticker in list(self._subscriptions.items()):
                if key in self._ticker_callbacks:
                    try:
                        ticker.updateEvent -= self._ticker_callbacks[key]
                    except ValueError:
                        pass
            self._ticker_callbacks.clear()

            # Cancel all market data subscriptions
            for ticker in self._subscriptions.values():
                try:
                    self._ib.cancelMktData(ticker.contract)
                except Exception as e:
                    logger.warning(f"Error canceling market data: {e}")

            self._subscriptions.clear()
            self._contracts.clear()

            # P0-9: Disconnect with timeout to prevent hanging
            # Socket disconnect can hang if the connection is in a bad state
            try:
                # Use a timeout for the disconnect operation
                disconnect_timeout = 5.0  # 5 seconds should be enough
                # P0-9 FIX: Use get_running_loop() instead of deprecated get_event_loop()
                await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None, self._ib.disconnect
                    ),
                    timeout=disconnect_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Disconnect timed out after {disconnect_timeout}s, "
                    f"forcing connection state to disconnected"
                )
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

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
                # Broad catch for resilience - reconnection should not crash
                # the system regardless of what error IB/network throws
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
        # P0-5: Use lock to prevent race condition during reconnection
        async with self._subscriptions_lock:
            # Store subscription keys before clearing
            subscription_keys = list(self._subscriptions.keys())

            # Clear old callbacks first
            for key in subscription_keys:
                if key in self._ticker_callbacks:
                    ticker = self._subscriptions.get(key)
                    if ticker:
                        try:
                            ticker.updateEvent -= self._ticker_callbacks[key]
                        except ValueError:
                            pass
                    del self._ticker_callbacks[key]

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
            # Broad catch - reconciliation is a best-effort operation that
            # should not prevent system operation even if it fails
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

        # P0-5: Check under lock to prevent race condition
        async with self._subscriptions_lock:
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

            # P0-1: Check rate limits before making API request
            rate_key = f"mktdata:{subscription_key}"
            can_request, reason = self._rate_limiter.can_make_request(rate_key)
            if not can_request:
                logger.warning(f"Rate limit hit for {subscription_key}: {reason}")
                # Wait for rate limit to clear
                wait_time = self._rate_limiter.get_wait_time(rate_key)
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time:.1f}s for rate limit...")
                    await asyncio.sleep(wait_time)

            # Request market data
            ticker = self._ib.reqMktData(
                contract,
                genericTickList="",  # Use default tick types
                snapshot=False,  # Stream continuously
                regulatorySnapshot=False,
            )
            # P0-1: Record the request for rate limiting
            self._rate_limiter.record_request(rate_key)

            # P0-3: Register update handler with stored callback reference (prevents memory leak)
            # Using functools.partial instead of lambda to avoid closure capture issues
            callback = partial(self._on_ticker_update, subscription_key=subscription_key)

            # P0-5: Update subscriptions under lock
            async with self._subscriptions_lock:
                self._ticker_callbacks[subscription_key] = callback
                ticker.updateEvent += callback
                self._subscriptions[subscription_key] = ticker

            logger.info(f"Subscribed to market data: {subscription_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False

    async def unsubscribe_market_data(self, symbol: str) -> None:
        """Unsubscribe from market data for a symbol."""
        # P0-5: Use lock to prevent race condition
        async with self._subscriptions_lock:
            keys_to_remove = [k for k in self._subscriptions if k.startswith(f"{symbol}:")]

            for key in keys_to_remove:
                ticker = self._subscriptions.get(key)
                if ticker:
                    # P0-3: Remove callback before canceling to prevent memory leak
                    if key in self._ticker_callbacks:
                        try:
                            ticker.updateEvent -= self._ticker_callbacks[key]
                        except ValueError:
                            pass  # Callback already removed
                        del self._ticker_callbacks[key]

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

    async def place_order(
        self,
        order_event: OrderEvent,
        idempotency_key: str | None = None,
    ) -> int | None:
        """
        Place an order with Interactive Brokers.

        Args:
            order_event: The order to place
            idempotency_key: Optional unique key for idempotency (ERR-001).
                             If provided and the same key was used before, returns
                             the existing broker_order_id instead of placing a new order.
                             This prevents duplicate orders on retry after timeout.

        Returns:
            Broker order ID if successful, None otherwise
        """
        if not self.is_connected:
            logger.error("Cannot place order: not connected to IB")
            return None

        # ERR-001: Check if this order was already processed (idempotency)
        if idempotency_key is not None:
            if idempotency_key in self._processed_orders:
                existing_order_id = self._processed_orders[idempotency_key]
                logger.info(
                    f"Idempotent order detected (key={idempotency_key}): "
                    f"returning existing order_id={existing_order_id}"
                )
                return existing_order_id

        async def _execute_order() -> int | None:
            """Inner function for circuit breaker wrapping."""
            # Detect asset class from symbol
            symbol = order_event.symbol.upper()

            # Forex symbols
            FOREX_SYMBOLS = {
                "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK", "DKK",
                "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
            }
            # Futures symbols (E-mini, Micro, Energy, Metals, Bonds, Agriculture)
            FUTURES_SYMBOLS = {
                "ES", "NQ", "YM", "RTY",  # E-mini
                "MES", "MNQ", "MYM", "M2K",  # Micro index
                "CL", "MCL", "NG", "RB", "HO",  # Energy
                "GC", "MGC", "SI", "SIL", "PL", "HG",  # Metals
                "ZC", "ZW", "ZS", "ZM", "ZL",  # Agriculture
                "ZB", "ZN", "ZF", "ZT",  # Bonds
            }

            # Determine sec_type and exchange
            if symbol in FOREX_SYMBOLS:
                sec_type = "CASH"
                exchange = "IDEALPRO"
            elif symbol in FUTURES_SYMBOLS:
                sec_type = "FUT"
                # Use cached contract from market data subscription if available
                cached_key = f"{symbol}:CME:USD"
                if cached_key not in self._contracts:
                    cached_key = f"{symbol}:CBOT:USD"
                if cached_key not in self._contracts:
                    cached_key = f"{symbol}:NYMEX:USD"
                if cached_key not in self._contracts:
                    cached_key = f"{symbol}:COMEX:USD"

                if cached_key in self._contracts:
                    # Use the cached front month contract
                    contract = self._contracts[cached_key]
                else:
                    # Try to get front month
                    contract = await self._get_front_month_future(symbol, "CME", "USD")
                    if contract is None:
                        logger.error(f"Failed to get front month future for {symbol}")
                        return None

                # Skip normal qualification for futures - already qualified
                # Jump to order creation
                action = "BUY" if order_event.side == OrderSide.BUY else "SELL"

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
                        action, order_event.quantity, order_event.limit_price, order_event.stop_price,
                    )
                else:
                    logger.error(f"Unsupported order type: {order_event.order_type}")
                    return None

                # Time in force
                from core.events import TimeInForce
                tif_map = {
                    TimeInForce.DAY: "DAY", TimeInForce.GTC: "GTC",
                    TimeInForce.IOC: "IOC", TimeInForce.FOK: "FOK",
                }
                tif = getattr(order_event, 'time_in_force', TimeInForce.DAY)
                ib_order.tif = tif_map.get(tif, "DAY")

                # Place order
                try:
                    trade = self._ib.placeOrder(contract, ib_order)
                    broker_order_id = trade.order.orderId
                    logger.info(f"Order placed: {action} {order_event.quantity} {symbol} @ {order_event.order_type.value} (order_id={broker_order_id})")
                    self._active_orders[broker_order_id] = trade
                    return broker_order_id
                except Exception as e:
                    logger.error(f"Failed to place futures order: {e}")
                    return None
            else:
                sec_type = "STK"
                exchange = "SMART"

            # Create contract for stocks/ETFs and forex
            contract = self._create_contract(
                order_event.symbol,
                exchange=exchange,
                currency="USD",
                sec_type=sec_type,
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
            # Use getattr with default to handle missing time_in_force attribute
            tif = getattr(order_event, 'time_in_force', TimeInForce.DAY)
            ib_order.tif = tif_map.get(tif, "DAY")

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

        try:
            # Wrap order execution in circuit breaker (#S6)
            broker_order_id = await self._circuit_breaker.call(_execute_order)

            # ERR-001: Store idempotency mapping after successful placement
            if broker_order_id is not None and idempotency_key is not None:
                self._processed_orders[idempotency_key] = broker_order_id
                logger.debug(
                    f"Stored idempotency mapping: {idempotency_key} -> {broker_order_id}"
                )

            return broker_order_id
        except CircuitOpenError:
            logger.error("Cannot place order: circuit breaker is open")
            return None
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

    def get_order_by_idempotency_key(self, idempotency_key: str) -> int | None:
        """
        Get broker order ID by idempotency key (ERR-001).

        Args:
            idempotency_key: The idempotency key used when placing the order

        Returns:
            Broker order ID if found, None otherwise
        """
        return self._processed_orders.get(idempotency_key)

    def clear_idempotency_cache(self, max_entries: int = 10000) -> int:
        """
        Clear old entries from idempotency cache to prevent memory leak (ERR-001).

        This should be called periodically (e.g., daily) to clean up old entries.
        Orders older than the retention period are removed.

        Args:
            max_entries: Maximum number of entries to keep (default: 10000).
                         Oldest entries are removed first when limit exceeded.

        Returns:
            Number of entries removed
        """
        if len(self._processed_orders) <= max_entries:
            return 0

        # Remove oldest entries (simple FIFO based on insertion order in Python 3.7+)
        entries_to_remove = len(self._processed_orders) - max_entries
        keys_to_remove = list(self._processed_orders.keys())[:entries_to_remove]

        for key in keys_to_remove:
            del self._processed_orders[key]

        logger.info(
            f"Cleared {entries_to_remove} old entries from idempotency cache "
            f"(remaining: {len(self._processed_orders)})"
        )
        return entries_to_remove

    async def modify_order(
        self,
        broker_order_id: int,
        new_quantity: int | None = None,
        new_limit_price: float | None = None,
        new_stop_price: float | None = None,
    ) -> bool:
        """
        Modify an existing order (P0-7).

        IB supports order modification without cancel/replace for most order types.
        This is more efficient and avoids losing queue priority.

        Args:
            broker_order_id: The broker's order ID to modify
            new_quantity: New order quantity (optional)
            new_limit_price: New limit price (optional)
            new_stop_price: New stop price (optional)

        Returns:
            True if modification was submitted successfully

        Note:
            Not all order fields can be modified. If modification fails,
            consider using cancel_order + place_order instead.
        """
        if not self.is_connected:
            logger.error("Cannot modify order: not connected to IB")
            return False

        try:
            # Find the trade
            target_trade = None
            for trade in self._ib.trades():
                if trade.order.orderId == broker_order_id:
                    target_trade = trade
                    break

            if not target_trade:
                logger.warning(f"Order not found for modification: {broker_order_id}")
                return False

            # Check if order is modifiable
            status = target_trade.orderStatus.status
            if status in ("Filled", "Cancelled", "Inactive"):
                logger.warning(f"Cannot modify order {broker_order_id}: status is {status}")
                return False

            # Get the order and apply modifications
            order = target_trade.order
            modified = False

            if new_quantity is not None and new_quantity != order.totalQuantity:
                # Ensure new quantity is >= filled quantity
                filled = target_trade.orderStatus.filled
                if new_quantity < filled:
                    logger.error(
                        f"Cannot reduce quantity below filled amount "
                        f"({new_quantity} < {filled})"
                    )
                    return False
                order.totalQuantity = new_quantity
                modified = True
                logger.info(f"Order {broker_order_id}: quantity -> {new_quantity}")

            if new_limit_price is not None and new_limit_price != order.lmtPrice:
                if order.orderType not in ("LMT", "STP LMT"):
                    logger.warning(
                        f"Cannot set limit price on {order.orderType} order"
                    )
                else:
                    order.lmtPrice = new_limit_price
                    modified = True
                    logger.info(f"Order {broker_order_id}: limit price -> {new_limit_price}")

            if new_stop_price is not None and new_stop_price != order.auxPrice:
                if order.orderType not in ("STP", "STP LMT"):
                    logger.warning(
                        f"Cannot set stop price on {order.orderType} order"
                    )
                else:
                    order.auxPrice = new_stop_price
                    modified = True
                    logger.info(f"Order {broker_order_id}: stop price -> {new_stop_price}")

            if not modified:
                logger.info(f"No modifications made to order {broker_order_id}")
                return True

            # Submit the modification
            self._ib.placeOrder(target_trade.contract, order)
            logger.info(f"Order modification submitted: {broker_order_id}")

            # Update tracking
            if broker_order_id in self._order_tracking:
                tracking = self._order_tracking[broker_order_id]
                if new_quantity is not None:
                    tracking.quantity = new_quantity
                tracking.last_update = datetime.now(timezone.utc)

            return True

        except Exception as e:
            logger.error(f"Failed to modify order {broker_order_id}: {e}")
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

        Note:
            P0-6: IB enforces strict rate limits on historical data requests:
            - Max 6 concurrent requests
            - Pacing violations can result in 24h bans
            - Same contract/bar-size/duration requests require 15s spacing
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

            # P0-6: Check rate limits before requesting historical data
            # IB is very strict about historical data pacing - violations cause 24h bans
            rate_key = f"historical:{symbol}:{duration}:{bar_size}"
            can_request, reason = self._rate_limiter.can_make_request(rate_key)
            if not can_request:
                logger.warning(f"Historical data rate limit for {symbol}: {reason}")
                wait_time = self._rate_limiter.get_wait_time(rate_key)
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time:.1f}s for historical data rate limit...")
                    await asyncio.sleep(wait_time)

            # Record the request before making it
            self._rate_limiter.record_request(rate_key)

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

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract = None) -> None:
        """
        Handle error event from IB.

        P0-4: Comprehensive error code handling for all critical IB states.
        Reference: https://interactivebrokers.github.io/tws-api/message_codes.html
        """
        # Connection state errors (critical - affect trading)
        if errorCode == 1100:
            # Connectivity between IB and TWS has been lost
            logger.critical(f"IB CONNECTIVITY LOST: {errorString}")
            self._connection_state = ConnectionState.ERROR
            # Trigger circuit breaker
            self._circuit_breaker.record_failure(ConnectionError(errorString))

        elif errorCode == 1101:
            # Connectivity restored, data lost - need to resubscribe
            logger.warning(f"IB connectivity restored (data lost): {errorString}")
            self._connection_state = ConnectionState.CONNECTED
            # Schedule resubscription
            asyncio.create_task(self._resubscribe_market_data())

        elif errorCode == 1102:
            # Connectivity restored, data maintained
            logger.info(f"IB connectivity restored (data maintained): {errorString}")
            self._connection_state = ConnectionState.CONNECTED
            self._circuit_breaker.record_success()

        # Market data farm status
        elif errorCode in (2104, 2106, 2158):  # Market data farm connected/disconnected
            logger.debug(f"IB info [{errorCode}]: {errorString}")

        elif errorCode == 2103:  # Market data farm connection broken
            logger.warning(f"IB market data connection issue: {errorString}")

        elif errorCode == 2107:
            # Historical data farm connected
            logger.info(f"Historical data farm connected: {errorString}")

        elif errorCode == 2110:
            # Connectivity between TWS and server is broken
            logger.critical(f"TWS-server connectivity broken: {errorString}")
            self._connection_state = ConnectionState.ERROR

        # Historical data errors
        elif errorCode == 162:
            # Historical data pacing violation (too many requests)
            # P0-4/P0-10 FIX: Must call record_pacing_violation() to trigger backoff
            backoff = self._rate_limiter.record_pacing_violation()
            logger.error(
                f"Historical data pacing violation: {errorString}. "
                f"IB may impose a 24h ban. Backing off for {backoff:.0f}s"
            )

        elif errorCode == 366:
            # No historical data query found for ticker
            logger.warning(f"No historical data available: {errorString}")

        # Order-related errors
        elif errorCode == 201:
            # Order rejected
            logger.error(f"Order rejected [{reqId}]: {errorString}")

        elif errorCode == 202:
            # Order cancelled
            logger.info(f"Order cancelled [{reqId}]: {errorString}")

        elif errorCode == 203:
            # Security not available for trading
            logger.error(f"Security not tradeable [{reqId}]: {errorString}")

        elif errorCode == 10197:  # No market data during competing session
            logger.warning(f"Market data unavailable: {errorString}")

        # Additional critical error codes (identified by IB Configuration Expert)
        elif errorCode == 103:
            # Duplicate order ID
            logger.error(f"Duplicate order ID [{reqId}]: {errorString}")

        elif errorCode == 104:
            # Can't modify a filled order
            logger.error(f"Cannot modify filled order [{reqId}]: {errorString}")

        elif errorCode == 110:
            # Price out of range
            logger.error(f"Price out of range [{reqId}]: {errorString}")

        elif errorCode == 135:
            # Can't find order with this ID
            logger.warning(f"Order not found [{reqId}]: {errorString}")

        elif errorCode == 200:
            # No security definition found - very common
            logger.error(f"Security not found [{reqId}]: {errorString}")

        elif errorCode == 354:
            # Requested market data not subscribed
            logger.warning(
                f"Market data not subscribed [{reqId}]: {errorString}. "
                f"Check your IB market data subscriptions."
            )

        elif errorCode == 2105:
            # HMDS data farm connection broken
            logger.error(f"HMDS data farm connection broken: {errorString}")

        # Client ID conflicts
        elif errorCode == 326:
            # Unable to connect - client ID already in use
            logger.critical(
                f"Client ID {self._config.client_id} already in use: {errorString}. "
                f"Another application may be connected with the same client ID."
            )
            self._connection_state = ConnectionState.ERROR

        elif errorCode == 502:
            # Couldn't connect to TWS
            logger.error(f"Could not connect to TWS: {errorString}")
            self._connection_state = ConnectionState.ERROR

        elif errorCode == 504:
            # Not connected
            logger.warning(f"Not connected to IB: {errorString}")

        elif errorCode == 1300:
            # Socket dropped during operation
            logger.error(f"Socket dropped: {errorString}")
            self._connection_state = ConnectionState.DISCONNECTED

        # Default handler for unrecognized errors
        else:
            if errorCode >= 2000 and errorCode < 3000:
                # 2000 series are warnings
                logger.warning(f"IB warning [{errorCode}] reqId={reqId}: {errorString}")
            elif errorCode >= 1000 and errorCode < 2000:
                # 1000 series are system messages
                logger.info(f"IB system [{errorCode}] reqId={reqId}: {errorString}")
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

            # P0-11: Data quality validation - check for obviously bad data
            bid = self._safe_float(ticker.bid)
            ask = self._safe_float(ticker.ask)
            last = self._safe_float(ticker.last)

            # Validate bid/ask relationship (bid should be <= ask)
            if bid > 0 and ask > 0 and bid > ask:
                logger.warning(
                    f"Data quality issue for {symbol}: bid ({bid}) > ask ({ask}). "
                    f"Possible crossed market or bad data."
                )
                # Track quality issues
                self._record_data_quality_issue(symbol, "crossed_market")

            # Validate prices are reasonable (not 0 or extreme)
            if last > 0:
                # Check for extreme spreads (> 10% of last price)
                if bid > 0 and ask > 0:
                    spread_pct = (ask - bid) / last * 100
                    if spread_pct > 10:
                        logger.warning(
                            f"Data quality issue for {symbol}: spread ({spread_pct:.1f}%) "
                            f"exceeds 10% threshold"
                        )
                        self._record_data_quality_issue(symbol, "wide_spread")

            # Create market data event (with safe conversions for NaN values)
            event = MarketDataEvent(
                source_agent="broker",
                symbol=symbol,
                exchange=ticker.contract.exchange if ticker.contract else "SMART",
                bid=bid if bid > 0 else 0.0,
                ask=ask if ask > 0 else 0.0,
                last=last if last > 0 else 0.0,
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

    def _record_data_quality_issue(self, symbol: str, issue_type: str) -> None:
        """
        Track data quality issues for circuit breaker logic (P0-11).

        Args:
            symbol: The symbol with the issue
            issue_type: Type of issue (crossed_market, wide_spread, etc.)
        """
        if not hasattr(self, '_data_quality_issues'):
            self._data_quality_issues: dict[str, list[tuple[datetime, str]]] = {}

        if symbol not in self._data_quality_issues:
            self._data_quality_issues[symbol] = []

        # Add issue with timestamp
        self._data_quality_issues[symbol].append((datetime.now(timezone.utc), issue_type))

        # Keep only last 100 issues per symbol
        if len(self._data_quality_issues[symbol]) > 100:
            self._data_quality_issues[symbol] = self._data_quality_issues[symbol][-100:]

        # Check if we should trigger a circuit breaker
        recent_issues = [
            i for i in self._data_quality_issues[symbol]
            if (datetime.now(timezone.utc) - i[0]).total_seconds() < 60
        ]

        if len(recent_issues) >= 10:
            logger.error(
                f"Data quality circuit breaker triggered for {symbol}: "
                f"{len(recent_issues)} issues in last 60 seconds"
            )
            # Could trigger reconnection or alert here

    def get_data_quality_stats(self) -> dict[str, Any]:
        """Get data quality statistics (P0-11)."""
        if not hasattr(self, '_data_quality_issues'):
            return {"symbols": {}, "total_issues": 0}

        stats = {"symbols": {}, "total_issues": 0}
        now = datetime.now(timezone.utc)

        for symbol, issues in self._data_quality_issues.items():
            recent = [i for i in issues if (now - i[0]).total_seconds() < 300]  # Last 5 min
            stats["symbols"][symbol] = {
                "total_issues": len(issues),
                "recent_issues": len(recent),
                "issue_types": list(set(i[1] for i in recent)),
            }
            stats["total_issues"] += len(recent)

        return stats

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

    # ========== P2: Order Status Polling Fallback ==========

    async def _start_order_polling(self, order_id: int) -> None:
        """
        Start polling for order status as a fallback.

        Used when IB events may be missed (e.g., after reconnection).

        Args:
            order_id: The broker order ID to poll
        """
        if not self._config.order_polling_enabled:
            return

        if order_id in self._orders_being_polled:
            return  # Already polling this order

        self._orders_being_polled[order_id] = datetime.now(timezone.utc)
        logger.debug(f"Started polling for order {order_id}")

    async def _poll_order_status(self, order_id: int) -> dict | None:
        """
        Poll IB for order status.

        Args:
            order_id: The broker order ID to query

        Returns:
            Order status dict or None if not found
        """
        if not self.is_connected:
            return None

        try:
            # Find the trade in IB's open trades
            for trade in self._ib.openTrades():
                if trade.order.orderId == order_id:
                    return {
                        "order_id": order_id,
                        "status": trade.orderStatus.status,
                        "filled": int(trade.orderStatus.filled),
                        "remaining": int(trade.orderStatus.remaining),
                        "avg_fill_price": trade.orderStatus.avgFillPrice,
                    }

            # Check completed trades (recent fills)
            for trade in self._ib.trades():
                if trade.order.orderId == order_id:
                    return {
                        "order_id": order_id,
                        "status": trade.orderStatus.status,
                        "filled": int(trade.orderStatus.filled),
                        "remaining": int(trade.orderStatus.remaining),
                        "avg_fill_price": trade.orderStatus.avgFillPrice,
                    }

            return None

        except Exception as e:
            logger.error(f"Error polling order {order_id}: {e}")
            return None

    async def poll_pending_orders(self) -> dict[int, dict]:
        """
        Poll all pending orders and update tracking.

        Returns:
            Dict mapping order_id to current status
        """
        results = {}
        now = datetime.now(timezone.utc)
        orders_to_remove = []

        for order_id, start_time in list(self._orders_being_polled.items()):
            # Check timeout
            elapsed = (now - start_time).total_seconds()
            if elapsed > self._config.order_polling_timeout_seconds:
                logger.warning(f"Order {order_id} polling timeout after {elapsed:.0f}s")
                orders_to_remove.append(order_id)
                continue

            status = await self._poll_order_status(order_id)
            if status:
                results[order_id] = status

                # Update local tracking
                if order_id in self._order_tracking:
                    tracking = self._order_tracking[order_id]
                    tracking.status = status["status"]
                    tracking.filled_quantity = status["filled"]
                    tracking.avg_fill_price = status["avg_fill_price"]
                    tracking.last_update = now

                # Stop polling if order is complete
                if status["status"] in ("Filled", "Cancelled", "Inactive"):
                    logger.info(f"Order {order_id} completed: {status['status']}")
                    orders_to_remove.append(order_id)

        # Clean up completed orders
        for order_id in orders_to_remove:
            self._orders_being_polled.pop(order_id, None)

        return results

    async def _run_order_polling_loop(self) -> None:
        """Background task to poll pending orders."""
        while self.is_connected and self._orders_being_polled:
            try:
                await self.poll_pending_orders()
                await asyncio.sleep(self._config.order_polling_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Order polling loop error: {e}")
                await asyncio.sleep(self._config.order_polling_interval_seconds)

    def get_polling_status(self) -> dict:
        """Get status of order polling."""
        return {
            "enabled": self._config.order_polling_enabled,
            "orders_being_polled": len(self._orders_being_polled),
            "order_ids": list(self._orders_being_polled.keys()),
            "polling_interval_seconds": self._config.order_polling_interval_seconds,
        }

    # ========== P2: Connection Quality Metrics ==========

    async def measure_latency(self) -> float:
        """
        Measure round-trip latency to IB server.

        Returns:
            Latency in milliseconds
        """
        if not self.is_connected:
            return -1.0

        try:
            start = datetime.now(timezone.utc)
            await asyncio.wait_for(
                self._ib.reqCurrentTimeAsync(),
                timeout=5.0
            )
            end = datetime.now(timezone.utc)

            latency_ms = (end - start).total_seconds() * 1000
            self._connection_quality.record_latency(latency_ms)
            self._connection_quality.message_count += 1

            return latency_ms

        except Exception as e:
            logger.error(f"Latency measurement failed: {e}")
            self._connection_quality.error_count += 1
            return -1.0

    def get_connection_quality(self) -> dict:
        """
        Get connection quality metrics.

        Returns:
            Dict with connection quality statistics
        """
        return {
            "is_connected": self.is_connected,
            "metrics": self._connection_quality.to_dict(),
            "rate_limiter": self._rate_limiter.get_stats(),
            "circuit_breaker": self._circuit_breaker.get_stats().to_dict(),
        }

    def record_connection_event(self, event_type: str) -> None:
        """
        Record a connection-related event for quality tracking.

        Args:
            event_type: Type of event ("connect", "disconnect", "error", etc.)
        """
        now = datetime.now(timezone.utc)

        if event_type == "connect":
            if self._connection_quality.connection_start_time is None:
                self._connection_quality.connection_start_time = now
            self._connection_quality.last_reconnect_time = now
            self._connection_quality.reconnect_count += 1

        elif event_type == "disconnect":
            self._connection_quality.last_disconnect_time = now
            self._connection_quality.total_disconnects += 1
            # Update uptime
            if self._connection_quality.connection_start_time:
                session_time = (now - self._connection_quality.connection_start_time).total_seconds()
                self._connection_quality.connection_uptime_seconds += session_time

        elif event_type == "error":
            self._connection_quality.error_count += 1

        elif event_type == "pacing_violation":
            self._connection_quality.pacing_violations += 1

    # ========== P2: Session Recovery ==========

    def save_session_state(self) -> None:
        """
        Save current session state for recovery.

        Called periodically to enable session recovery after disconnect.
        """
        if not self._config.session_recovery_enabled:
            return

        now = datetime.now(timezone.utc)

        # Save subscriptions
        self._session_state.subscribed_symbols = list(self._subscriptions.keys())

        # Save pending orders
        self._session_state.pending_orders = {
            oid: {
                "symbol": status.symbol,
                "side": status.side.value,
                "quantity": status.quantity,
                "filled": status.filled_quantity,
                "status": status.status,
            }
            for oid, status in self._order_tracking.items()
            if status.status not in ("Filled", "Cancelled", "Inactive")
        }

        # Save positions from last portfolio state
        if self._last_portfolio_state:
            self._session_state.last_known_positions = {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "market_value": pos.market_value,
                }
                for symbol, pos in self._last_portfolio_state.positions.items()
            }

        self._session_state.last_save_time = now
        logger.debug(f"Session state saved: {len(self._session_state.subscribed_symbols)} subscriptions, "
                     f"{len(self._session_state.pending_orders)} pending orders")

    async def recover_session(self) -> dict:
        """
        Recover session state after reconnection.

        Returns:
            Dict with recovery results
        """
        if not self._config.session_recovery_enabled:
            return {"enabled": False}

        results = {
            "enabled": True,
            "subscriptions_recovered": 0,
            "subscriptions_failed": 0,
            "orders_reconciled": 0,
            "orders_missing": 0,
        }

        if not self.is_connected:
            logger.warning("Cannot recover session: not connected")
            return results

        logger.info("Starting session recovery...")

        # Recover market data subscriptions
        for sub_key in self._session_state.subscribed_symbols:
            parts = sub_key.split(":")
            if len(parts) >= 3:
                symbol, exchange, currency = parts[0], parts[1], parts[2]
                try:
                    success = await self.subscribe_market_data(symbol, exchange, currency)
                    if success:
                        results["subscriptions_recovered"] += 1
                    else:
                        results["subscriptions_failed"] += 1
                except Exception as e:
                    logger.error(f"Failed to recover subscription {sub_key}: {e}")
                    results["subscriptions_failed"] += 1

        # Reconcile orders
        await self._reconcile_orders_on_reconnect()
        results["orders_reconciled"] = len(self._session_state.pending_orders)

        # Start polling for any pending orders
        for order_id in self._session_state.pending_orders.keys():
            await self._start_order_polling(order_id)
            results["orders_missing"] = len(self._orders_being_polled)

        logger.info(
            f"Session recovery complete: {results['subscriptions_recovered']} subscriptions, "
            f"{results['orders_reconciled']} orders reconciled"
        )

        return results

    def get_session_state(self) -> dict:
        """Get current session state for debugging."""
        return self._session_state.to_dict()

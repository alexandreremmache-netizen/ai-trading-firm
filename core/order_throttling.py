"""
Order Throttling Module
=======================

Per-venue order throttling to prevent excessive order submission (Issue #E25).

Features:
- Per-venue rate limits
- Sliding window throttling
- Adaptive throttling based on rejections
- Burst allowance with recovery
- Exchange-specific configurations
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class ThrottleAction(str, Enum):
    """Throttle decision."""
    ALLOW = "allow"
    DELAY = "delay"
    REJECT = "reject"


@dataclass
class VenueConfig:
    """Rate limit configuration for a venue."""
    venue_id: str
    max_orders_per_second: float = 10.0
    max_orders_per_minute: float = 300.0
    max_orders_per_hour: float = 5000.0
    burst_limit: int = 20  # Max orders in burst
    burst_recovery_seconds: float = 1.0  # Time to recover burst capacity
    rejection_backoff_factor: float = 2.0  # Multiply limits by this on rejection
    min_order_interval_ms: float = 50.0  # Minimum time between orders


@dataclass
class ThrottleState:
    """Current throttle state for a venue."""
    venue_id: str
    orders_last_second: deque = field(default_factory=lambda: deque(maxlen=1000))
    orders_last_minute: deque = field(default_factory=lambda: deque(maxlen=10000))
    orders_last_hour: deque = field(default_factory=lambda: deque(maxlen=100000))
    last_order_time: float = 0.0
    burst_tokens: float = 0.0
    last_burst_refill: float = 0.0
    consecutive_rejections: int = 0
    last_rejection_time: float = 0.0
    total_orders: int = 0
    total_throttled: int = 0
    total_delayed: int = 0

    def __post_init__(self):
        self.burst_tokens = 20.0  # Start with full burst capacity
        self.last_burst_refill = time.time()


@dataclass
class ThrottleResult:
    """Result of throttle check."""
    action: ThrottleAction
    venue_id: str
    delay_ms: float = 0.0
    reason: str = ""
    current_rate_per_second: float = 0.0
    limit_per_second: float = 0.0


class OrderThrottler:
    """
    Per-venue order throttling implementation (#E25).

    Prevents excessive order submission to protect against:
    - Exchange rate limit violations
    - Runaway algorithms
    - Unintended order spam
    """

    # Default configurations for common venues
    DEFAULT_CONFIGS = {
        "SMART": VenueConfig(
            venue_id="SMART",
            max_orders_per_second=10.0,
            max_orders_per_minute=200.0,
            burst_limit=15,
        ),
        "NYSE": VenueConfig(
            venue_id="NYSE",
            max_orders_per_second=5.0,
            max_orders_per_minute=100.0,
            burst_limit=10,
        ),
        "NASDAQ": VenueConfig(
            venue_id="NASDAQ",
            max_orders_per_second=8.0,
            max_orders_per_minute=150.0,
            burst_limit=12,
        ),
        "ARCA": VenueConfig(
            venue_id="ARCA",
            max_orders_per_second=10.0,
            max_orders_per_minute=200.0,
            burst_limit=15,
        ),
        "BATS": VenueConfig(
            venue_id="BATS",
            max_orders_per_second=10.0,
            max_orders_per_minute=200.0,
            burst_limit=15,
        ),
        "IEX": VenueConfig(
            venue_id="IEX",
            max_orders_per_second=5.0,
            max_orders_per_minute=100.0,
            burst_limit=8,
        ),
        "CME": VenueConfig(
            venue_id="CME",
            max_orders_per_second=20.0,
            max_orders_per_minute=500.0,
            burst_limit=30,
        ),
        "GLOBEX": VenueConfig(
            venue_id="GLOBEX",
            max_orders_per_second=20.0,
            max_orders_per_minute=500.0,
            burst_limit=30,
        ),
        "DEFAULT": VenueConfig(
            venue_id="DEFAULT",
            max_orders_per_second=5.0,
            max_orders_per_minute=100.0,
            burst_limit=10,
        ),
    }

    def __init__(
        self,
        custom_configs: dict[str, VenueConfig] | None = None,
        global_max_per_second: float = 50.0,
        adaptive_throttling: bool = True,
    ):
        self._configs: dict[str, VenueConfig] = {**self.DEFAULT_CONFIGS}
        if custom_configs:
            self._configs.update(custom_configs)

        self._states: dict[str, ThrottleState] = {}
        self._global_orders: deque = deque(maxlen=1000)
        self._global_max_per_second = global_max_per_second
        self._adaptive_throttling = adaptive_throttling
        self._lock = Lock()

    def get_config(self, venue_id: str) -> VenueConfig:
        """Get configuration for venue."""
        return self._configs.get(venue_id, self._configs["DEFAULT"])

    def get_state(self, venue_id: str) -> ThrottleState:
        """Get or create state for venue."""
        if venue_id not in self._states:
            self._states[venue_id] = ThrottleState(venue_id=venue_id)
        return self._states[venue_id]

    def check_throttle(self, venue_id: str) -> ThrottleResult:
        """
        Check if order should be throttled for a venue.

        Returns ThrottleResult with action (ALLOW, DELAY, or REJECT).
        """
        with self._lock:
            config = self.get_config(venue_id)
            state = self.get_state(venue_id)
            now = time.time()

            # Clean old entries
            self._clean_old_entries(state, now)

            # Check global limit first
            global_check = self._check_global_limit(now)
            if global_check.action != ThrottleAction.ALLOW:
                return global_check

            # Apply adaptive backoff if there were recent rejections
            effective_limits = self._get_effective_limits(config, state)

            # Check minimum interval
            if state.last_order_time > 0:
                elapsed_ms = (now - state.last_order_time) * 1000
                if elapsed_ms < config.min_order_interval_ms:
                    delay_needed = config.min_order_interval_ms - elapsed_ms
                    state.total_delayed += 1
                    return ThrottleResult(
                        action=ThrottleAction.DELAY,
                        venue_id=venue_id,
                        delay_ms=delay_needed,
                        reason=f"Min interval not met ({elapsed_ms:.0f}ms < {config.min_order_interval_ms}ms)",
                        current_rate_per_second=len(state.orders_last_second),
                        limit_per_second=effective_limits['per_second'],
                    )

            # Check per-second limit
            orders_in_second = len([t for t in state.orders_last_second if now - t < 1.0])
            if orders_in_second >= effective_limits['per_second']:
                state.total_throttled += 1
                return ThrottleResult(
                    action=ThrottleAction.REJECT,
                    venue_id=venue_id,
                    reason=f"Per-second limit exceeded ({orders_in_second} >= {effective_limits['per_second']:.0f})",
                    current_rate_per_second=orders_in_second,
                    limit_per_second=effective_limits['per_second'],
                )

            # Check per-minute limit
            orders_in_minute = len([t for t in state.orders_last_minute if now - t < 60.0])
            if orders_in_minute >= effective_limits['per_minute']:
                state.total_throttled += 1
                return ThrottleResult(
                    action=ThrottleAction.REJECT,
                    venue_id=venue_id,
                    reason=f"Per-minute limit exceeded ({orders_in_minute} >= {effective_limits['per_minute']:.0f})",
                    current_rate_per_second=orders_in_second,
                    limit_per_second=effective_limits['per_second'],
                )

            # Check burst using token bucket
            self._refill_burst_tokens(state, config, now)
            if state.burst_tokens < 1.0:
                # Can delay instead of reject for burst
                delay_needed = config.burst_recovery_seconds * 1000 / config.burst_limit
                state.total_delayed += 1
                return ThrottleResult(
                    action=ThrottleAction.DELAY,
                    venue_id=venue_id,
                    delay_ms=delay_needed,
                    reason=f"Burst limit reached (tokens={state.burst_tokens:.1f})",
                    current_rate_per_second=orders_in_second,
                    limit_per_second=effective_limits['per_second'],
                )

            # All checks passed
            return ThrottleResult(
                action=ThrottleAction.ALLOW,
                venue_id=venue_id,
                current_rate_per_second=orders_in_second,
                limit_per_second=effective_limits['per_second'],
            )

    def record_order(self, venue_id: str) -> None:
        """Record that an order was submitted."""
        with self._lock:
            state = self.get_state(venue_id)
            now = time.time()

            state.orders_last_second.append(now)
            state.orders_last_minute.append(now)
            state.orders_last_hour.append(now)
            state.last_order_time = now
            state.total_orders += 1

            # Consume burst token
            if state.burst_tokens >= 1.0:
                state.burst_tokens -= 1.0

            # Record global
            self._global_orders.append(now)

    def record_rejection(self, venue_id: str, reason: str = "") -> None:
        """Record that an order was rejected by the venue."""
        with self._lock:
            state = self.get_state(venue_id)
            state.consecutive_rejections += 1
            state.last_rejection_time = time.time()

            logger.warning(
                f"Order rejected by {venue_id}: {reason} "
                f"(consecutive rejections: {state.consecutive_rejections})"
            )

    def record_success(self, venue_id: str) -> None:
        """Record that an order was accepted by the venue."""
        with self._lock:
            state = self.get_state(venue_id)
            # Gradually recover from rejections
            if state.consecutive_rejections > 0:
                state.consecutive_rejections = max(0, state.consecutive_rejections - 1)

    def _clean_old_entries(self, state: ThrottleState, now: float) -> None:
        """Remove entries older than their window."""
        # Clean per-second (keep last 2 seconds for safety)
        while state.orders_last_second and now - state.orders_last_second[0] > 2.0:
            state.orders_last_second.popleft()

        # Clean per-minute (keep last 70 seconds)
        while state.orders_last_minute and now - state.orders_last_minute[0] > 70.0:
            state.orders_last_minute.popleft()

        # Clean per-hour (keep last 65 minutes)
        while state.orders_last_hour and now - state.orders_last_hour[0] > 3900.0:
            state.orders_last_hour.popleft()

    def _check_global_limit(self, now: float) -> ThrottleResult:
        """Check global rate limit across all venues."""
        # Clean old entries
        while self._global_orders and now - self._global_orders[0] > 2.0:
            self._global_orders.popleft()

        orders_in_second = len([t for t in self._global_orders if now - t < 1.0])

        if orders_in_second >= self._global_max_per_second:
            return ThrottleResult(
                action=ThrottleAction.REJECT,
                venue_id="GLOBAL",
                reason=f"Global limit exceeded ({orders_in_second} >= {self._global_max_per_second})",
                current_rate_per_second=orders_in_second,
                limit_per_second=self._global_max_per_second,
            )

        return ThrottleResult(action=ThrottleAction.ALLOW, venue_id="GLOBAL")

    def _get_effective_limits(
        self,
        config: VenueConfig,
        state: ThrottleState,
    ) -> dict[str, float]:
        """Get effective limits considering adaptive backoff."""
        limits = {
            'per_second': config.max_orders_per_second,
            'per_minute': config.max_orders_per_minute,
            'per_hour': config.max_orders_per_hour,
        }

        if not self._adaptive_throttling:
            return limits

        # Apply backoff based on consecutive rejections
        if state.consecutive_rejections > 0:
            backoff_factor = config.rejection_backoff_factor ** min(state.consecutive_rejections, 5)
            limits = {k: v / backoff_factor for k, v in limits.items()}

            logger.debug(
                f"Adaptive throttle for {state.venue_id}: "
                f"backoff_factor={backoff_factor:.1f}, "
                f"effective_per_second={limits['per_second']:.1f}"
            )

        return limits

    def _refill_burst_tokens(
        self,
        state: ThrottleState,
        config: VenueConfig,
        now: float,
    ) -> None:
        """Refill burst tokens based on time elapsed."""
        elapsed = now - state.last_burst_refill
        tokens_to_add = (elapsed / config.burst_recovery_seconds) * config.burst_limit

        state.burst_tokens = min(
            config.burst_limit,
            state.burst_tokens + tokens_to_add
        )
        state.last_burst_refill = now

    def get_venue_stats(self, venue_id: str) -> dict:
        """Get throttle statistics for a venue."""
        state = self.get_state(venue_id)
        config = self.get_config(venue_id)
        now = time.time()

        with self._lock:
            self._clean_old_entries(state, now)

            return {
                'venue_id': venue_id,
                'config': {
                    'max_per_second': config.max_orders_per_second,
                    'max_per_minute': config.max_orders_per_minute,
                    'burst_limit': config.burst_limit,
                },
                'current': {
                    'orders_last_second': len([t for t in state.orders_last_second if now - t < 1.0]),
                    'orders_last_minute': len([t for t in state.orders_last_minute if now - t < 60.0]),
                    'burst_tokens': state.burst_tokens,
                },
                'totals': {
                    'total_orders': state.total_orders,
                    'total_throttled': state.total_throttled,
                    'total_delayed': state.total_delayed,
                    'throttle_rate': state.total_throttled / state.total_orders if state.total_orders > 0 else 0,
                },
                'health': {
                    'consecutive_rejections': state.consecutive_rejections,
                    'seconds_since_rejection': now - state.last_rejection_time if state.last_rejection_time > 0 else None,
                },
            }

    def get_all_stats(self) -> dict[str, dict]:
        """Get statistics for all venues."""
        return {venue_id: self.get_venue_stats(venue_id) for venue_id in self._states}

    def reset_venue(self, venue_id: str) -> None:
        """Reset throttle state for a venue."""
        with self._lock:
            if venue_id in self._states:
                del self._states[venue_id]
                logger.info(f"Reset throttle state for {venue_id}")


class ThrottledOrderExecutor:
    """
    Wrapper that applies throttling before order execution.

    Use this to wrap your order submission logic.
    """

    def __init__(
        self,
        throttler: OrderThrottler,
        max_delay_ms: float = 5000.0,  # Max delay before rejecting
    ):
        self.throttler = throttler
        self.max_delay_ms = max_delay_ms

    async def execute_with_throttle(
        self,
        venue_id: str,
        order_func,
        *args,
        **kwargs,
    ):
        """
        Execute order function with throttling.

        Args:
            venue_id: Target venue
            order_func: Async function to execute order
            *args, **kwargs: Arguments for order_func

        Returns:
            Result of order_func

        Raises:
            ThrottleRejectedException if order is rejected
        """
        import asyncio

        total_delay = 0.0

        while True:
            result = self.throttler.check_throttle(venue_id)

            if result.action == ThrottleAction.ALLOW:
                # Record and execute
                self.throttler.record_order(venue_id)
                try:
                    order_result = await order_func(*args, **kwargs)
                    self.throttler.record_success(venue_id)
                    return order_result
                except Exception as e:
                    if "rate" in str(e).lower() or "limit" in str(e).lower():
                        self.throttler.record_rejection(venue_id, str(e))
                    raise

            elif result.action == ThrottleAction.DELAY:
                if total_delay + result.delay_ms > self.max_delay_ms:
                    raise ThrottleRejectedException(
                        f"Max delay exceeded for {venue_id}: {total_delay + result.delay_ms}ms"
                    )

                await asyncio.sleep(result.delay_ms / 1000.0)
                total_delay += result.delay_ms

            else:  # REJECT
                raise ThrottleRejectedException(
                    f"Order throttled for {venue_id}: {result.reason}"
                )


class ThrottleRejectedException(Exception):
    """Exception raised when order is rejected due to throttling."""
    pass

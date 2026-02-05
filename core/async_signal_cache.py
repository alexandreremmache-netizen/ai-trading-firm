"""
Async Signal Cache
==================

Out-of-band cache for heavy agent signals (HMM, LLM).
Heavy agents compute in background and update cache asynchronously.
CIO reads instantly from cache without blocking on computation.

Features:
- Staleness detection (30s threshold)
- Confidence decay for stale signals
- Thread-safe concurrent access
- Computation time tracking
- Per-agent freshness monitoring
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from core.events import SignalEvent, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class CachedSignal:
    """A cached signal with metadata."""
    signal: SignalEvent
    cached_at: datetime
    computation_time_ms: float
    agent_name: str
    symbol: str

    @property
    def age_seconds(self) -> float:
        """Get age of cached signal in seconds."""
        return (datetime.now(timezone.utc) - self.cached_at).total_seconds()

    def is_stale(self, max_age_seconds: float = 30.0) -> bool:
        """Check if signal is stale."""
        return self.age_seconds > max_age_seconds


@dataclass
class AsyncSignalCacheConfig:
    """Configuration for async signal cache."""
    max_staleness_seconds: float = 30.0  # Signals older than this are stale
    confidence_decay_per_second: float = 0.02  # 2% decay per second
    min_confidence_floor: float = 0.1  # Never decay below this
    max_history_per_agent: int = 100  # History entries per agent
    cleanup_interval_seconds: float = 60.0  # How often to clean old entries


@dataclass
class AgentCacheStats:
    """Statistics for an agent's cache usage."""
    total_updates: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    stale_reads: int = 0
    avg_computation_time_ms: float = 0.0
    last_update: datetime | None = None
    last_read: datetime | None = None


class AsyncSignalCache:
    """
    Async Signal Cache for out-of-band agent signals.

    Heavy agents (HMM regime detection, LLM sentiment) can update this
    cache asynchronously in background tasks. The CIO reads from cache
    instantly without waiting for these computations.

    Usage:
        cache = AsyncSignalCache()

        # Heavy agent updates cache in background
        async def background_hmm_task():
            while running:
                signal = compute_hmm_signal()
                await cache.update_signal("MacroAgent", "ES", signal, computation_time_ms)
                await asyncio.sleep(30.0)

        # CIO reads instantly
        cached = await cache.get_signal("MacroAgent", "ES")
        if cached:
            merged_signals["MacroAgent"] = cached
    """

    def __init__(self, config: AsyncSignalCacheConfig | None = None):
        self._config = config or AsyncSignalCacheConfig()

        # Primary cache: agent_name -> symbol -> CachedSignal
        self._cache: Dict[str, Dict[str, CachedSignal]] = {}

        # History for analysis
        self._history: Dict[str, deque[CachedSignal]] = {}

        # Per-agent statistics
        self._stats: Dict[str, AgentCacheStats] = {}

        # Thread safety
        self._lock = asyncio.Lock()

        # Agents registered for out-of-band processing
        self._registered_agents: set[str] = set()

        # Cleanup tracking
        self._last_cleanup = datetime.now(timezone.utc)

    async def update_signal(
        self,
        agent_name: str,
        symbol: str,
        signal: SignalEvent,
        computation_time_ms: float = 0.0,
    ) -> None:
        """
        Update cached signal for an agent/symbol.

        Called by heavy agents after completing background computation.

        Args:
            agent_name: Name of the agent
            symbol: Trading symbol
            signal: The computed signal
            computation_time_ms: Time taken to compute the signal
        """
        now = datetime.now(timezone.utc)

        cached = CachedSignal(
            signal=signal,
            cached_at=now,
            computation_time_ms=computation_time_ms,
            agent_name=agent_name,
            symbol=symbol,
        )

        async with self._lock:
            # Initialize agent structures if needed
            if agent_name not in self._cache:
                self._cache[agent_name] = {}
            if agent_name not in self._history:
                self._history[agent_name] = deque(maxlen=self._config.max_history_per_agent)
            if agent_name not in self._stats:
                self._stats[agent_name] = AgentCacheStats()

            # Update cache
            self._cache[agent_name][symbol] = cached

            # Add to history
            self._history[agent_name].append(cached)

            # Update stats
            stats = self._stats[agent_name]
            stats.total_updates += 1
            stats.last_update = now

            # Update average computation time (exponential moving average)
            if stats.avg_computation_time_ms == 0:
                stats.avg_computation_time_ms = computation_time_ms
            else:
                alpha = 0.2
                stats.avg_computation_time_ms = (
                    alpha * computation_time_ms +
                    (1 - alpha) * stats.avg_computation_time_ms
                )

            self._registered_agents.add(agent_name)

        logger.debug(
            f"AsyncCache: Updated {agent_name}/{symbol} "
            f"(computation: {computation_time_ms:.1f}ms, confidence: {signal.confidence:.2f})"
        )

    async def get_signal(
        self,
        agent_name: str,
        symbol: str,
        apply_staleness_decay: bool = True,
    ) -> SignalEvent | None:
        """
        Get cached signal for an agent/symbol.

        Returns signal with confidence decayed based on staleness.

        Args:
            agent_name: Name of the agent
            symbol: Trading symbol
            apply_staleness_decay: If True, decay confidence for stale signals

        Returns:
            SignalEvent with potentially decayed confidence, or None if not cached
        """
        async with self._lock:
            if agent_name not in self._cache:
                self._stats.setdefault(agent_name, AgentCacheStats()).cache_misses += 1
                return None

            if symbol not in self._cache[agent_name]:
                self._stats[agent_name].cache_misses += 1
                return None

            cached = self._cache[agent_name][symbol]
            stats = self._stats[agent_name]
            stats.cache_hits += 1
            stats.last_read = datetime.now(timezone.utc)

            # Check staleness
            if cached.is_stale(self._config.max_staleness_seconds):
                stats.stale_reads += 1
                logger.debug(
                    f"AsyncCache: Stale read for {agent_name}/{symbol} "
                    f"(age: {cached.age_seconds:.1f}s)"
                )

            # Return original if no decay requested
            if not apply_staleness_decay:
                return cached.signal

            # Apply confidence decay based on age
            return self._apply_confidence_decay(cached)

    def _apply_confidence_decay(self, cached: CachedSignal) -> SignalEvent:
        """Apply confidence decay to a cached signal based on age."""
        age_seconds = cached.age_seconds

        # No decay if fresh
        if age_seconds <= 0:
            return cached.signal

        # CRITICAL FIX: Validate confidence before decay calculation
        original_confidence = cached.signal.confidence
        if original_confidence is None or original_confidence <= 0:
            # Invalid confidence - return signal with floor confidence
            logger.warning(f"Signal has invalid confidence ({original_confidence}), using floor")
            original_confidence = self._config.min_confidence_floor

        # Calculate decay factor with proper division protection
        decay = age_seconds * self._config.confidence_decay_per_second
        # Calculate minimum acceptable factor to reach floor (avoid division by zero)
        min_factor = self._config.min_confidence_floor / original_confidence if original_confidence > 0 else 0.1
        decay_factor = max(min_factor, 1.0 - decay)

        decayed_confidence = max(
            self._config.min_confidence_floor,
            original_confidence * decay_factor
        )

        # Create new signal with decayed confidence
        from dataclasses import replace
        try:
            return replace(cached.signal, confidence=decayed_confidence)
        except Exception:
            # Fallback: return original
            return cached.signal

    async def get_all_signals_for_symbol(
        self,
        symbol: str,
        agents: list[str] | None = None,
    ) -> Dict[str, SignalEvent]:
        """
        Get all cached signals for a symbol from specified agents.

        Args:
            symbol: Trading symbol
            agents: List of agent names (defaults to all registered)

        Returns:
            Dict of agent_name -> SignalEvent
        """
        agents_to_check = agents or list(self._registered_agents)
        result = {}

        for agent_name in agents_to_check:
            signal = await self.get_signal(agent_name, symbol)
            if signal:
                result[agent_name] = signal

        return result

    async def is_fresh(
        self,
        agent_name: str,
        symbol: str,
        max_age_seconds: float | None = None,
    ) -> bool:
        """
        Check if a cached signal is fresh (not stale).

        Args:
            agent_name: Name of the agent
            symbol: Trading symbol
            max_age_seconds: Override staleness threshold

        Returns:
            True if signal exists and is fresh
        """
        max_age = max_age_seconds or self._config.max_staleness_seconds

        async with self._lock:
            if agent_name not in self._cache:
                return False
            if symbol not in self._cache[agent_name]:
                return False
            return not self._cache[agent_name][symbol].is_stale(max_age)

    async def clear_agent(self, agent_name: str) -> int:
        """
        Clear all cached signals for an agent.

        Returns:
            Number of signals cleared
        """
        async with self._lock:
            if agent_name not in self._cache:
                return 0
            count = len(self._cache[agent_name])
            self._cache[agent_name] = {}
            logger.info(f"AsyncCache: Cleared {count} signals for {agent_name}")
            return count

    async def clear_symbol(self, symbol: str) -> int:
        """
        Clear cached signals for a symbol across all agents.

        Returns:
            Number of signals cleared
        """
        count = 0
        async with self._lock:
            for agent_cache in self._cache.values():
                if symbol in agent_cache:
                    del agent_cache[symbol]
                    count += 1
        logger.info(f"AsyncCache: Cleared {count} signals for symbol {symbol}")
        return count

    async def cleanup_stale(self) -> int:
        """
        Remove all stale entries from cache.

        Returns:
            Number of entries removed
        """
        removed = 0
        async with self._lock:
            for agent_name, agent_cache in self._cache.items():
                stale_symbols = [
                    symbol for symbol, cached in agent_cache.items()
                    if cached.is_stale(self._config.max_staleness_seconds * 2)  # Double threshold for cleanup
                ]
                for symbol in stale_symbols:
                    del agent_cache[symbol]
                    removed += 1

            self._last_cleanup = datetime.now(timezone.utc)

        if removed > 0:
            logger.info(f"AsyncCache: Cleaned up {removed} stale entries")
        return removed

    def get_statistics_sync(self) -> Dict[str, Any]:
        """Get cache statistics (synchronous, for use from sync context)."""
        # Note: This is inherently racy without the async lock, but provides
        # best-effort statistics. For accurate stats, use get_statistics_locked.
        total_entries = sum(len(agent_cache) for agent_cache in self._cache.values())
        total_stale = sum(
            1 for agent_cache in self._cache.values()
            for cached in agent_cache.values()
            if cached.is_stale(self._config.max_staleness_seconds)
        )

        return {
            "total_entries": total_entries,
            "total_stale": total_stale,
            "registered_agents": list(self._registered_agents),
            "config": {
                "max_staleness_seconds": self._config.max_staleness_seconds,
                "confidence_decay_per_second": self._config.confidence_decay_per_second,
            },
            "per_agent": {
                agent_name: {
                    "cached_symbols": list(self._cache.get(agent_name, {}).keys()),
                    "total_updates": stats.total_updates,
                    "cache_hits": stats.cache_hits,
                    "cache_misses": stats.cache_misses,
                    "stale_reads": stats.stale_reads,
                    "hit_rate_pct": (
                        stats.cache_hits / (stats.cache_hits + stats.cache_misses) * 100
                        if (stats.cache_hits + stats.cache_misses) > 0 else 0
                    ),
                    "avg_computation_ms": round(stats.avg_computation_time_ms, 2),
                    "last_update": stats.last_update.isoformat() if stats.last_update else None,
                }
                for agent_name, stats in self._stats.items()
            },
            "last_cleanup": self._last_cleanup.isoformat(),
        }

    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics (thread-safe with async lock)."""
        async with self._lock:
            return self.get_statistics_sync()

    @property
    def registered_agents(self) -> set[str]:
        """Get set of registered out-of-band agents."""
        return self._registered_agents.copy()


def create_async_signal_cache(config: Dict[str, Any] | None = None) -> AsyncSignalCache:
    """Factory function to create AsyncSignalCache from config dict."""
    if config is None:
        return AsyncSignalCache()

    cache_config = AsyncSignalCacheConfig(
        max_staleness_seconds=config.get("max_staleness_seconds", 30.0),
        confidence_decay_per_second=config.get("confidence_decay_per_second", 0.02),
        min_confidence_floor=config.get("min_confidence_floor", 0.1),
        max_history_per_agent=config.get("max_history_per_agent", 100),
        cleanup_interval_seconds=config.get("cleanup_interval_seconds", 60.0),
    )
    return AsyncSignalCache(cache_config)

"""
Historical Data Warmup
======================

Fetches historical bars from IB on startup to warm up agents immediately,
avoiding the need to wait for live data to accumulate.

This allows agents to generate signals within seconds of startup instead
of waiting 30-60 minutes for enough bars to accumulate.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from core.broker import IBBroker
    from core.event_bus import EventBus

from core.events import MarketDataEvent

logger = logging.getLogger(__name__)


@dataclass
class WarmupConfig:
    """Configuration for historical warmup."""
    # Number of bars to fetch per symbol
    bars_to_fetch: int = 100
    # Bar size (1 min, 5 mins, 15 mins, 1 hour, 1 day)
    bar_size: str = "1 min"
    # What data to fetch (TRADES, MIDPOINT, BID, ASK)
    what_to_show: str = "TRADES"
    # Use RTH (Regular Trading Hours) only
    use_rth: bool = False
    # Timeout per symbol request
    timeout_seconds: float = 30.0
    # Max concurrent requests (IB rate limit)
    max_concurrent: int = 5
    # Delay between batches (IB pacing)
    batch_delay_seconds: float = 1.0


class HistoricalWarmup:
    """
    Fetches historical data from IB to warm up agents on startup.

    Instead of waiting 30-60 minutes for bars to accumulate,
    this fetches the last N bars instantly from IB.
    """

    def __init__(
        self,
        broker: "IBBroker",
        event_bus: "EventBus",
        config: Optional[WarmupConfig] = None,
    ):
        self._broker = broker
        self._event_bus = event_bus
        self._config = config or WarmupConfig()
        self._warmup_complete = False
        self._symbols_warmed: Dict[str, int] = {}  # symbol -> bars fetched

    async def warmup_symbols(self, symbols: List[str]) -> Dict[str, int]:
        """
        Fetch historical data for multiple symbols and publish to event bus.

        Args:
            symbols: List of symbols to warm up

        Returns:
            Dict mapping symbol to number of bars fetched
        """
        if not self._broker.is_connected:
            logger.warning("Broker not connected - cannot fetch historical data")
            return {}

        logger.info(f"Starting historical warmup for {len(symbols)} symbols...")
        start_time = datetime.now(timezone.utc)

        results = {}

        # Process in batches to respect IB rate limits
        batch_size = self._config.max_concurrent
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]

            # Fetch batch concurrently
            tasks = [self._warmup_symbol(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for symbol, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to warm up {symbol}: {result}")
                    results[symbol] = 0
                else:
                    results[symbol] = result

            # Delay between batches (IB pacing)
            if i + batch_size < len(symbols):
                await asyncio.sleep(self._config.batch_delay_seconds)

        self._symbols_warmed = results
        self._warmup_complete = True

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        total_bars = sum(results.values())
        successful = sum(1 for v in results.values() if v > 0)

        logger.info(
            f"Historical warmup complete: {successful}/{len(symbols)} symbols, "
            f"{total_bars} total bars in {duration:.1f}s"
        )

        return results

    async def _warmup_symbol(self, symbol: str) -> int:
        """
        Fetch historical data for a single symbol and publish bars to event bus.

        Returns:
            Number of bars fetched
        """
        try:
            # Request historical data from IB using existing broker method
            bars = await asyncio.wait_for(
                self._broker.get_historical_data(
                    symbol=symbol,
                    duration=self._calculate_duration(),
                    bar_size=self._config.bar_size,
                    what_to_show=self._config.what_to_show,
                ),
                timeout=self._config.timeout_seconds,
            )

            if not bars:
                logger.debug(f"No historical data for {symbol}")
                return 0

            # Publish each bar as a MarketDataEvent (oldest first)
            for bar in bars:
                # Parse timestamp from bar date
                bar_date = bar.get("date")
                if isinstance(bar_date, str):
                    try:
                        timestamp = datetime.strptime(bar_date, "%Y%m%d %H:%M:%S")
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    except ValueError:
                        timestamp = datetime.now(timezone.utc)
                elif isinstance(bar_date, datetime):
                    timestamp = bar_date if bar_date.tzinfo else bar_date.replace(tzinfo=timezone.utc)
                else:
                    timestamp = datetime.now(timezone.utc)

                event = MarketDataEvent(
                    symbol=symbol,
                    timestamp=timestamp,
                    bid=bar.get("close", 0),  # Approximate
                    ask=bar.get("close", 0),
                    last=bar.get("close", 0),
                    volume=bar.get("volume", 0),
                    high=bar.get("high", 0),
                    low=bar.get("low", 0),
                    open_price=bar.get("open", 0),
                )

                # Publish directly to event bus (bypasses normal flow)
                await self._event_bus.publish(event)

            logger.debug(f"Warmed up {symbol}: {len(bars)} bars")
            return len(bars)

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching historical data for {symbol}")
            return 0
        except Exception as e:
            logger.warning(f"Error fetching historical data for {symbol}: {e}")
            return 0

    def _calculate_duration(self) -> str:
        """
        Calculate IB duration string based on bar size and count.

        IB duration formats: "60 S", "30 D", "1 M", "1 Y"
        """
        bars = self._config.bars_to_fetch
        bar_size = self._config.bar_size.lower()

        if "sec" in bar_size:
            seconds = bars * int(bar_size.split()[0])
            return f"{seconds} S"
        elif "min" in bar_size:
            minutes = bars * int(bar_size.split()[0])
            # Convert to days if > 1 day worth
            if minutes > 1440:
                days = (minutes // 1440) + 1
                return f"{days} D"
            else:
                return f"{minutes * 60} S"
        elif "hour" in bar_size:
            hours = bars * int(bar_size.split()[0])
            days = (hours // 24) + 1
            return f"{days} D"
        elif "day" in bar_size:
            return f"{bars} D"
        else:
            # Default: 1 day
            return "1 D"

    @property
    def is_complete(self) -> bool:
        """Check if warmup has been completed."""
        return self._warmup_complete

    def get_status(self) -> dict:
        """Get warmup status for monitoring."""
        return {
            "complete": self._warmup_complete,
            "symbols_warmed": len(self._symbols_warmed),
            "total_bars": sum(self._symbols_warmed.values()),
            "config": {
                "bars_to_fetch": self._config.bars_to_fetch,
                "bar_size": self._config.bar_size,
            },
        }


def create_historical_warmup(
    broker: "IBBroker",
    event_bus: "EventBus",
    config: Optional[dict] = None,
) -> HistoricalWarmup:
    """Factory function to create HistoricalWarmup."""
    warmup_config = None
    if config:
        warmup_config = WarmupConfig(
            bars_to_fetch=config.get("bars_to_fetch", 100),
            bar_size=config.get("bar_size", "1 min"),
            what_to_show=config.get("what_to_show", "TRADES"),
            use_rth=config.get("use_rth", False),
            timeout_seconds=config.get("timeout_seconds", 30.0),
            max_concurrent=config.get("max_concurrent", 5),
            batch_delay_seconds=config.get("batch_delay_seconds", 1.0),
        )

    return HistoricalWarmup(broker, event_bus, warmup_config)

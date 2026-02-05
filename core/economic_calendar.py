"""
Economic Calendar Integration
=============================

Fetches and manages economic events for the EventDrivenAgent.
Supports multiple data sources with fallback.

Sources:
1. FCS API (free tier)
2. Static known events (FOMC schedule, NFP dates)
3. Manual registration

Usage:
    calendar = EconomicCalendar()
    await calendar.initialize()
    events = calendar.get_upcoming_events(hours_ahead=48)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Optional, List
from enum import Enum
import json

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from strategies.event_driven_strategy import (
    EconomicEvent,
    EventType as EconEventType,
    EventImpact,
)

if TYPE_CHECKING:
    from agents.event_driven_agent import EventDrivenAgent

logger = logging.getLogger(__name__)


# Known FOMC meeting dates for 2026 (Fed releases schedule in advance)
FOMC_DATES_2026 = [
    datetime(2026, 1, 28, 19, 0, tzinfo=timezone.utc),   # January
    datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc),   # March
    datetime(2026, 5, 6, 18, 0, tzinfo=timezone.utc),    # May
    datetime(2026, 6, 17, 18, 0, tzinfo=timezone.utc),   # June
    datetime(2026, 7, 29, 18, 0, tzinfo=timezone.utc),   # July
    datetime(2026, 9, 16, 18, 0, tzinfo=timezone.utc),   # September
    datetime(2026, 11, 4, 18, 0, tzinfo=timezone.utc),   # November
    datetime(2026, 12, 16, 19, 0, tzinfo=timezone.utc),  # December
]

# NFP is always first Friday of the month at 8:30 AM ET (13:30 UTC)
def get_nfp_dates_2026() -> List[datetime]:
    """Generate NFP release dates for 2026."""
    nfp_dates = []
    for month in range(1, 13):
        # Find first Friday of the month
        first_day = datetime(2026, month, 1, 13, 30, tzinfo=timezone.utc)
        # weekday(): Monday=0, Friday=4
        days_until_friday = (4 - first_day.weekday()) % 7
        if days_until_friday == 0 and first_day.weekday() != 4:
            days_until_friday = 7
        first_friday = first_day + timedelta(days=days_until_friday)
        nfp_dates.append(first_friday)
    return nfp_dates

NFP_DATES_2026 = get_nfp_dates_2026()

# CPI is typically released mid-month (around 13th) at 8:30 AM ET
CPI_DATES_2026 = [
    datetime(2026, 1, 14, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 2, 12, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 3, 11, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 4, 14, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 5, 13, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 6, 10, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 7, 14, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 8, 12, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 9, 15, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 10, 13, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 11, 12, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 12, 10, 13, 30, tzinfo=timezone.utc),
]

# GDP is released end of month (advance estimate)
GDP_DATES_2026 = [
    datetime(2026, 1, 29, 13, 30, tzinfo=timezone.utc),   # Q4 2025 Advance
    datetime(2026, 2, 26, 13, 30, tzinfo=timezone.utc),   # Q4 2025 Second
    datetime(2026, 3, 26, 13, 30, tzinfo=timezone.utc),   # Q4 2025 Third
    datetime(2026, 4, 30, 13, 30, tzinfo=timezone.utc),   # Q1 2026 Advance
    datetime(2026, 5, 28, 13, 30, tzinfo=timezone.utc),   # Q1 2026 Second
    datetime(2026, 6, 25, 13, 30, tzinfo=timezone.utc),   # Q1 2026 Third
    datetime(2026, 7, 30, 13, 30, tzinfo=timezone.utc),   # Q2 2026 Advance
    datetime(2026, 8, 27, 13, 30, tzinfo=timezone.utc),   # Q2 2026 Second
    datetime(2026, 9, 24, 13, 30, tzinfo=timezone.utc),   # Q2 2026 Third
    datetime(2026, 10, 29, 13, 30, tzinfo=timezone.utc),  # Q3 2026 Advance
    datetime(2026, 11, 25, 13, 30, tzinfo=timezone.utc),  # Q3 2026 Second
    datetime(2026, 12, 23, 13, 30, tzinfo=timezone.utc),  # Q3 2026 Third
]

# ISM PMI - first business day of the month at 10:00 AM ET (15:00 UTC)
ISM_PMI_DATES_2026 = [
    datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 2, 2, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 4, 1, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 6, 1, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 8, 3, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 9, 1, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 12, 1, 15, 0, tzinfo=timezone.utc),
]

# Retail Sales - mid-month at 8:30 AM ET
RETAIL_SALES_DATES_2026 = [
    datetime(2026, 1, 16, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 2, 17, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 3, 17, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 4, 15, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 5, 15, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 6, 16, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 7, 16, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 8, 14, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 9, 16, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 10, 16, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 11, 17, 13, 30, tzinfo=timezone.utc),
    datetime(2026, 12, 16, 13, 30, tzinfo=timezone.utc),
]


@dataclass
class CalendarConfig:
    """Configuration for economic calendar."""
    use_api: bool = True
    api_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    update_interval_hours: int = 4
    lookahead_hours: int = 168  # 1 week
    include_low_impact: bool = False


class EconomicCalendar:
    """
    Economic Calendar manager.

    Fetches and maintains a list of upcoming economic events
    from multiple sources with automatic fallback.
    """

    def __init__(self, config: Optional[CalendarConfig] = None):
        self._config = config or CalendarConfig()
        self._events: List[EconomicEvent] = []
        self._last_update: Optional[datetime] = None
        self._update_task: Optional[asyncio.Task] = None
        self._running = False

    async def initialize(self) -> None:
        """Initialize calendar with static and API events."""
        logger.info("Initializing Economic Calendar...")

        # Load static known events
        self._load_static_events()

        # Try to fetch from API
        if self._config.use_api and HAS_AIOHTTP:
            await self._fetch_api_events()

        logger.info(f"Economic Calendar initialized with {len(self._events)} events")

    def _load_static_events(self) -> None:
        """Load statically known economic events."""
        now = datetime.now(timezone.utc)

        # FOMC meetings (HIGH impact)
        for dt in FOMC_DATES_2026:
            if dt > now:
                self._events.append(EconomicEvent(
                    event_type=EconEventType.FOMC,
                    timestamp=dt,
                    impact=EventImpact.HIGH,
                    description="FOMC Interest Rate Decision",
                ))

        # NFP (HIGH impact)
        for dt in NFP_DATES_2026:
            if dt > now:
                self._events.append(EconomicEvent(
                    event_type=EconEventType.NFP,
                    timestamp=dt,
                    impact=EventImpact.HIGH,
                    description="Non-Farm Payrolls",
                ))

        # CPI (HIGH impact)
        for dt in CPI_DATES_2026:
            if dt > now:
                self._events.append(EconomicEvent(
                    event_type=EconEventType.CPI,
                    timestamp=dt,
                    impact=EventImpact.HIGH,
                    description="Consumer Price Index",
                ))

        # GDP (MEDIUM impact)
        for dt in GDP_DATES_2026:
            if dt > now:
                self._events.append(EconomicEvent(
                    event_type=EconEventType.GDP,
                    timestamp=dt,
                    impact=EventImpact.MEDIUM,
                    description="Gross Domestic Product",
                ))

        # ISM Manufacturing PMI (MEDIUM impact)
        for dt in ISM_PMI_DATES_2026:
            if dt > now:
                self._events.append(EconomicEvent(
                    event_type=EconEventType.ISM_MFG,
                    timestamp=dt,
                    impact=EventImpact.MEDIUM,
                    description="ISM Manufacturing PMI",
                ))

        # Retail Sales (MEDIUM impact)
        for dt in RETAIL_SALES_DATES_2026:
            if dt > now:
                self._events.append(EconomicEvent(
                    event_type=EconEventType.RETAIL_SALES,
                    timestamp=dt,
                    impact=EventImpact.MEDIUM,
                    description="Retail Sales MoM",
                ))

        # Sort by timestamp
        self._events.sort(key=lambda e: e.timestamp)

        logger.info(f"Loaded {len(self._events)} static economic events")

    async def _fetch_api_events(self) -> None:
        """Fetch events from ForexFactory JSON API."""
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not available, skipping API fetch")
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._config.api_url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._parse_forex_factory_events(data)
                        self._last_update = datetime.now(timezone.utc)
                        logger.info(f"Fetched {len(data)} events from API")
                    else:
                        logger.warning(f"API returned status {response.status}")
        except asyncio.TimeoutError:
            logger.warning("API request timed out")
        except Exception as e:
            logger.warning(f"Failed to fetch API events: {e}")

    def _parse_forex_factory_events(self, data: list) -> None:
        """Parse ForexFactory JSON format."""
        event_type_map = {
            "FOMC": EconEventType.FOMC,
            "Non-Farm": EconEventType.NFP,
            "Nonfarm": EconEventType.NFP,
            "NFP": EconEventType.NFP,
            "CPI": EconEventType.CPI,
            "Consumer Price": EconEventType.CPI,
            "GDP": EconEventType.GDP,
            "Gross Domestic": EconEventType.GDP,
            "ISM Manufacturing": EconEventType.ISM_MFG,
            "ISM PMI": EconEventType.ISM_MFG,
            "ISM Services": EconEventType.ISM_SVC,
            "Retail Sales": EconEventType.RETAIL_SALES,
            "Core Retail": EconEventType.RETAIL_SALES,
            "PPI": EconEventType.PPI,
            "Producer Price": EconEventType.PPI,
            "Initial Claims": EconEventType.JOBLESS_CLAIMS,
            "Jobless Claims": EconEventType.JOBLESS_CLAIMS,
        }

        impact_map = {
            "High": EventImpact.HIGH,
            "Medium": EventImpact.MEDIUM,
            "Low": EventImpact.LOW,
        }

        now = datetime.now(timezone.utc)

        for item in data:
            try:
                # Parse date/time
                date_str = item.get("date", "")
                time_str = item.get("time", "")
                title = item.get("title", "")
                impact_str = item.get("impact", "Low")
                country = item.get("country", "")

                # Only US events
                if country != "USD":
                    continue

                # Skip low impact unless configured
                impact = impact_map.get(impact_str, EventImpact.LOW)
                if impact == EventImpact.LOW and not self._config.include_low_impact:
                    continue

                # Match event type
                event_type = None
                for keyword, etype in event_type_map.items():
                    if keyword.lower() in title.lower():
                        event_type = etype
                        break

                if event_type is None:
                    continue

                # Parse timestamp
                try:
                    # ForexFactory format: "Jan 15" and "8:30am"
                    dt_str = f"{date_str} 2026 {time_str}"
                    dt = datetime.strptime(dt_str, "%b %d %Y %I:%M%p")
                    dt = dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                if dt <= now:
                    continue

                # Check if already exists (from static)
                exists = any(
                    e.event_type == event_type and
                    abs((e.timestamp - dt).total_seconds()) < 3600
                    for e in self._events
                )

                if not exists:
                    forecast = item.get("forecast")
                    previous = item.get("previous")

                    self._events.append(EconomicEvent(
                        event_type=event_type,
                        timestamp=dt,
                        impact=impact,
                        description=title,
                        forecast=float(forecast.strip('%')) if forecast else None,
                        previous=float(previous.strip('%')) if previous else None,
                    ))

            except Exception as e:
                logger.debug(f"Failed to parse event: {e}")
                continue

        # Re-sort after adding API events
        self._events.sort(key=lambda e: e.timestamp)

    def get_upcoming_events(
        self,
        hours_ahead: int = 48,
        event_types: Optional[List[EconEventType]] = None,
    ) -> List[EconomicEvent]:
        """
        Get upcoming economic events.

        Args:
            hours_ahead: Look ahead window in hours
            event_types: Filter by specific event types

        Returns:
            List of upcoming EconomicEvent objects
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        events = [
            e for e in self._events
            if now <= e.timestamp <= cutoff
        ]

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        return events

    def get_next_event(
        self,
        event_types: Optional[List[EconEventType]] = None,
    ) -> Optional[EconomicEvent]:
        """Get the next upcoming event."""
        now = datetime.now(timezone.utc)

        for event in self._events:
            if event.timestamp > now:
                if event_types is None or event.event_type in event_types:
                    return event

        return None

    def register_with_agent(self, agent: "EventDrivenAgent") -> int:
        """
        Register all upcoming events with an EventDrivenAgent.

        Returns:
            Number of events registered
        """
        count = 0
        for event in self.get_upcoming_events(hours_ahead=self._config.lookahead_hours):
            agent.register_event(event)
            count += 1

        logger.info(f"Registered {count} events with EventDrivenAgent")
        return count

    async def start_auto_update(self) -> None:
        """Start background task to periodically update events."""
        self._running = True
        self._update_task = asyncio.create_task(self._auto_update_loop())
        logger.info("Started economic calendar auto-update")

    async def stop_auto_update(self) -> None:
        """Stop the background update task."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped economic calendar auto-update")

    async def _auto_update_loop(self) -> None:
        """Background loop to update events periodically."""
        while self._running:
            try:
                await asyncio.sleep(self._config.update_interval_hours * 3600)
                if self._config.use_api and HAS_AIOHTTP:
                    await self._fetch_api_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-update loop: {e}")

    def get_status(self) -> dict:
        """Get calendar status for monitoring."""
        now = datetime.now(timezone.utc)
        upcoming = [e for e in self._events if e.timestamp > now]

        next_event = self.get_next_event()

        return {
            "total_events": len(self._events),
            "upcoming_events": len(upcoming),
            "next_event": {
                "type": next_event.event_type.value if next_event else None,
                "timestamp": next_event.timestamp.isoformat() if next_event else None,
                "hours_until": (
                    (next_event.timestamp - now).total_seconds() / 3600
                    if next_event else None
                ),
            },
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "auto_update_running": self._running,
        }


# Factory function
def create_economic_calendar(config: Optional[dict] = None) -> EconomicCalendar:
    """Create an EconomicCalendar with optional config."""
    if config:
        cal_config = CalendarConfig(
            use_api=config.get("use_api", True),
            api_url=config.get("api_url", CalendarConfig.api_url),
            update_interval_hours=config.get("update_interval_hours", 4),
            lookahead_hours=config.get("lookahead_hours", 168),
            include_low_impact=config.get("include_low_impact", False),
        )
        return EconomicCalendar(cal_config)
    return EconomicCalendar()

"""
FX Session Awareness Module
===========================

Forex trading session awareness and analytics (Issue #X9).

Features:
- Major FX session tracking (Sydney, Tokyo, London, New York)
- Session overlap detection
- Liquidity estimation by session
- Volatility patterns by session
- Best execution timing recommendations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timezone, timedelta
from enum import Enum
from typing import Any
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class FXSession(str, Enum):
    """Major FX trading sessions."""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"


class SessionOverlap(str, Enum):
    """Session overlap periods."""
    TOKYO_LONDON = "tokyo_london"  # Brief overlap
    LONDON_NEW_YORK = "london_new_york"  # Most liquid period
    NEW_YORK_SYDNEY = "new_york_sydney"  # Quietest period
    SYDNEY_TOKYO = "sydney_tokyo"  # Asia session


@dataclass
class SessionTime:
    """Trading session times."""
    session: FXSession
    open_utc: time
    close_utc: time
    peak_hours: tuple[time, time]  # Most active period

    def is_open(self, utc_time: time) -> bool:
        """Check if session is open at given UTC time."""
        if self.open_utc <= self.close_utc:
            return self.open_utc <= utc_time <= self.close_utc
        else:
            # Crosses midnight
            return utc_time >= self.open_utc or utc_time <= self.close_utc

    def is_peak(self, utc_time: time) -> bool:
        """Check if in peak hours."""
        return self.peak_hours[0] <= utc_time <= self.peak_hours[1]


# Standard session times (approximate, may vary by season due to DST)
SESSION_TIMES = {
    FXSession.SYDNEY: SessionTime(
        session=FXSession.SYDNEY,
        open_utc=time(21, 0),  # 9 PM UTC (previous day)
        close_utc=time(6, 0),  # 6 AM UTC
        peak_hours=(time(22, 0), time(2, 0)),
    ),
    FXSession.TOKYO: SessionTime(
        session=FXSession.TOKYO,
        open_utc=time(0, 0),  # Midnight UTC
        close_utc=time(9, 0),  # 9 AM UTC
        peak_hours=(time(1, 0), time(6, 0)),
    ),
    FXSession.LONDON: SessionTime(
        session=FXSession.LONDON,
        open_utc=time(7, 0),  # 7 AM UTC
        close_utc=time(16, 0),  # 4 PM UTC
        peak_hours=(time(8, 0), time(12, 0)),
    ),
    FXSession.NEW_YORK: SessionTime(
        session=FXSession.NEW_YORK,
        open_utc=time(12, 0),  # Noon UTC
        close_utc=time(21, 0),  # 9 PM UTC
        peak_hours=(time(13, 0), time(17, 0)),
    ),
}


@dataclass
class SessionCharacteristics:
    """Trading characteristics for a session."""
    session: FXSession
    avg_volatility_pips: float
    avg_spread_pips: float
    liquidity_score: float  # 0-100
    dominant_pairs: list[str]
    typical_moves: dict[str, float]  # pair -> typical range in pips


# Session-specific characteristics
SESSION_CHARACTERISTICS = {
    FXSession.SYDNEY: SessionCharacteristics(
        session=FXSession.SYDNEY,
        avg_volatility_pips=30,
        avg_spread_pips=2.5,
        liquidity_score=40,
        dominant_pairs=["AUD/USD", "NZD/USD", "AUD/NZD", "AUD/JPY"],
        typical_moves={"AUD/USD": 40, "NZD/USD": 35, "AUD/JPY": 45},
    ),
    FXSession.TOKYO: SessionCharacteristics(
        session=FXSession.TOKYO,
        avg_volatility_pips=40,
        avg_spread_pips=2.0,
        liquidity_score=60,
        dominant_pairs=["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY"],
        typical_moves={"USD/JPY": 50, "EUR/JPY": 60, "GBP/JPY": 70},
    ),
    FXSession.LONDON: SessionCharacteristics(
        session=FXSession.LONDON,
        avg_volatility_pips=80,
        avg_spread_pips=1.0,
        liquidity_score=95,
        dominant_pairs=["EUR/USD", "GBP/USD", "EUR/GBP", "USD/CHF"],
        typical_moves={"EUR/USD": 80, "GBP/USD": 100, "EUR/GBP": 40},
    ),
    FXSession.NEW_YORK: SessionCharacteristics(
        session=FXSession.NEW_YORK,
        avg_volatility_pips=70,
        avg_spread_pips=1.2,
        liquidity_score=90,
        dominant_pairs=["EUR/USD", "USD/CAD", "USD/MXN", "GBP/USD"],
        typical_moves={"EUR/USD": 70, "USD/CAD": 60, "GBP/USD": 90},
    ),
}


@dataclass
class FXSessionState:
    """Current FX session state."""
    timestamp: datetime
    active_sessions: list[FXSession]
    overlaps: list[SessionOverlap]
    liquidity_level: str  # 'low', 'normal', 'high', 'peak'
    recommended_pairs: list[str]
    spread_multiplier: float  # 1.0 = normal, higher = wider
    volatility_adjustment: float  # 1.0 = normal

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'active_sessions': [s.value for s in self.active_sessions],
            'overlaps': [o.value for o in self.overlaps],
            'liquidity_level': self.liquidity_level,
            'recommended_pairs': self.recommended_pairs,
            'spread_multiplier': self.spread_multiplier,
            'volatility_adjustment': self.volatility_adjustment,
        }


class FXSessionManager:
    """
    Manages FX session awareness and provides trading recommendations.

    Tracks which sessions are active and their characteristics.
    """

    def __init__(self):
        self._session_times = SESSION_TIMES
        self._characteristics = SESSION_CHARACTERISTICS

        # Historical volatility tracking per session
        self._session_volatility: dict[FXSession, list[float]] = defaultdict(list)
        self._session_spreads: dict[FXSession, list[float]] = defaultdict(list)

    def get_current_state(self, utc_now: datetime | None = None) -> FXSessionState:
        """
        Get current FX session state.

        Args:
            utc_now: Current UTC time (defaults to now)

        Returns:
            FXSessionState with active sessions and recommendations
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        current_time = utc_now.time()

        # Find active sessions
        active_sessions = []
        for session, times in self._session_times.items():
            if times.is_open(current_time):
                active_sessions.append(session)

        # Detect overlaps
        overlaps = self._detect_overlaps(active_sessions)

        # Calculate liquidity level
        liquidity_level = self._calculate_liquidity_level(active_sessions, overlaps)

        # Get recommended pairs
        recommended_pairs = self._get_recommended_pairs(active_sessions)

        # Calculate adjustments
        spread_mult = self._calculate_spread_multiplier(liquidity_level)
        vol_adj = self._calculate_volatility_adjustment(active_sessions, overlaps)

        return FXSessionState(
            timestamp=utc_now,
            active_sessions=active_sessions,
            overlaps=overlaps,
            liquidity_level=liquidity_level,
            recommended_pairs=recommended_pairs,
            spread_multiplier=spread_mult,
            volatility_adjustment=vol_adj,
        )

    def _detect_overlaps(self, active_sessions: list[FXSession]) -> list[SessionOverlap]:
        """Detect session overlaps."""
        overlaps = []

        if FXSession.TOKYO in active_sessions and FXSession.LONDON in active_sessions:
            overlaps.append(SessionOverlap.TOKYO_LONDON)

        if FXSession.LONDON in active_sessions and FXSession.NEW_YORK in active_sessions:
            overlaps.append(SessionOverlap.LONDON_NEW_YORK)

        if FXSession.NEW_YORK in active_sessions and FXSession.SYDNEY in active_sessions:
            overlaps.append(SessionOverlap.NEW_YORK_SYDNEY)

        if FXSession.SYDNEY in active_sessions and FXSession.TOKYO in active_sessions:
            overlaps.append(SessionOverlap.SYDNEY_TOKYO)

        return overlaps

    def _calculate_liquidity_level(
        self,
        active_sessions: list[FXSession],
        overlaps: list[SessionOverlap],
    ) -> str:
        """Calculate overall liquidity level."""
        if not active_sessions:
            return "low"

        # London-NY overlap is most liquid
        if SessionOverlap.LONDON_NEW_YORK in overlaps:
            return "peak"

        # London alone is very liquid
        if FXSession.LONDON in active_sessions:
            return "high"

        # NY alone is liquid
        if FXSession.NEW_YORK in active_sessions:
            return "high"

        # Tokyo is moderate
        if FXSession.TOKYO in active_sessions:
            return "normal"

        # Sydney alone is quieter
        return "low"

    def _get_recommended_pairs(self, active_sessions: list[FXSession]) -> list[str]:
        """Get recommended pairs for active sessions."""
        pairs = set()

        for session in active_sessions:
            if session in self._characteristics:
                pairs.update(self._characteristics[session].dominant_pairs)

        # Prioritize majors during liquid times
        majors = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"]
        recommended = [p for p in majors if p in pairs]
        recommended.extend([p for p in pairs if p not in majors])

        return recommended[:8]  # Top 8 pairs

    def _calculate_spread_multiplier(self, liquidity_level: str) -> float:
        """Calculate spread multiplier based on liquidity."""
        multipliers = {
            "peak": 0.8,  # Tighter spreads
            "high": 1.0,  # Normal
            "normal": 1.3,  # Slightly wider
            "low": 2.0,  # Much wider
        }
        return multipliers.get(liquidity_level, 1.5)

    def _calculate_volatility_adjustment(
        self,
        active_sessions: list[FXSession],
        overlaps: list[SessionOverlap],
    ) -> float:
        """Calculate volatility adjustment factor."""
        if SessionOverlap.LONDON_NEW_YORK in overlaps:
            return 1.5  # Higher volatility expected

        if not active_sessions:
            return 0.5  # Very quiet

        # Average of active session volatilities
        total_vol = 0
        count = 0
        for session in active_sessions:
            if session in self._characteristics:
                total_vol += self._characteristics[session].avg_volatility_pips
                count += 1

        if count > 0:
            avg_vol = total_vol / count
            # Normalize to 1.0 = 60 pips average
            return avg_vol / 60

        return 1.0

    def get_best_execution_window(self, pair: str) -> dict:
        """
        Get best execution window for a currency pair.

        Returns recommended times and expected conditions.
        """
        # Determine primary sessions for pair
        base = pair[:3]
        quote = pair[4:]

        primary_sessions = []

        # Map currencies to sessions
        currency_sessions = {
            "AUD": FXSession.SYDNEY,
            "NZD": FXSession.SYDNEY,
            "JPY": FXSession.TOKYO,
            "EUR": FXSession.LONDON,
            "GBP": FXSession.LONDON,
            "CHF": FXSession.LONDON,
            "USD": FXSession.NEW_YORK,
            "CAD": FXSession.NEW_YORK,
        }

        if base in currency_sessions:
            primary_sessions.append(currency_sessions[base])
        if quote in currency_sessions:
            primary_sessions.append(currency_sessions[quote])

        # Major pairs best during London/NY overlap
        if pair in ["EUR/USD", "GBP/USD"]:
            best_utc_start = time(13, 0)  # NY open, London afternoon
            best_utc_end = time(16, 0)  # London close
            expected_liquidity = "peak"
        elif "JPY" in pair:
            best_utc_start = time(1, 0)  # Tokyo active
            best_utc_end = time(8, 0)  # London morning
            expected_liquidity = "high"
        elif "AUD" in pair or "NZD" in pair:
            best_utc_start = time(22, 0)  # Sydney
            best_utc_end = time(6, 0)  # Asia
            expected_liquidity = "normal"
        else:
            best_utc_start = time(12, 0)
            best_utc_end = time(16, 0)
            expected_liquidity = "high"

        return {
            'pair': pair,
            'best_start_utc': best_utc_start.isoformat(),
            'best_end_utc': best_utc_end.isoformat(),
            'primary_sessions': [s.value for s in primary_sessions],
            'expected_liquidity': expected_liquidity,
            'spread_expectation': 'tight' if expected_liquidity in ['peak', 'high'] else 'normal',
        }

    def record_volatility(self, session: FXSession, volatility: float) -> None:
        """Record observed volatility for session."""
        self._session_volatility[session].append(volatility)

        # Keep last 100 observations
        if len(self._session_volatility[session]) > 100:
            self._session_volatility[session] = self._session_volatility[session][-100:]

    def record_spread(self, session: FXSession, spread: float) -> None:
        """Record observed spread for session."""
        self._session_spreads[session].append(spread)

        # Keep last 100 observations
        if len(self._session_spreads[session]) > 100:
            self._session_spreads[session] = self._session_spreads[session][-100:]

    def get_session_stats(self) -> dict:
        """Get session statistics from recorded data."""
        stats = {}

        for session in FXSession:
            vol_data = self._session_volatility.get(session, [])
            spread_data = self._session_spreads.get(session, [])

            stats[session.value] = {
                'default_characteristics': {
                    'avg_volatility': self._characteristics[session].avg_volatility_pips,
                    'avg_spread': self._characteristics[session].avg_spread_pips,
                    'liquidity_score': self._characteristics[session].liquidity_score,
                },
                'observed': {
                    'avg_volatility': statistics.mean(vol_data) if vol_data else None,
                    'avg_spread': statistics.mean(spread_data) if spread_data else None,
                    'observation_count': len(vol_data),
                },
            }

        return stats

    def is_trading_recommended(
        self,
        pair: str,
        utc_now: datetime | None = None,
    ) -> tuple[bool, str]:
        """
        Check if trading is recommended for a pair at current time.

        Returns (recommended, reason).
        """
        state = self.get_current_state(utc_now)

        # No active sessions
        if not state.active_sessions:
            return False, "No major sessions active, low liquidity expected"

        # Weekend gap risk (Friday evening to Sunday evening)
        if utc_now:
            weekday = utc_now.weekday()
            hour = utc_now.hour

            # Friday after NY close
            if weekday == 4 and hour >= 22:
                return False, "Weekend approaching, gap risk"

            # Saturday
            if weekday == 5:
                return False, "Weekend, market closed"

            # Sunday before Sydney open
            if weekday == 6 and hour < 21:
                return False, "Weekend, market not yet open"

        # Check if pair is in recommended list
        if pair not in state.recommended_pairs:
            return True, f"Trading possible but {pair} not optimal for current sessions"

        # Low liquidity warning
        if state.liquidity_level == "low":
            return True, "Low liquidity, expect wider spreads"

        return True, f"Optimal trading conditions ({state.liquidity_level} liquidity)"


@dataclass
class SessionVolatilityPattern:
    """Volatility pattern by hour for a session."""
    session: FXSession
    hourly_vol_multipliers: dict[int, float]  # hour -> multiplier (1.0 = average)


# Typical intraday volatility patterns
VOLATILITY_PATTERNS = {
    FXSession.LONDON: SessionVolatilityPattern(
        session=FXSession.LONDON,
        hourly_vol_multipliers={
            7: 0.8,  # Opening
            8: 1.2,  # Active
            9: 1.3,  # Peak
            10: 1.2,
            11: 1.0,
            12: 0.9,  # Lunch
            13: 1.1,  # NY overlap starts
            14: 1.3,  # Peak overlap
            15: 1.2,
            16: 0.8,  # Closing
        },
    ),
    FXSession.NEW_YORK: SessionVolatilityPattern(
        session=FXSession.NEW_YORK,
        hourly_vol_multipliers={
            12: 0.9,
            13: 1.2,  # Opening
            14: 1.3,  # London overlap
            15: 1.2,
            16: 1.0,  # London close
            17: 0.9,
            18: 0.8,
            19: 0.7,
            20: 0.6,
            21: 0.5,  # Closing
        },
    ),
}


def get_expected_volatility_multiplier(utc_now: datetime | None = None) -> float:
    """Get expected volatility multiplier for current time."""
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)

    hour = utc_now.hour
    multiplier = 1.0

    # Check active patterns
    for session, pattern in VOLATILITY_PATTERNS.items():
        if hour in pattern.hourly_vol_multipliers:
            mult = pattern.hourly_vol_multipliers[hour]
            multiplier = max(multiplier, mult)

    return multiplier

"""
Session-Based Trading Strategy (Phase 6.1)
==========================================

Exploits intraday session patterns, particularly the London/NY overlap
which historically shows the highest volume and best trending behavior.

Key features:
- Opening range breakout detection
- Session-aware signal generation
- Volume confirmation for entries
- Optimal session windows by asset class

Research basis:
- London/NY overlap (13:00-17:00 UTC) accounts for ~35% of daily forex volume
- Opening range breakout has 65%+ success rate with proper filters
- Session momentum tends to persist for 2-4 hours after breakout

MATURITY: ALPHA
---------------
Status: New implementation
- [x] Session window definitions
- [x] Opening range calculation
- [x] Breakout detection
- [x] Volume confirmation
- [ ] Integration with main system
- [ ] Backtesting validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timezone, timedelta
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Major trading sessions."""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    AFTER_HOURS = "after_hours"


class SessionQuality(Enum):
    """Session quality rating for trading."""
    EXCELLENT = "excellent"  # High volume, good trends
    GOOD = "good"            # Decent conditions
    FAIR = "fair"            # Acceptable but cautious
    POOR = "poor"            # Low volume, choppy
    AVOID = "avoid"          # Market closed or very low liquidity


@dataclass
class SessionWindow:
    """Definition of a trading session window."""
    name: str
    session: TradingSession
    start_time: time  # UTC
    end_time: time    # UTC
    quality: SessionQuality
    typical_volume_pct: float  # % of daily volume
    best_for: list[str] = field(default_factory=list)  # Asset classes
    notes: str = ""


@dataclass
class OpeningRange:
    """Opening range calculation result."""
    high: float
    low: float
    mid: float
    range_size: float
    range_pct: float  # As % of mid price
    volume_in_range: float
    n_bars: int
    session: TradingSession
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SessionSignal:
    """Signal from session-based analysis."""
    symbol: str
    session: TradingSession
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    signal_type: str  # "breakout", "range_trade", "momentum"
    strength: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    opening_range: OpeningRange | None
    volume_confirmed: bool
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# SESSION DEFINITIONS
# =============================================================================

SESSION_WINDOWS: dict[str, SessionWindow] = {
    "asian": SessionWindow(
        name="Asian Session",
        session=TradingSession.ASIAN,
        start_time=time(0, 0),   # 00:00 UTC
        end_time=time(8, 0),     # 08:00 UTC
        quality=SessionQuality.FAIR,
        typical_volume_pct=15.0,
        best_for=["forex_jpy", "forex_aud", "nikkei"],
        notes="Lower volatility, range-bound behavior common",
    ),
    "london": SessionWindow(
        name="London Session",
        session=TradingSession.LONDON,
        start_time=time(7, 0),   # 07:00 UTC
        end_time=time(16, 0),    # 16:00 UTC
        quality=SessionQuality.EXCELLENT,
        typical_volume_pct=35.0,
        best_for=["forex_eur", "forex_gbp", "dax", "ftse"],
        notes="Highest volume for European pairs",
    ),
    "new_york": SessionWindow(
        name="New York Session",
        session=TradingSession.NEW_YORK,
        start_time=time(13, 0),  # 13:00 UTC
        end_time=time(21, 0),    # 21:00 UTC
        quality=SessionQuality.EXCELLENT,
        typical_volume_pct=30.0,
        best_for=["equity", "forex_usd", "es", "nq"],
        notes="US market hours, high liquidity",
    ),
    "london_ny_overlap": SessionWindow(
        name="London/NY Overlap",
        session=TradingSession.LONDON_NY_OVERLAP,
        start_time=time(13, 0),  # 13:00 UTC
        end_time=time(17, 0),    # 17:00 UTC (London closes at 17:00, not 16:00)
        quality=SessionQuality.EXCELLENT,
        typical_volume_pct=25.0,
        best_for=["forex", "gold", "oil", "indices"],
        notes="OPTIMAL: Highest liquidity and trend potential",
    ),
}


class SessionStrategy:
    """
    Session-based trading strategy (Phase 6.1).

    Strategies:
    1. Opening Range Breakout: Trade breaks of first N minutes range
    2. Session Momentum: Follow session direction after confirmation
    3. Session Fade: Counter-trend at session extremes

    Configuration:
        opening_range_minutes: Duration to calculate opening range (default: 30)
        breakout_threshold_atr: Minimum breakout size in ATR units
        volume_confirmation_mult: Volume must be N x average for confirmation
        session_filter: Only trade during specified sessions
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize session strategy."""
        config = config or {}

        # Opening range settings
        self._opening_range_minutes = config.get("opening_range_minutes", 30)
        # Breakout threshold: 1.5 ATR reduces false breakouts (was 0.5, too sensitive)
        # Research: Toby Crabel's ORB uses 1.0-2.0 ATR for reliable breakouts
        self._breakout_threshold_atr = config.get("breakout_threshold_atr", 1.5)
        self._breakout_confirmation_bars = config.get("breakout_confirmation_bars", 2)

        # Volume settings
        self._volume_confirmation_mult = config.get("volume_confirmation_mult", 1.5)
        self._min_volume_percentile = config.get("min_volume_percentile", 30)

        # Session filter
        self._allowed_sessions = config.get("allowed_sessions", [
            TradingSession.LONDON_NY_OVERLAP,
            TradingSession.LONDON,
            TradingSession.NEW_YORK,
        ])

        # Risk settings
        # Stop-loss: 2.0 ATR prevents early stop-outs (was 1.5, too tight)
        # Take-profit: 4.0 ATR for proper trend capture (was 2.5)
        # ORB research shows session trends often run 3-5 ATR
        self._stop_loss_atr_mult = config.get("stop_loss_atr_mult", 2.0)
        self._take_profit_atr_mult = config.get("take_profit_atr_mult", 4.0)
        self._min_risk_reward = config.get("min_risk_reward", 1.5)

        # State tracking
        self._opening_ranges: dict[str, OpeningRange] = {}
        self._session_signals: dict[str, SessionSignal] = {}

        logger.info(
            f"SessionStrategy initialized: "
            f"opening_range={self._opening_range_minutes}min, "
            f"breakout_threshold={self._breakout_threshold_atr}ATR"
        )

    def get_current_session(
        self,
        timestamp: datetime | None = None,
    ) -> tuple[TradingSession, SessionQuality]:
        """
        Determine current trading session.

        Args:
            timestamp: Time to check (default: now UTC)

        Returns:
            (session, quality) tuple
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        current_time = timestamp.time()

        # Check for overlap first (highest priority)
        overlap = SESSION_WINDOWS["london_ny_overlap"]
        if overlap.start_time <= current_time < overlap.end_time:
            return TradingSession.LONDON_NY_OVERLAP, SessionQuality.EXCELLENT

        # Check other sessions
        for key, window in SESSION_WINDOWS.items():
            if key == "london_ny_overlap":
                continue

            # Handle overnight sessions
            if window.start_time <= current_time < window.end_time:
                return window.session, window.quality

        # After hours
        return TradingSession.AFTER_HOURS, SessionQuality.POOR

    def is_session_allowed(
        self,
        session: TradingSession,
    ) -> bool:
        """Check if session is in allowed list."""
        return session in self._allowed_sessions

    def calculate_opening_range(
        self,
        symbol: str,
        prices: np.ndarray,  # OHLCV: (n, 5) or just highs/lows
        timestamps: list[datetime],
        session: TradingSession,
        bar_minutes: int = 1,
    ) -> OpeningRange | None:
        """
        Calculate opening range for a session.

        The opening range is the high and low of the first N minutes
        after a session opens. Breakouts from this range often indicate
        the session's direction.

        Args:
            symbol: Symbol being analyzed
            prices: Price data (OHLCV format or high/low arrays)
            timestamps: Timestamps for each bar
            session: Session to calculate range for
            bar_minutes: Minutes per bar

        Returns:
            OpeningRange or None if insufficient data
        """
        if len(prices) < 2:
            return None

        # Get session window
        session_key = session.value
        if session_key not in SESSION_WINDOWS:
            # Use overlap as default
            session_key = "london_ny_overlap"

        window = SESSION_WINDOWS[session_key]

        # Find bars within opening range period
        range_bars = self._opening_range_minutes // bar_minutes
        range_bars = min(range_bars, len(prices))

        if range_bars < 2:
            return None

        # Filter to session start bars
        session_start_idx = None
        for i, ts in enumerate(timestamps):
            if ts.time() >= window.start_time:
                session_start_idx = i
                break

        if session_start_idx is None:
            session_start_idx = 0

        # Get opening range bars
        end_idx = min(session_start_idx + range_bars, len(prices))
        range_prices = prices[session_start_idx:end_idx]

        if len(range_prices) < 2:
            return None

        # Calculate range
        if range_prices.ndim == 2:
            # OHLCV format
            highs = range_prices[:, 1]
            lows = range_prices[:, 2]
            volumes = range_prices[:, 4] if range_prices.shape[1] > 4 else np.ones(len(range_prices))
        else:
            # Assume close prices
            highs = range_prices
            lows = range_prices
            volumes = np.ones(len(range_prices))

        range_high = np.max(highs)
        range_low = np.min(lows)
        range_mid = (range_high + range_low) / 2
        range_size = range_high - range_low
        range_pct = (range_size / range_mid) * 100 if range_mid > 0 else 0

        opening_range = OpeningRange(
            high=range_high,
            low=range_low,
            mid=range_mid,
            range_size=range_size,
            range_pct=range_pct,
            volume_in_range=np.sum(volumes),
            n_bars=len(range_prices),
            session=session,
        )

        # Cache it
        self._opening_ranges[symbol] = opening_range

        return opening_range

    def detect_breakout(
        self,
        symbol: str,
        current_price: float,
        atr: float,
        volume: float,
        avg_volume: float,
        opening_range: OpeningRange | None = None,
    ) -> SessionSignal | None:
        """
        Detect opening range breakout.

        Args:
            symbol: Symbol
            current_price: Current price
            atr: Average True Range
            volume: Current volume
            avg_volume: Average volume for comparison
            opening_range: Opening range (uses cached if not provided)

        Returns:
            SessionSignal if breakout detected, None otherwise
        """
        if opening_range is None:
            opening_range = self._opening_ranges.get(symbol)

        if opening_range is None:
            return None

        # Check breakout threshold
        min_breakout = atr * self._breakout_threshold_atr

        # Check volume confirmation
        volume_confirmed = volume >= avg_volume * self._volume_confirmation_mult

        # Detect direction
        direction = "NEUTRAL"
        entry_price = current_price

        if current_price > opening_range.high + min_breakout:
            direction = "LONG"
            stop_loss = opening_range.low - atr * 0.5
            take_profit = current_price + atr * self._take_profit_atr_mult

        elif current_price < opening_range.low - min_breakout:
            direction = "SHORT"
            stop_loss = opening_range.high + atr * 0.5
            take_profit = current_price - atr * self._take_profit_atr_mult

        else:
            return None

        # Validate risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        if risk_reward < self._min_risk_reward:
            return None

        # Calculate strength
        breakout_size = abs(current_price - (opening_range.high if direction == "LONG" else opening_range.low))
        strength = min(1.0, breakout_size / (atr * 2))
        if volume_confirmed:
            strength = min(1.0, strength * 1.2)

        signal = SessionSignal(
            symbol=symbol,
            session=opening_range.session,
            direction=direction,
            signal_type="breakout",
            strength=strength,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opening_range=opening_range,
            volume_confirmed=volume_confirmed,
            rationale=(
                f"Opening range breakout {direction}: "
                f"price={current_price:.2f}, range=[{opening_range.low:.2f}, {opening_range.high:.2f}], "
                f"volume_confirmed={volume_confirmed}, R:R={risk_reward:.1f}"
            ),
        )

        self._session_signals[symbol] = signal
        return signal

    def generate_session_momentum_signal(
        self,
        symbol: str,
        prices: np.ndarray,
        session: TradingSession,
        atr: float,
    ) -> SessionSignal | None:
        """
        Generate session momentum signal.

        Looks for directional moves during high-quality sessions.

        Args:
            symbol: Symbol
            prices: Recent price data (close prices)
            session: Current session
            atr: Average True Range

        Returns:
            SessionSignal if momentum detected
        """
        if len(prices) < 10:
            return None

        # Check session quality
        _, quality = self.get_current_session()
        if quality in [SessionQuality.POOR, SessionQuality.AVOID]:
            return None

        # Calculate momentum
        returns = np.diff(prices) / prices[:-1]
        recent_returns = returns[-5:]  # Last 5 bars

        # Directional consistency
        pos_count = np.sum(recent_returns > 0)
        neg_count = np.sum(recent_returns < 0)

        direction = "NEUTRAL"
        if pos_count >= 4:
            direction = "LONG"
        elif neg_count >= 4:
            direction = "SHORT"
        else:
            return None

        # Momentum strength
        total_move = np.sum(recent_returns)
        strength = min(1.0, abs(total_move) / (atr / prices[-1] * 3))

        current_price = prices[-1]

        if direction == "LONG":
            stop_loss = current_price - atr * self._stop_loss_atr_mult
            take_profit = current_price + atr * self._take_profit_atr_mult
        else:
            stop_loss = current_price + atr * self._stop_loss_atr_mult
            take_profit = current_price - atr * self._take_profit_atr_mult

        return SessionSignal(
            symbol=symbol,
            session=session,
            direction=direction,
            signal_type="momentum",
            strength=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opening_range=self._opening_ranges.get(symbol),
            volume_confirmed=False,  # Not checked for momentum
            rationale=(
                f"Session momentum {direction}: "
                f"{pos_count}/{len(recent_returns)} bars positive, "
                f"total_move={total_move:.4f}"
            ),
        )

    def analyze(
        self,
        symbol: str,
        prices: np.ndarray,
        timestamps: list[datetime],
        atr: float,
        volume: float,
        avg_volume: float,
        bar_minutes: int = 1,
    ) -> SessionSignal | None:
        """
        Main analysis entry point.

        Analyzes current market conditions and generates session-based signals.

        Args:
            symbol: Symbol to analyze
            prices: Price data (OHLCV preferred)
            timestamps: Timestamps for price data
            atr: Current ATR
            volume: Current volume
            avg_volume: Average volume
            bar_minutes: Minutes per bar

        Returns:
            SessionSignal if conditions met, None otherwise
        """
        # Get current session - use last timestamp from data for consistency
        # This allows backtesting and testing with historical timestamps
        current_timestamp = timestamps[-1] if timestamps else None
        session, quality = self.get_current_session(current_timestamp)

        # Check if session is allowed
        if not self.is_session_allowed(session):
            logger.debug(f"Session {session.value} not in allowed list")
            return None

        if quality in [SessionQuality.POOR, SessionQuality.AVOID]:
            logger.debug(f"Session quality {quality.value} too low")
            return None

        # Calculate opening range if not cached
        if symbol not in self._opening_ranges:
            self.calculate_opening_range(
                symbol, prices, timestamps, session, bar_minutes
            )

        # Get current price
        if prices.ndim == 2:
            current_price = prices[-1, 3]  # Close
        else:
            current_price = prices[-1]

        # Try breakout detection first
        signal = self.detect_breakout(
            symbol, current_price, atr, volume, avg_volume
        )

        if signal is not None:
            return signal

        # Try momentum signal as fallback
        if prices.ndim == 2:
            close_prices = prices[:, 3]
        else:
            close_prices = prices

        signal = self.generate_session_momentum_signal(
            symbol, close_prices, session, atr
        )

        return signal

    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics about current session state."""
        session, quality = self.get_current_session()

        return {
            "current_session": session.value,
            "session_quality": quality.value,
            "cached_opening_ranges": len(self._opening_ranges),
            "cached_signals": len(self._session_signals),
            "allowed_sessions": [s.value for s in self._allowed_sessions],
        }

    def clear_session_cache(self) -> None:
        """Clear cached data (call at session change)."""
        self._opening_ranges.clear()
        self._session_signals.clear()
        logger.debug("Session cache cleared")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_session_strategy(config: dict[str, Any] | None = None) -> SessionStrategy:
    """Create a SessionStrategy instance with configuration."""
    return SessionStrategy(config)

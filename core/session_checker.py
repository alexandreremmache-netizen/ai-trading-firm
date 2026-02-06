"""
Session Checker Module
======================

QUICK WIN #3: Session-Based Trading Windows

Provides functions to check if current time is within optimal trading sessions
for different asset classes and instruments.

Research shows:
- EUR/USD: Best during London/NY overlap (13:00-17:00 UTC), avoid Asian
- Futures: Best during RTH, especially 09:30-11:30 and 14:00-16:00 ET
- Commodities: Vary by product (oil during NYMEX hours, gold during London/NY)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, time
from typing import Any

logger = logging.getLogger(__name__)


# Session definitions (all in UTC)
TRADING_SESSIONS = {
    # Forex sessions
    "asian": {"start": time(0, 0), "end": time(9, 0)},      # 00:00-09:00 UTC
    "london": {"start": time(8, 0), "end": time(17, 0)},    # 08:00-17:00 UTC
    "ny": {"start": time(13, 0), "end": time(22, 0)},       # 13:00-22:00 UTC
    "ny_overlap": {"start": time(13, 0), "end": time(17, 0)},  # 13:00-17:00 UTC (best liquidity)
}

# Default session preferences by asset type
DEFAULT_SESSION_PREFS = {
    "forex": {
        "EURUSD": {"active": ["london", "ny_overlap"], "avoid": ["asian"]},
        "USDJPY": {"active": ["asian", "london", "ny"], "avoid": []},
        "GBPUSD": {"active": ["london", "ny_overlap"], "avoid": ["asian"]},
        "AUDUSD": {"active": ["asian", "london"], "avoid": []},
        "USDCAD": {"active": ["ny"], "avoid": ["asian"]},
        "USDCHF": {"active": ["london", "ny_overlap"], "avoid": ["asian"]},
    },
    "futures": {
        # US equity index futures
        # Optimal: 9:30-11:30 ET (14:30-16:30 UTC) = open + morning momentum
        #          14:00-16:00 ET (19:00-21:00 UTC) = afternoon + MOC imbalances
        # Avoid:   16:15-17:00 ET (21:15-22:00 UTC) = CME daily settlement + maintenance
        "ES": {"optimal_hours_utc": [(14, 30, 16, 30), (19, 0, 21, 0)], "avoid_hours_utc": [(21, 15, 22, 0)]},
        "MES": {"optimal_hours_utc": [(14, 30, 16, 30), (19, 0, 21, 0)], "avoid_hours_utc": [(21, 15, 22, 0)]},
        "NQ": {"optimal_hours_utc": [(14, 30, 16, 30), (19, 0, 21, 0)], "avoid_hours_utc": [(21, 15, 22, 0)]},
        "MNQ": {"optimal_hours_utc": [(14, 30, 16, 30), (19, 0, 21, 0)], "avoid_hours_utc": [(21, 15, 22, 0)]},
        "YM": {"optimal_hours_utc": [(14, 30, 16, 30), (19, 0, 21, 0)], "avoid_hours_utc": [(21, 15, 22, 0)]},
        "MYM": {"optimal_hours_utc": [(14, 30, 16, 30), (19, 0, 21, 0)], "avoid_hours_utc": [(21, 15, 22, 0)]},
        "RTY": {"optimal_hours_utc": [(14, 30, 16, 30), (19, 0, 21, 0)], "avoid_hours_utc": [(21, 15, 22, 0)]},
        "M2K": {"optimal_hours_utc": [(14, 30, 16, 30), (19, 0, 21, 0)], "avoid_hours_utc": [(21, 15, 22, 0)]},
    },
    "commodities": {
        # Energy - NYMEX hours
        "CL": {"optimal_hours_utc": [(14, 0, 19, 30)], "avoid_hours_utc": []},
        "MCL": {"optimal_hours_utc": [(14, 0, 19, 30)], "avoid_hours_utc": []},
        "NG": {"optimal_hours_utc": [(14, 30, 19, 0)], "avoid_hours_utc": []},
        # Metals - London + NY
        "GC": {"optimal_hours_utc": [(8, 0, 12, 0), (13, 30, 17, 0)], "avoid_hours_utc": []},
        "MGC": {"optimal_hours_utc": [(8, 0, 12, 0), (13, 30, 17, 0)], "avoid_hours_utc": []},
        "SI": {"optimal_hours_utc": [(8, 0, 12, 0), (13, 30, 17, 0)], "avoid_hours_utc": []},
        "SIL": {"optimal_hours_utc": [(8, 0, 12, 0), (13, 30, 17, 0)], "avoid_hours_utc": []},
    }
}


def is_in_session(session_name: str, current_time: datetime | None = None) -> bool:
    """
    Check if current time is within a named trading session.

    Args:
        session_name: Name of session (asian, london, ny, ny_overlap)
        current_time: Time to check (default: now UTC)

    Returns:
        True if within session
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)

    session = TRADING_SESSIONS.get(session_name.lower())
    if session is None:
        return True  # Unknown session, allow trading

    current_t = current_time.time()
    start = session["start"]
    end = session["end"]

    # Handle sessions that cross midnight
    if start <= end:
        return start <= current_t <= end
    else:
        return current_t >= start or current_t <= end


def is_optimal_trading_time(
    symbol: str,
    asset_class: str,
    current_time: datetime | None = None,
    config: dict[str, Any] | None = None
) -> tuple[bool, str]:
    """
    QUICK WIN #3: Check if current time is optimal for trading a symbol.

    Args:
        symbol: Trading symbol (e.g., "EURUSD", "ES", "CL")
        asset_class: Asset class (forex, futures, commodities, equity)
        current_time: Time to check (default: now UTC)
        config: Optional config with custom session settings

    Returns:
        (is_optimal, reason) - True if good time to trade, with explanation
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)

    # Normalize symbol (remove micro prefix for matching)
    base_symbol = symbol.upper().replace("MICRO", "").strip()
    if base_symbol.startswith("M") and len(base_symbol) <= 4 and base_symbol != "META":
        # Micro contracts like MES, MNQ, MCL - try to find base
        possible_base = base_symbol[1:]
        if possible_base in DEFAULT_SESSION_PREFS.get("futures", {}):
            base_symbol = possible_base

    # Get session preferences from config or defaults
    session_prefs = None
    if config and "trading_sessions" in config:
        ts_config = config["trading_sessions"]
        for asset_type in [asset_class, "forex", "futures", "commodities"]:
            if asset_type in ts_config:
                for key, prefs in ts_config[asset_type].items():
                    # Match by symbol or pattern (e.g., "es_mes" matches ES and MES)
                    if symbol.lower() in key.lower() or base_symbol.lower() in key.lower():
                        session_prefs = prefs
                        break

    # Fall back to defaults
    if session_prefs is None:
        for asset_type, symbols in DEFAULT_SESSION_PREFS.items():
            if symbol.upper() in symbols:
                session_prefs = symbols[symbol.upper()]
                break
            if base_symbol in symbols:
                session_prefs = symbols[base_symbol]
                break

    # If no preferences found, allow trading
    if session_prefs is None:
        return True, "No session preferences configured"

    # Check forex-style session preferences
    if "active" in session_prefs or "active_sessions" in session_prefs:
        active_sessions = session_prefs.get("active") or session_prefs.get("active_sessions", [])
        avoid_sessions = session_prefs.get("avoid", [])

        # Check if in any avoided session
        for session in avoid_sessions:
            if is_in_session(session, current_time):
                return False, f"Avoided session: {session}"

        # Check if in any active session
        for session in active_sessions:
            if is_in_session(session, current_time):
                return True, f"Active session: {session}"

        # Not in any active session
        return False, f"Not in active sessions: {active_sessions}"

    # Check futures/commodities-style hour ranges
    current_t = current_time.time()
    current_hour = current_t.hour
    current_minute = current_t.minute
    current_minutes = current_hour * 60 + current_minute

    # Check avoided hours first
    avoid_hours = session_prefs.get("avoid_hours_utc", [])
    for (start_h, start_m, end_h, end_m) in avoid_hours:
        start_mins = start_h * 60 + start_m
        end_mins = end_h * 60 + end_m
        if start_mins <= current_minutes <= end_mins:
            return False, f"Avoided hours: {start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d} UTC"

    # Check optimal hours
    optimal_hours = session_prefs.get("optimal_hours_utc", [])
    if optimal_hours:
        for (start_h, start_m, end_h, end_m) in optimal_hours:
            start_mins = start_h * 60 + start_m
            end_mins = end_h * 60 + end_m
            if start_mins <= current_minutes <= end_mins:
                return True, f"Optimal hours: {start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d} UTC"

        # Not in optimal hours (but not explicitly avoided)
        return False, f"Not in optimal hours"

    return True, "No time restrictions"


def get_session_quality(
    symbol: str,
    asset_class: str,
    current_time: datetime | None = None,
    config: dict[str, Any] | None = None
) -> float:
    """
    Get a quality score for the current trading session.

    Returns:
        Float from 0.0 to 1.0:
        - 1.0 = Optimal session
        - 0.5 = Acceptable but not optimal
        - 0.0 = Should avoid trading
    """
    is_optimal, reason = is_optimal_trading_time(symbol, asset_class, current_time, config)

    if "Optimal" in reason or "Active session" in reason:
        return 1.0
    elif "Avoided" in reason:
        return 0.0
    elif "Not in" in reason:
        return 0.5
    else:
        return 0.75  # No restrictions, slightly prefer caution


def filter_signal_by_session(
    symbol: str,
    asset_class: str,
    signal_strength: float,
    current_time: datetime | None = None,
    config: dict[str, Any] | None = None
) -> tuple[float, str]:
    """
    QUICK WIN #3: Adjust signal strength based on trading session quality.

    During sub-optimal sessions:
    - Reduce signal strength by 50%
    - Log warning

    During avoided sessions:
    - Set signal strength to 0
    - Log rejection

    Args:
        symbol: Trading symbol
        asset_class: Asset class
        signal_strength: Original signal strength (-1 to 1)
        current_time: Time to check
        config: Optional config

    Returns:
        (adjusted_strength, reason)
    """
    is_optimal, reason = is_optimal_trading_time(symbol, asset_class, current_time, config)
    session_quality = get_session_quality(symbol, asset_class, current_time, config)

    if session_quality == 0.0:
        logger.info(f"Signal for {symbol} rejected: {reason}")
        return 0.0, f"Session rejected: {reason}"
    elif session_quality < 1.0:
        adjusted = signal_strength * session_quality
        logger.debug(f"Signal for {symbol} reduced {signal_strength:.2f} -> {adjusted:.2f}: {reason}")
        return adjusted, f"Session adjusted ({session_quality:.0%}): {reason}"
    else:
        return signal_strength, f"Optimal session: {reason}"

"""
Time-Based Risk Module
======================

Risk limits that are time-of-day aware (Issue #R19).
Overnight vs intraday risk differentiation (Issue #R22).

Features:
- Time-of-day dependent risk limits
- Market hours awareness
- Overnight position limits
- Intraday vs overnight risk metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, time, timedelta, date
from enum import Enum
from typing import Any, Callable
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class TradingSession(str, Enum):
    """Trading session periods."""
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"  # First 30 minutes
    MORNING = "morning"
    MIDDAY = "midday"
    AFTERNOON = "afternoon"
    MARKET_CLOSE = "market_close"  # Last 30 minutes
    AFTER_HOURS = "after_hours"
    OVERNIGHT = "overnight"
    WEEKEND = "weekend"


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class SessionRiskLimits:
    """Risk limits for a trading session."""
    session: TradingSession

    # Position limits (% of normal)
    max_position_pct: float = 100.0
    max_new_orders_pct: float = 100.0

    # Risk limits
    max_var_pct: float = 100.0
    max_concentration_pct: float = 100.0

    # Execution limits
    min_order_interval_ms: float = 0.0
    max_market_orders: bool = True

    # Volatility adjustment
    vol_multiplier: float = 1.0

    def to_dict(self) -> dict:
        return {
            'session': self.session.value,
            'max_position_pct': self.max_position_pct,
            'max_new_orders_pct': self.max_new_orders_pct,
            'max_var_pct': self.max_var_pct,
            'max_concentration_pct': self.max_concentration_pct,
            'min_order_interval_ms': self.min_order_interval_ms,
            'max_market_orders': self.max_market_orders,
            'vol_multiplier': self.vol_multiplier,
        }


@dataclass
class OvernightRiskMetrics:
    """Overnight risk metrics (#R22)."""
    as_of: datetime

    # Position summary
    total_overnight_exposure: float
    long_exposure: float
    short_exposure: float
    net_exposure: float

    # Risk metrics
    overnight_var: float
    overnight_var_pct: float  # As % of account

    # Gap risk
    estimated_gap_risk: float
    gap_risk_pct: float

    # By asset class
    exposure_by_asset: dict[str, float]
    var_by_asset: dict[str, float]

    # Compared to intraday
    exposure_vs_intraday_avg: float  # Ratio
    var_vs_intraday_avg: float

    def to_dict(self) -> dict:
        return {
            'as_of': self.as_of.isoformat(),
            'total_overnight_exposure': self.total_overnight_exposure,
            'long_exposure': self.long_exposure,
            'short_exposure': self.short_exposure,
            'net_exposure': self.net_exposure,
            'overnight_var': self.overnight_var,
            'overnight_var_pct': self.overnight_var_pct,
            'estimated_gap_risk': self.estimated_gap_risk,
            'gap_risk_pct': self.gap_risk_pct,
            'exposure_by_asset': self.exposure_by_asset,
            'var_by_asset': self.var_by_asset,
        }


@dataclass
class IntradayRiskMetrics:
    """Intraday risk metrics (#R22)."""
    as_of: datetime
    session: TradingSession

    # Current exposure
    current_exposure: float
    peak_exposure_today: float
    avg_exposure_today: float

    # Current risk
    current_var: float
    peak_var_today: float

    # Trading activity
    orders_today: int
    trades_today: int
    pnl_today: float

    # Session context
    time_to_close_minutes: float
    vol_regime: str

    def to_dict(self) -> dict:
        return {
            'as_of': self.as_of.isoformat(),
            'session': self.session.value,
            'current_exposure': self.current_exposure,
            'peak_exposure_today': self.peak_exposure_today,
            'avg_exposure_today': self.avg_exposure_today,
            'current_var': self.current_var,
            'peak_var_today': self.peak_var_today,
            'orders_today': self.orders_today,
            'trades_today': self.trades_today,
            'pnl_today': self.pnl_today,
            'time_to_close_minutes': self.time_to_close_minutes,
        }


class TimeBasedRiskManager:
    """
    Manages time-of-day dependent risk limits (#R19).

    Adjusts risk limits based on market session and time.
    """

    # Default session times (US Eastern)
    SESSION_TIMES = {
        TradingSession.PRE_MARKET: (time(4, 0), time(9, 30)),
        TradingSession.MARKET_OPEN: (time(9, 30), time(10, 0)),
        TradingSession.MORNING: (time(10, 0), time(12, 0)),
        TradingSession.MIDDAY: (time(12, 0), time(14, 0)),
        TradingSession.AFTERNOON: (time(14, 0), time(15, 30)),
        TradingSession.MARKET_CLOSE: (time(15, 30), time(16, 0)),
        TradingSession.AFTER_HOURS: (time(16, 0), time(20, 0)),
    }

    # Default limits by session
    DEFAULT_LIMITS = {
        TradingSession.PRE_MARKET: SessionRiskLimits(
            session=TradingSession.PRE_MARKET,
            max_position_pct=50.0,
            max_new_orders_pct=50.0,
            max_var_pct=75.0,
            max_market_orders=False,  # Limit orders only
            min_order_interval_ms=500.0,
            vol_multiplier=1.5,
        ),
        TradingSession.MARKET_OPEN: SessionRiskLimits(
            session=TradingSession.MARKET_OPEN,
            max_position_pct=75.0,
            max_new_orders_pct=75.0,
            max_var_pct=80.0,
            min_order_interval_ms=100.0,
            vol_multiplier=1.3,
        ),
        TradingSession.MORNING: SessionRiskLimits(
            session=TradingSession.MORNING,
            max_position_pct=100.0,
            max_new_orders_pct=100.0,
            max_var_pct=100.0,
            vol_multiplier=1.0,
        ),
        TradingSession.MIDDAY: SessionRiskLimits(
            session=TradingSession.MIDDAY,
            max_position_pct=100.0,
            max_new_orders_pct=100.0,
            max_var_pct=100.0,
            vol_multiplier=0.9,  # Typically lower vol
        ),
        TradingSession.AFTERNOON: SessionRiskLimits(
            session=TradingSession.AFTERNOON,
            max_position_pct=100.0,
            max_new_orders_pct=100.0,
            max_var_pct=100.0,
            vol_multiplier=1.1,
        ),
        TradingSession.MARKET_CLOSE: SessionRiskLimits(
            session=TradingSession.MARKET_CLOSE,
            max_position_pct=80.0,  # Reduce near close
            max_new_orders_pct=60.0,
            max_var_pct=90.0,
            min_order_interval_ms=200.0,
            vol_multiplier=1.4,
        ),
        TradingSession.AFTER_HOURS: SessionRiskLimits(
            session=TradingSession.AFTER_HOURS,
            max_position_pct=30.0,
            max_new_orders_pct=30.0,
            max_var_pct=50.0,
            max_market_orders=False,
            min_order_interval_ms=1000.0,
            vol_multiplier=2.0,
        ),
        TradingSession.OVERNIGHT: SessionRiskLimits(
            session=TradingSession.OVERNIGHT,
            max_position_pct=40.0,
            max_new_orders_pct=0.0,  # No new orders
            max_var_pct=60.0,
            max_market_orders=False,
        ),
        TradingSession.WEEKEND: SessionRiskLimits(
            session=TradingSession.WEEKEND,
            max_position_pct=30.0,
            max_new_orders_pct=0.0,
            max_var_pct=50.0,
            max_market_orders=False,
        ),
    }

    def __init__(
        self,
        base_var_limit: float = 100_000.0,
        base_position_limit: float = 1_000_000.0,
        timezone_str: str = "America/New_York",
    ):
        self.base_var_limit = base_var_limit
        self.base_position_limit = base_position_limit
        self.tz = ZoneInfo(timezone_str)

        # Custom limits override
        self._custom_limits: dict[TradingSession, SessionRiskLimits] = {}

        # Event dates (holidays, half days)
        self._holidays: set[date] = set()
        self._half_days: set[date] = set()  # Early close at 1pm ET

    def add_holiday(self, holiday: date) -> None:
        """Add a market holiday."""
        self._holidays.add(holiday)

    def add_half_day(self, half_day: date) -> None:
        """Add a half-day (early close)."""
        self._half_days.add(half_day)

    def set_custom_limits(self, session: TradingSession, limits: SessionRiskLimits) -> None:
        """Override default limits for a session."""
        self._custom_limits[session] = limits

    def get_current_session(self, dt: datetime | None = None) -> TradingSession:
        """Get current trading session."""
        if dt is None:
            dt = datetime.now(self.tz)
        else:
            dt = dt.astimezone(self.tz)

        current_date = dt.date()
        current_time = dt.time()

        # Check weekend
        if dt.weekday() >= 5:  # Saturday or Sunday
            return TradingSession.WEEKEND

        # Check holiday
        if current_date in self._holidays:
            return TradingSession.WEEKEND

        # Check half day
        if current_date in self._half_days:
            if current_time >= time(13, 0):
                return TradingSession.AFTER_HOURS

        # Check overnight (20:00 - 04:00)
        if current_time >= time(20, 0) or current_time < time(4, 0):
            return TradingSession.OVERNIGHT

        # Check regular sessions
        for session, (start, end) in self.SESSION_TIMES.items():
            if start <= current_time < end:
                return session

        return TradingSession.OVERNIGHT

    def get_session_limits(
        self,
        session: TradingSession | None = None,
    ) -> SessionRiskLimits:
        """Get risk limits for a session."""
        if session is None:
            session = self.get_current_session()

        # Check custom override
        if session in self._custom_limits:
            return self._custom_limits[session]

        return self.DEFAULT_LIMITS.get(session, self.DEFAULT_LIMITS[TradingSession.OVERNIGHT])

    def get_effective_limits(
        self,
        dt: datetime | None = None,
    ) -> dict:
        """Get effective risk limits for current time."""
        session = self.get_current_session(dt)
        limits = self.get_session_limits(session)

        return {
            'session': session.value,
            'timestamp': (dt or datetime.now(self.tz)).isoformat(),
            'max_position': self.base_position_limit * limits.max_position_pct / 100,
            'max_new_orders_exposure': self.base_position_limit * limits.max_new_orders_pct / 100,
            'max_var': self.base_var_limit * limits.max_var_pct / 100,
            'vol_multiplier': limits.vol_multiplier,
            'min_order_interval_ms': limits.min_order_interval_ms,
            'market_orders_allowed': limits.max_market_orders,
            'limits_detail': limits.to_dict(),
        }

    def check_limit(
        self,
        current_position: float,
        current_var: float,
        proposed_order_size: float,
        is_market_order: bool = False,
    ) -> dict:
        """
        Check if proposed action is within time-based limits.

        Returns dict with 'allowed' and any violations.
        """
        session = self.get_current_session()
        limits = self.get_session_limits(session)
        effective = self.get_effective_limits()

        violations = []

        # Position limit
        max_pos = effective['max_position']
        new_pos = abs(current_position) + abs(proposed_order_size)
        if new_pos > max_pos:
            violations.append(f"Position would exceed {session.value} limit (${new_pos:,.0f} > ${max_pos:,.0f})")

        # VaR limit
        max_var = effective['max_var']
        if current_var > max_var:
            violations.append(f"VaR exceeds {session.value} limit (${current_var:,.0f} > ${max_var:,.0f})")

        # Market order check
        if is_market_order and not limits.max_market_orders:
            violations.append(f"Market orders not allowed during {session.value}")

        # New orders limit
        max_new = effective['max_new_orders_exposure']
        if proposed_order_size > max_new:
            violations.append(f"Order size exceeds {session.value} limit (${proposed_order_size:,.0f} > ${max_new:,.0f})")

        return {
            'allowed': len(violations) == 0,
            'session': session.value,
            'violations': violations,
            'effective_limits': effective,
        }

    def get_time_to_session_change(self) -> timedelta:
        """Get time until next session change."""
        now = datetime.now(self.tz)
        current_time = now.time()

        for session, (start, end) in self.SESSION_TIMES.items():
            if start <= current_time < end:
                # In this session, return time to end
                end_dt = now.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
                return end_dt - now

        # Overnight - return time to pre-market
        pre_start = self.SESSION_TIMES[TradingSession.PRE_MARKET][0]
        if current_time >= time(20, 0):
            # After 8pm, next pre-market is tomorrow
            tomorrow = now.date() + timedelta(days=1)
            pre_dt = datetime.combine(tomorrow, pre_start, tzinfo=self.tz)
        else:
            # Before 4am, pre-market is today
            pre_dt = now.replace(hour=pre_start.hour, minute=pre_start.minute, second=0, microsecond=0)

        return pre_dt - now


class OvernightRiskManager:
    """
    Manages overnight vs intraday risk (#R22).

    Differentiates risk treatment based on holding period.
    """

    def __init__(
        self,
        overnight_var_multiplier: float = 1.5,  # Overnight VaR is higher
        gap_risk_pct: float = 3.0,  # Assume 3% gap risk for equities
        max_overnight_exposure_pct: float = 50.0,  # Max % of account in overnight
        timezone_str: str = "America/New_York",
    ):
        self.overnight_var_mult = overnight_var_multiplier
        self.gap_risk_pct = gap_risk_pct / 100
        self.max_overnight_pct = max_overnight_exposure_pct
        self.tz = ZoneInfo(timezone_str)

        # Positions
        self._positions: dict[str, dict] = {}  # symbol -> {qty, value, asset_class, beta}

        # Intraday tracking
        self._intraday_exposures: list[tuple[datetime, float]] = []
        self._peak_exposure_today: float = 0.0
        self._peak_var_today: float = 0.0
        self._orders_today: int = 0
        self._trades_today: int = 0
        self._pnl_today: float = 0.0
        self._last_reset: date | None = None

        # Gap risk by asset class
        self._gap_risk_rates: dict[str, float] = {
            'equity': 0.03,  # 3%
            'etf': 0.02,
            'futures': 0.04,
            'options': 0.10,
            'fx': 0.01,
            'commodity': 0.05,
        }

    def update_position(
        self,
        symbol: str,
        quantity: int,
        value: float,
        asset_class: str = "equity",
        beta: float = 1.0,
    ) -> None:
        """Update position for overnight tracking."""
        self._positions[symbol] = {
            'quantity': quantity,
            'value': value,
            'asset_class': asset_class,
            'beta': beta,
        }

    def record_trade(self, pnl: float) -> None:
        """Record a trade execution."""
        self._trades_today += 1
        self._pnl_today += pnl

    def record_order(self) -> None:
        """Record an order submission."""
        self._orders_today += 1

    def record_exposure(self, exposure: float, var: float) -> None:
        """Record current exposure for intraday tracking."""
        now = datetime.now(self.tz)

        # Reset at start of new day
        if self._last_reset != now.date():
            self._reset_daily_tracking()
            self._last_reset = now.date()

        self._intraday_exposures.append((now, exposure))
        self._peak_exposure_today = max(self._peak_exposure_today, exposure)
        self._peak_var_today = max(self._peak_var_today, var)

    def _reset_daily_tracking(self) -> None:
        """Reset daily tracking metrics."""
        self._intraday_exposures.clear()
        self._peak_exposure_today = 0.0
        self._peak_var_today = 0.0
        self._orders_today = 0
        self._trades_today = 0
        self._pnl_today = 0.0

    def calculate_overnight_risk(
        self,
        account_equity: float,
    ) -> OvernightRiskMetrics:
        """Calculate overnight risk metrics."""
        now = datetime.now(self.tz)

        # Calculate exposures
        long_exposure = sum(
            p['value'] for p in self._positions.values()
            if p['quantity'] > 0
        )
        short_exposure = abs(sum(
            p['value'] for p in self._positions.values()
            if p['quantity'] < 0
        ))
        total_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure

        # Calculate by asset class
        exposure_by_asset: dict[str, float] = {}
        var_by_asset: dict[str, float] = {}

        for symbol, pos in self._positions.items():
            asset = pos['asset_class']
            exposure_by_asset[asset] = exposure_by_asset.get(asset, 0) + abs(pos['value'])

        # Calculate overnight VaR (with multiplier)
        base_var = total_exposure * 0.02  # Simplified 2% base VaR
        overnight_var = base_var * self.overnight_var_mult

        # Calculate gap risk
        total_gap_risk = 0.0
        for asset, exposure in exposure_by_asset.items():
            gap_rate = self._gap_risk_rates.get(asset, 0.03)
            gap_risk = exposure * gap_rate
            var_by_asset[asset] = gap_risk
            total_gap_risk += gap_risk

        # Calculate ratios vs intraday
        if self._intraday_exposures:
            avg_intraday = sum(e[1] for e in self._intraday_exposures) / len(self._intraday_exposures)
            exposure_ratio = total_exposure / avg_intraday if avg_intraday > 0 else 1.0
        else:
            exposure_ratio = 1.0

        return OvernightRiskMetrics(
            as_of=now,
            total_overnight_exposure=total_exposure,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            net_exposure=net_exposure,
            overnight_var=overnight_var,
            overnight_var_pct=overnight_var / account_equity * 100 if account_equity > 0 else 0,
            estimated_gap_risk=total_gap_risk,
            gap_risk_pct=total_gap_risk / account_equity * 100 if account_equity > 0 else 0,
            exposure_by_asset=exposure_by_asset,
            var_by_asset=var_by_asset,
            exposure_vs_intraday_avg=exposure_ratio,
            var_vs_intraday_avg=overnight_var / (base_var if base_var > 0 else 1.0),
        )

    def calculate_intraday_risk(
        self,
        current_exposure: float,
        current_var: float,
    ) -> IntradayRiskMetrics:
        """Calculate intraday risk metrics."""
        now = datetime.now(self.tz)

        # Get session
        time_mgr = TimeBasedRiskManager(timezone_str=str(self.tz))
        session = time_mgr.get_current_session(now)

        # Calculate average exposure
        if self._intraday_exposures:
            avg_exposure = sum(e[1] for e in self._intraday_exposures) / len(self._intraday_exposures)
        else:
            avg_exposure = current_exposure

        # Time to close
        market_close = time(16, 0)
        if now.time() < market_close:
            close_dt = now.replace(hour=16, minute=0, second=0, microsecond=0)
            time_to_close = (close_dt - now).total_seconds() / 60
        else:
            time_to_close = 0

        # Volatility regime (simplified)
        if current_var > self._peak_var_today * 0.8:
            vol_regime = "high"
        elif current_var > self._peak_var_today * 0.5:
            vol_regime = "elevated"
        else:
            vol_regime = "normal"

        return IntradayRiskMetrics(
            as_of=now,
            session=session,
            current_exposure=current_exposure,
            peak_exposure_today=self._peak_exposure_today,
            avg_exposure_today=avg_exposure,
            current_var=current_var,
            peak_var_today=self._peak_var_today,
            orders_today=self._orders_today,
            trades_today=self._trades_today,
            pnl_today=self._pnl_today,
            time_to_close_minutes=time_to_close,
            vol_regime=vol_regime,
        )

    def check_overnight_limits(
        self,
        account_equity: float,
    ) -> dict:
        """Check if overnight position limits are exceeded."""
        metrics = self.calculate_overnight_risk(account_equity)

        violations = []

        # Max overnight exposure
        max_overnight = account_equity * self.max_overnight_pct / 100
        if metrics.total_overnight_exposure > max_overnight:
            violations.append(
                f"Overnight exposure exceeds limit "
                f"(${metrics.total_overnight_exposure:,.0f} > ${max_overnight:,.0f})"
            )

        # Gap risk check
        max_gap_pct = 5.0  # Max 5% gap risk
        if metrics.gap_risk_pct > max_gap_pct:
            violations.append(
                f"Overnight gap risk too high ({metrics.gap_risk_pct:.1f}% > {max_gap_pct}%)"
            )

        return {
            'within_limits': len(violations) == 0,
            'violations': violations,
            'metrics': metrics.to_dict(),
            'reduction_needed': max(0, metrics.total_overnight_exposure - max_overnight),
        }

    def suggest_position_reduction(
        self,
        account_equity: float,
    ) -> list[dict]:
        """Suggest positions to reduce for overnight limits."""
        metrics = self.calculate_overnight_risk(account_equity)
        max_overnight = account_equity * self.max_overnight_pct / 100

        if metrics.total_overnight_exposure <= max_overnight:
            return []

        reduction_needed = metrics.total_overnight_exposure - max_overnight

        # Sort positions by risk (gap risk contribution)
        positions_with_risk = []
        for symbol, pos in self._positions.items():
            gap_rate = self._gap_risk_rates.get(pos['asset_class'], 0.03)
            risk_contribution = abs(pos['value']) * gap_rate
            positions_with_risk.append({
                'symbol': symbol,
                'value': pos['value'],
                'risk_contribution': risk_contribution,
                'asset_class': pos['asset_class'],
            })

        # Sort by risk contribution (highest first)
        positions_with_risk.sort(key=lambda x: x['risk_contribution'], reverse=True)

        suggestions = []
        remaining_reduction = reduction_needed

        for pos in positions_with_risk:
            if remaining_reduction <= 0:
                break

            reduce_by = min(abs(pos['value']), remaining_reduction)
            reduce_pct = reduce_by / abs(pos['value']) * 100

            suggestions.append({
                'symbol': pos['symbol'],
                'current_value': pos['value'],
                'reduce_by': reduce_by,
                'reduce_pct': reduce_pct,
                'reason': f"High overnight risk contribution ({pos['asset_class']})",
            })

            remaining_reduction -= reduce_by

        return suggestions

"""
Session Risk Manager
====================

Phase 2: Risk Management Enhancement

Integrates trading session quality with risk management:
- Session-specific risk limits
- Performance tracking by session
- Dynamic position sizing based on session quality
- Session-based trade filtering

Research basis:
- London/NY overlap: Highest liquidity, tightest spreads
- Asian session: Lower liquidity for EUR pairs
- RTH hours: Better execution for equity futures
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, time, timedelta
from enum import Enum
from typing import Any

from core.session_checker import (
    is_optimal_trading_time,
    get_session_quality,
    filter_signal_by_session,
    is_in_session,
    TRADING_SESSIONS,
)

logger = logging.getLogger(__name__)


class SessionRiskLevel(Enum):
    """Risk level classification based on session."""
    OPTIMAL = "optimal"         # Best conditions, full risk allocation
    ACCEPTABLE = "acceptable"   # Reduced position sizes
    MARGINAL = "marginal"       # Minimal positions only
    AVOID = "avoid"             # No new positions


@dataclass
class SessionRiskConfig:
    """Configuration for session-based risk management."""
    # Position size multipliers by risk level
    optimal_position_multiplier: float = 1.0
    acceptable_position_multiplier: float = 0.75
    marginal_position_multiplier: float = 0.5
    avoid_position_multiplier: float = 0.0

    # Max leverage by session quality
    optimal_max_leverage: float = 2.0
    acceptable_max_leverage: float = 1.5
    marginal_max_leverage: float = 1.0

    # Stop loss adjustments
    optimal_stop_multiplier: float = 1.0
    marginal_stop_multiplier: float = 1.3  # Wider stops in poor sessions

    # Session-specific parameters
    require_optimal_for_new_positions: bool = False
    block_trades_outside_sessions: bool = False
    track_session_performance: bool = True


@dataclass
class SessionPerformance:
    """Performance metrics for a trading session."""
    session_name: str
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    total_pnl: float = 0.0
    avg_trade_duration_minutes: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count

    @property
    def avg_pnl(self) -> float:
        """Calculate average P&L per trade."""
        if self.trade_count == 0:
            return 0.0
        return self.total_pnl / self.trade_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_name": self.session_name,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "avg_pnl": self.avg_pnl,
            "avg_trade_duration_minutes": self.avg_trade_duration_minutes,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class SessionRiskAssessment:
    """Result of session risk assessment."""
    symbol: str
    asset_class: str
    timestamp: datetime
    session_name: str | None
    risk_level: SessionRiskLevel
    position_multiplier: float
    max_leverage: float
    stop_multiplier: float
    is_optimal_time: bool
    reason: str
    session_performance: SessionPerformance | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "timestamp": self.timestamp.isoformat(),
            "session_name": self.session_name,
            "risk_level": self.risk_level.value,
            "position_multiplier": self.position_multiplier,
            "max_leverage": self.max_leverage,
            "stop_multiplier": self.stop_multiplier,
            "is_optimal_time": self.is_optimal_time,
            "reason": self.reason,
            "session_performance": self.session_performance.to_dict() if self.session_performance else None,
        }


class SessionRiskManager:
    """
    Manages risk parameters based on trading sessions.

    Integrates session quality with risk management for:
    - Position sizing adjustments
    - Leverage limits
    - Stop loss widths
    - Trade filtering

    Usage:
        manager = SessionRiskManager(config)
        assessment = manager.assess_session_risk("EURUSD", "forex")
        if assessment.risk_level != SessionRiskLevel.AVOID:
            position_size = base_size * assessment.position_multiplier
    """

    def __init__(
        self,
        config: SessionRiskConfig | None = None,
        session_config: dict[str, Any] | None = None,
    ):
        """
        Initialize session risk manager.

        Args:
            config: SessionRiskConfig for risk parameters
            session_config: Optional session preferences from main config
        """
        self._config = config or SessionRiskConfig()
        self._session_config = session_config or {}

        # Session performance tracking
        self._session_performance: dict[str, SessionPerformance] = {}
        for session_name in TRADING_SESSIONS:
            self._session_performance[session_name] = SessionPerformance(
                session_name=session_name
            )

        logger.info(
            f"SessionRiskManager initialized: "
            f"track_performance={self._config.track_session_performance}, "
            f"block_outside_sessions={self._config.block_trades_outside_sessions}"
        )

    def assess_session_risk(
        self,
        symbol: str,
        asset_class: str,
        current_time: datetime | None = None,
    ) -> SessionRiskAssessment:
        """
        Assess risk parameters for current trading session.

        Args:
            symbol: Trading symbol
            asset_class: Asset class (forex, futures, commodities, equity)
            current_time: Time to assess (default: now UTC)

        Returns:
            SessionRiskAssessment with risk parameters
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Get session quality
        is_optimal, reason = is_optimal_trading_time(
            symbol, asset_class, current_time, self._session_config
        )
        session_quality = get_session_quality(
            symbol, asset_class, current_time, self._session_config
        )

        # Determine current session
        current_session = self._get_current_session(current_time)

        # Map quality to risk level
        if session_quality >= 1.0:
            risk_level = SessionRiskLevel.OPTIMAL
        elif session_quality >= 0.75:
            risk_level = SessionRiskLevel.ACCEPTABLE
        elif session_quality > 0.0:
            risk_level = SessionRiskLevel.MARGINAL
        else:
            risk_level = SessionRiskLevel.AVOID

        # Get risk parameters
        position_mult, max_lev, stop_mult = self._get_risk_parameters(risk_level)

        # Get session performance if tracking
        session_perf = None
        if self._config.track_session_performance and current_session:
            session_perf = self._session_performance.get(current_session)

            # Adjust multiplier based on historical performance
            if session_perf and session_perf.trade_count >= 10:
                if session_perf.win_rate < 0.4 and session_perf.avg_pnl < 0:
                    # Poor historical performance in this session
                    position_mult *= 0.7
                    logger.info(
                        f"Reducing position size for {symbol} in {current_session} "
                        f"due to poor historical performance: WR={session_perf.win_rate:.0%}"
                    )
                elif session_perf.win_rate > 0.6 and session_perf.avg_pnl > 0:
                    # Good historical performance
                    position_mult = min(1.0, position_mult * 1.1)

        return SessionRiskAssessment(
            symbol=symbol,
            asset_class=asset_class,
            timestamp=current_time,
            session_name=current_session,
            risk_level=risk_level,
            position_multiplier=position_mult,
            max_leverage=max_lev,
            stop_multiplier=stop_mult,
            is_optimal_time=is_optimal,
            reason=reason,
            session_performance=session_perf,
        )

    def _get_risk_parameters(
        self,
        risk_level: SessionRiskLevel
    ) -> tuple[float, float, float]:
        """Get risk parameters for a risk level."""
        if risk_level == SessionRiskLevel.OPTIMAL:
            return (
                self._config.optimal_position_multiplier,
                self._config.optimal_max_leverage,
                self._config.optimal_stop_multiplier,
            )
        elif risk_level == SessionRiskLevel.ACCEPTABLE:
            return (
                self._config.acceptable_position_multiplier,
                self._config.acceptable_max_leverage,
                self._config.optimal_stop_multiplier,
            )
        elif risk_level == SessionRiskLevel.MARGINAL:
            return (
                self._config.marginal_position_multiplier,
                self._config.marginal_max_leverage,
                self._config.marginal_stop_multiplier,
            )
        else:  # AVOID
            return (
                self._config.avoid_position_multiplier,
                1.0,
                self._config.marginal_stop_multiplier,
            )

    def _get_current_session(self, current_time: datetime) -> str | None:
        """Determine which trading session is currently active."""
        for session_name in ["ny_overlap", "ny", "london", "asian"]:
            if is_in_session(session_name, current_time):
                return session_name
        return None

    def should_allow_trade(
        self,
        symbol: str,
        asset_class: str,
        current_time: datetime | None = None,
    ) -> tuple[bool, str]:
        """
        Check if a trade should be allowed based on session.

        Args:
            symbol: Trading symbol
            asset_class: Asset class
            current_time: Time to check

        Returns:
            (allowed, reason) tuple
        """
        assessment = self.assess_session_risk(symbol, asset_class, current_time)

        if assessment.risk_level == SessionRiskLevel.AVOID:
            return False, f"Trade blocked: {assessment.reason}"

        if self._config.require_optimal_for_new_positions:
            if assessment.risk_level != SessionRiskLevel.OPTIMAL:
                return False, f"Optimal session required: {assessment.reason}"

        if self._config.block_trades_outside_sessions:
            if assessment.session_name is None:
                return False, "Trade blocked: Outside defined trading sessions"

        return True, f"Trade allowed: {assessment.reason}"

    def adjust_position_size(
        self,
        symbol: str,
        asset_class: str,
        base_position_size: float,
        current_time: datetime | None = None,
    ) -> tuple[float, str]:
        """
        Adjust position size based on session quality.

        Args:
            symbol: Trading symbol
            asset_class: Asset class
            base_position_size: Base position size before adjustment
            current_time: Time to check

        Returns:
            (adjusted_size, reason) tuple
        """
        assessment = self.assess_session_risk(symbol, asset_class, current_time)
        adjusted_size = base_position_size * assessment.position_multiplier

        return adjusted_size, (
            f"Position adjusted by {assessment.position_multiplier:.0%} "
            f"({assessment.risk_level.value}): {assessment.reason}"
        )

    def record_trade_result(
        self,
        session_name: str,
        pnl: float,
        duration_minutes: float,
    ) -> None:
        """
        Record trade result for session performance tracking.

        Args:
            session_name: Name of session when trade was opened
            pnl: Trade P&L (positive = win)
            duration_minutes: Trade duration in minutes
        """
        if not self._config.track_session_performance:
            return

        if session_name not in self._session_performance:
            self._session_performance[session_name] = SessionPerformance(
                session_name=session_name
            )

        perf = self._session_performance[session_name]
        perf.trade_count += 1
        perf.total_pnl += pnl

        if pnl > 0:
            perf.win_count += 1
        else:
            perf.loss_count += 1

        # Update average duration (rolling average)
        if perf.avg_trade_duration_minutes == 0:
            perf.avg_trade_duration_minutes = duration_minutes
        else:
            perf.avg_trade_duration_minutes = (
                perf.avg_trade_duration_minutes * 0.9 + duration_minutes * 0.1
            )

        perf.last_updated = datetime.now(timezone.utc)

        logger.debug(
            f"Recorded trade in {session_name}: PnL=${pnl:.2f}, "
            f"cumulative WR={perf.win_rate:.1%}"
        )

    def get_session_performance(self) -> dict[str, dict]:
        """Get performance statistics for all sessions."""
        return {
            name: perf.to_dict()
            for name, perf in self._session_performance.items()
        }

    def get_best_sessions(
        self,
        symbol: str | None = None,
        min_trades: int = 5
    ) -> list[str]:
        """
        Get ranked list of best performing sessions.

        Args:
            symbol: Optional symbol filter (not yet implemented)
            min_trades: Minimum trades to include in ranking

        Returns:
            List of session names ranked by performance
        """
        ranked = []
        for name, perf in self._session_performance.items():
            if perf.trade_count >= min_trades:
                # Score based on win rate and average P&L
                score = perf.win_rate * 0.5 + (1.0 if perf.avg_pnl > 0 else 0.0) * 0.5
                ranked.append((name, score, perf))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _, _ in ranked]

    def get_status(self) -> dict[str, Any]:
        """Get manager status for monitoring."""
        current_time = datetime.now(timezone.utc)
        current_session = self._get_current_session(current_time)

        return {
            "enabled": True,
            "current_session": current_session,
            "current_time_utc": current_time.isoformat(),
            "config": {
                "require_optimal": self._config.require_optimal_for_new_positions,
                "block_outside": self._config.block_trades_outside_sessions,
                "track_performance": self._config.track_session_performance,
            },
            "session_performance": self.get_session_performance(),
            "best_sessions": self.get_best_sessions(),
        }


# =============================================================================
# Helper Functions
# =============================================================================

def create_session_risk_manager(config: dict[str, Any]) -> SessionRiskManager:
    """
    Factory function to create SessionRiskManager from config.

    Args:
        config: Configuration dictionary with session_risk section

    Returns:
        Configured SessionRiskManager
    """
    session_risk_config = config.get("session_risk", {})

    risk_config = SessionRiskConfig(
        optimal_position_multiplier=session_risk_config.get(
            "optimal_position_multiplier", 1.0
        ),
        acceptable_position_multiplier=session_risk_config.get(
            "acceptable_position_multiplier", 0.75
        ),
        marginal_position_multiplier=session_risk_config.get(
            "marginal_position_multiplier", 0.5
        ),
        avoid_position_multiplier=session_risk_config.get(
            "avoid_position_multiplier", 0.0
        ),
        require_optimal_for_new_positions=session_risk_config.get(
            "require_optimal_for_new_positions", False
        ),
        block_trades_outside_sessions=session_risk_config.get(
            "block_trades_outside_sessions", False
        ),
        track_session_performance=session_risk_config.get(
            "track_session_performance", True
        ),
    )

    session_config = config.get("trading_sessions", {})

    return SessionRiskManager(
        config=risk_config,
        session_config=session_config,
    )

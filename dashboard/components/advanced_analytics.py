"""
Advanced Analytics Dashboard Components (Phase 8)
==================================================

Dashboard components for advanced performance analytics:
- 8.1 Rolling Sharpe/Sortino Display
- 8.2 Win Rate by Session Panel
- 8.3 Strategy Performance Comparison
- 8.4 Risk Visualization Heatmap
- 8.5 Trade Journal Integration
- 8.6 Signal Consensus Heatmap

MATURITY: ALPHA
---------------
Status: New implementation
- [x] Rolling metrics calculation
- [x] Session performance tracking
- [x] Strategy comparison
- [x] Risk heatmap data
- [x] Trade journal entries
- [x] Signal consensus
- [ ] Frontend integration
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


# =============================================================================
# 8.1 ROLLING SHARPE/SORTINO DISPLAY
# =============================================================================

class RollingPeriod(Enum):
    """Time periods for rolling calculations."""
    DAY_1 = "1D"
    WEEK_1 = "1W"
    MONTH_1 = "1M"
    MONTH_3 = "3M"
    YTD = "YTD"
    YEAR_1 = "1Y"


@dataclass
class RollingMetrics:
    """Rolling performance metrics."""
    period: RollingPeriod
    sharpe_ratio: float | None
    sortino_ratio: float | None
    calmar_ratio: float | None
    total_return_pct: float
    volatility_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float | None
    num_trades: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period.value,
            "sharpe_ratio": round(self.sharpe_ratio, 2) if self.sharpe_ratio else None,
            "sortino_ratio": round(self.sortino_ratio, 2) if self.sortino_ratio else None,
            "calmar_ratio": round(self.calmar_ratio, 2) if self.calmar_ratio else None,
            "total_return_pct": round(self.total_return_pct, 2),
            "volatility_pct": round(self.volatility_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "win_rate_pct": round(self.win_rate_pct, 1),
            "profit_factor": round(self.profit_factor, 2) if self.profit_factor else None,
            "num_trades": self.num_trades,
            "timestamp": self.timestamp.isoformat(),
        }


class RollingMetricsCalculator:
    """
    Calculator for rolling performance metrics (Phase 8.1).

    Calculates Sharpe, Sortino, and other metrics over multiple time periods.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino
        """
        self._risk_free_rate = risk_free_rate
        self._equity_history: list[tuple[datetime, float]] = []
        self._trade_history: list[dict[str, Any]] = []

        logger.info(f"RollingMetricsCalculator initialized: rf={risk_free_rate}")

    def add_equity_point(self, timestamp: datetime, equity: float) -> None:
        """Record equity value at timestamp."""
        self._equity_history.append((timestamp, equity))
        # Keep last year of data
        cutoff = datetime.now(timezone.utc) - timedelta(days=400)
        self._equity_history = [
            (t, e) for t, e in self._equity_history
            if t > cutoff
        ]

    def add_trade(self, trade: dict[str, Any]) -> None:
        """Record completed trade."""
        self._trade_history.append(trade)
        # Keep last year of trades
        cutoff = datetime.now(timezone.utc) - timedelta(days=400)
        self._trade_history = [
            t for t in self._trade_history
            if t.get("close_time", datetime.now(timezone.utc)) > cutoff
        ]

    def _get_period_days(self, period: RollingPeriod) -> int:
        """Convert period to days."""
        mapping = {
            RollingPeriod.DAY_1: 1,
            RollingPeriod.WEEK_1: 7,
            RollingPeriod.MONTH_1: 30,
            RollingPeriod.MONTH_3: 90,
            RollingPeriod.YTD: (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
            RollingPeriod.YEAR_1: 365,
        }
        return mapping.get(period, 30)

    def calculate_metrics(self, period: RollingPeriod) -> RollingMetrics:
        """
        Calculate rolling metrics for specified period.

        Args:
            period: Time period for calculation

        Returns:
            RollingMetrics with calculated values
        """
        days = self._get_period_days(period)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Filter equity history
        equity_points = [
            (t, e) for t, e in self._equity_history
            if t > cutoff
        ]

        # Filter trades
        trades = [
            t for t in self._trade_history
            if t.get("close_time", datetime.now(timezone.utc)) > cutoff
        ]

        # Calculate returns
        if len(equity_points) < 2:
            return self._empty_metrics(period)

        equities = [e for _, e in equity_points]
        returns = np.diff(equities) / equities[:-1]

        # Total return
        total_return = (equities[-1] / equities[0] - 1) * 100 if equities[0] > 0 else 0

        # Volatility (annualized)
        daily_vol = np.std(returns) if len(returns) > 1 else 0
        ann_vol = daily_vol * np.sqrt(252) * 100

        # Max drawdown
        running_max = np.maximum.accumulate(equities)
        drawdowns = (equities - running_max) / running_max * 100
        max_dd = abs(min(drawdowns)) if len(drawdowns) > 0 else 0

        # Sharpe ratio (annualized)
        if daily_vol > 0:
            excess_return = np.mean(returns) - self._risk_free_rate / 252
            sharpe = excess_return / daily_vol * np.sqrt(252)
        else:
            sharpe = None

        # Sortino ratio (annualized)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_vol = np.std(negative_returns)
            if downside_vol > 0:
                excess_return = np.mean(returns) - self._risk_free_rate / 252
                sortino = excess_return / downside_vol * np.sqrt(252)
            else:
                sortino = None
        else:
            sortino = sharpe  # No negative returns = use Sharpe

        # Calmar ratio
        if max_dd > 0:
            ann_return = total_return * 365 / days if days > 0 else 0
            calmar = ann_return / max_dd
        else:
            calmar = None

        # Win rate and profit factor from trades
        if trades:
            winning = [t for t in trades if t.get("pnl", 0) > 0]
            losing = [t for t in trades if t.get("pnl", 0) < 0]
            win_rate = len(winning) / len(trades) * 100

            gross_profit = sum(t.get("pnl", 0) for t in winning)
            gross_loss = abs(sum(t.get("pnl", 0) for t in losing))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        else:
            win_rate = 0
            profit_factor = None

        return RollingMetrics(
            period=period,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_return_pct=total_return,
            volatility_pct=ann_vol,
            max_drawdown_pct=max_dd,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            num_trades=len(trades),
        )

    def _empty_metrics(self, period: RollingPeriod) -> RollingMetrics:
        """Return empty metrics for insufficient data."""
        return RollingMetrics(
            period=period,
            sharpe_ratio=None,
            sortino_ratio=None,
            calmar_ratio=None,
            total_return_pct=0,
            volatility_pct=0,
            max_drawdown_pct=0,
            win_rate_pct=0,
            profit_factor=None,
            num_trades=0,
        )

    def get_all_periods(self) -> dict[str, dict]:
        """Get metrics for all periods."""
        return {
            period.value: self.calculate_metrics(period).to_dict()
            for period in RollingPeriod
        }


# =============================================================================
# 8.2 WIN RATE BY SESSION PANEL
# =============================================================================

class TradingSession(Enum):
    """Trading session definitions."""
    ASIAN = "asian"           # 00:00 - 08:00 UTC
    EUROPEAN = "european"     # 07:00 - 16:00 UTC
    US = "us"                 # 13:00 - 21:00 UTC
    OVERLAP_EU_US = "overlap" # 13:00 - 16:00 UTC


@dataclass
class SessionPerformance:
    """Performance metrics for a trading session."""
    session: TradingSession
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float | None
    total_pnl: float
    avg_trade_duration_minutes: float
    best_trade: float
    worst_trade: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session.value,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": round(self.win_rate_pct, 1),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 2) if self.profit_factor else None,
            "total_pnl": round(self.total_pnl, 2),
            "avg_trade_duration_minutes": round(self.avg_trade_duration_minutes, 1),
            "best_trade": round(self.best_trade, 2),
            "worst_trade": round(self.worst_trade, 2),
        }


class SessionPerformanceTracker:
    """
    Track and analyze performance by trading session (Phase 8.2).
    """

    def __init__(self):
        """Initialize session tracker."""
        self._trades_by_session: dict[TradingSession, list[dict]] = {
            session: [] for session in TradingSession
        }
        logger.info("SessionPerformanceTracker initialized")

    def _get_session(self, timestamp: datetime) -> TradingSession:
        """Determine trading session from timestamp."""
        hour = timestamp.hour

        # Overlap takes priority
        if 13 <= hour < 16:
            return TradingSession.OVERLAP_EU_US
        elif 0 <= hour < 8:
            return TradingSession.ASIAN
        elif 7 <= hour < 16:
            return TradingSession.EUROPEAN
        elif 13 <= hour < 21:
            return TradingSession.US
        else:
            return TradingSession.US  # Default to US for after-hours

    def record_trade(
        self,
        open_time: datetime,
        close_time: datetime,
        pnl: float,
        symbol: str,
    ) -> None:
        """Record a completed trade."""
        session = self._get_session(open_time)
        duration = (close_time - open_time).total_seconds() / 60

        self._trades_by_session[session].append({
            "open_time": open_time,
            "close_time": close_time,
            "pnl": pnl,
            "symbol": symbol,
            "duration_minutes": duration,
        })

        # Keep last 6 months of trades per session
        cutoff = datetime.now(timezone.utc) - timedelta(days=180)
        self._trades_by_session[session] = [
            t for t in self._trades_by_session[session]
            if t["close_time"] > cutoff
        ]

    def get_session_performance(self, session: TradingSession) -> SessionPerformance:
        """Get performance metrics for a session."""
        trades = self._trades_by_session[session]

        if not trades:
            return SessionPerformance(
                session=session,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=None,
                total_pnl=0,
                avg_trade_duration_minutes=0,
                best_trade=0,
                worst_trade=0,
            )

        winning = [t for t in trades if t["pnl"] > 0]
        losing = [t for t in trades if t["pnl"] < 0]

        win_rate = len(winning) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t["pnl"] for t in winning]) if winning else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losing])) if losing else 0

        gross_profit = sum(t["pnl"] for t in winning)
        gross_loss = abs(sum(t["pnl"] for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

        total_pnl = sum(t["pnl"] for t in trades)
        avg_duration = np.mean([t["duration_minutes"] for t in trades])

        pnls = [t["pnl"] for t in trades]

        return SessionPerformance(
            session=session,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate_pct=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            avg_trade_duration_minutes=avg_duration,
            best_trade=max(pnls) if pnls else 0,
            worst_trade=min(pnls) if pnls else 0,
        )

    def get_all_sessions(self) -> dict[str, dict]:
        """Get performance for all sessions."""
        return {
            session.value: self.get_session_performance(session).to_dict()
            for session in TradingSession
        }

    def get_best_session(self) -> TradingSession | None:
        """Get the session with best win rate (min 10 trades)."""
        best = None
        best_rate = 0

        for session in TradingSession:
            perf = self.get_session_performance(session)
            if perf.total_trades >= 10 and perf.win_rate_pct > best_rate:
                best = session
                best_rate = perf.win_rate_pct

        return best


# =============================================================================
# 8.3 STRATEGY PERFORMANCE COMPARISON
# =============================================================================

@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy/agent."""
    strategy_name: str
    total_signals: int
    signals_acted_on: int
    hit_rate_pct: float
    avg_conviction: float
    total_pnl: float
    pnl_contribution_pct: float
    sharpe_contribution: float
    avg_holding_period_hours: float
    win_rate_pct: float
    profit_factor: float | None
    max_drawdown_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "total_signals": self.total_signals,
            "signals_acted_on": self.signals_acted_on,
            "signal_follow_rate_pct": round(self.signals_acted_on / self.total_signals * 100, 1) if self.total_signals > 0 else 0,
            "hit_rate_pct": round(self.hit_rate_pct, 1),
            "avg_conviction": round(self.avg_conviction, 2),
            "total_pnl": round(self.total_pnl, 2),
            "pnl_contribution_pct": round(self.pnl_contribution_pct, 1),
            "sharpe_contribution": round(self.sharpe_contribution, 2),
            "avg_holding_period_hours": round(self.avg_holding_period_hours, 1),
            "win_rate_pct": round(self.win_rate_pct, 1),
            "profit_factor": round(self.profit_factor, 2) if self.profit_factor else None,
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
        }


class StrategyComparisonTracker:
    """
    Track and compare strategy performance (Phase 8.3).
    """

    def __init__(self):
        """Initialize strategy tracker."""
        self._strategy_signals: dict[str, list[dict]] = defaultdict(list)
        self._strategy_trades: dict[str, list[dict]] = defaultdict(list)
        self._total_portfolio_pnl: float = 0

        logger.info("StrategyComparisonTracker initialized")

    def record_signal(
        self,
        strategy: str,
        symbol: str,
        direction: str,
        conviction: float,
        was_acted_on: bool,
        timestamp: datetime,
    ) -> None:
        """Record a signal from a strategy."""
        self._strategy_signals[strategy].append({
            "symbol": symbol,
            "direction": direction,
            "conviction": conviction,
            "was_acted_on": was_acted_on,
            "timestamp": timestamp,
        })

        # Keep last 6 months
        cutoff = datetime.now(timezone.utc) - timedelta(days=180)
        self._strategy_signals[strategy] = [
            s for s in self._strategy_signals[strategy]
            if s["timestamp"] > cutoff
        ]

    def record_trade(
        self,
        strategy: str,
        symbol: str,
        pnl: float,
        holding_hours: float,
        timestamp: datetime,
    ) -> None:
        """Record a completed trade attributed to a strategy."""
        self._strategy_trades[strategy].append({
            "symbol": symbol,
            "pnl": pnl,
            "holding_hours": holding_hours,
            "timestamp": timestamp,
        })
        self._total_portfolio_pnl += pnl

        # Keep last 6 months
        cutoff = datetime.now(timezone.utc) - timedelta(days=180)
        self._strategy_trades[strategy] = [
            t for t in self._strategy_trades[strategy]
            if t["timestamp"] > cutoff
        ]

    def get_strategy_performance(self, strategy: str) -> StrategyPerformance:
        """Get performance metrics for a strategy."""
        signals = self._strategy_signals.get(strategy, [])
        trades = self._strategy_trades.get(strategy, [])

        if not signals and not trades:
            return StrategyPerformance(
                strategy_name=strategy,
                total_signals=0,
                signals_acted_on=0,
                hit_rate_pct=0,
                avg_conviction=0,
                total_pnl=0,
                pnl_contribution_pct=0,
                sharpe_contribution=0,
                avg_holding_period_hours=0,
                win_rate_pct=0,
                profit_factor=None,
                max_drawdown_pct=0,
            )

        total_signals = len(signals)
        signals_acted = sum(1 for s in signals if s["was_acted_on"])
        avg_conviction = np.mean([s["conviction"] for s in signals]) if signals else 0

        # Trade metrics
        winning = [t for t in trades if t["pnl"] > 0]
        losing = [t for t in trades if t["pnl"] < 0]

        total_pnl = sum(t["pnl"] for t in trades)
        pnl_contribution = (
            total_pnl / self._total_portfolio_pnl * 100
            if self._total_portfolio_pnl != 0 else 0
        )

        win_rate = len(winning) / len(trades) * 100 if trades else 0

        gross_profit = sum(t["pnl"] for t in winning)
        gross_loss = abs(sum(t["pnl"] for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

        avg_holding = np.mean([t["holding_hours"] for t in trades]) if trades else 0

        # Calculate hit rate (signals that led to profitable trades)
        acted_signals = [s for s in signals if s["was_acted_on"]]
        if trades and acted_signals:
            profitable_signals = len(winning)
            hit_rate = profitable_signals / len(acted_signals) * 100 if acted_signals else 0
        else:
            hit_rate = 0

        # Simplified drawdown (running PnL)
        if trades:
            running_pnl = np.cumsum([t["pnl"] for t in trades])
            running_max = np.maximum.accumulate(running_pnl)
            drawdowns = running_pnl - running_max
            max_dd = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
            max_dd_pct = max_dd / max(abs(running_max.max()), 1) * 100
        else:
            max_dd_pct = 0

        return StrategyPerformance(
            strategy_name=strategy,
            total_signals=total_signals,
            signals_acted_on=signals_acted,
            hit_rate_pct=hit_rate,
            avg_conviction=avg_conviction,
            total_pnl=total_pnl,
            pnl_contribution_pct=pnl_contribution,
            sharpe_contribution=0,  # TODO: Calculate from returns
            avg_holding_period_hours=avg_holding,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd_pct,
        )

    def get_all_strategies(self) -> dict[str, dict]:
        """Get performance for all strategies."""
        strategies = set(self._strategy_signals.keys()) | set(self._strategy_trades.keys())
        return {
            strategy: self.get_strategy_performance(strategy).to_dict()
            for strategy in strategies
        }

    def get_ranking(self, metric: str = "pnl_contribution_pct") -> list[dict]:
        """Get strategies ranked by specified metric."""
        all_perf = self.get_all_strategies()
        return sorted(
            all_perf.values(),
            key=lambda x: x.get(metric, 0) or 0,
            reverse=True
        )


# =============================================================================
# 8.4 RISK VISUALIZATION HEATMAP
# =============================================================================

@dataclass
class PositionRiskScore:
    """Risk score for a position."""
    symbol: str
    risk_score: int  # 0-100 (higher = riskier)
    var_contribution_pct: float
    correlation_to_portfolio: float
    concentration_pct: float
    volatility_percentile: int
    drawdown_pct: float
    days_in_position: int
    risk_factors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "risk_score": self.risk_score,
            "var_contribution_pct": round(self.var_contribution_pct, 2),
            "correlation_to_portfolio": round(self.correlation_to_portfolio, 2),
            "concentration_pct": round(self.concentration_pct, 2),
            "volatility_percentile": self.volatility_percentile,
            "drawdown_pct": round(self.drawdown_pct, 2),
            "days_in_position": self.days_in_position,
            "risk_factors": self.risk_factors,
        }


class RiskHeatmapGenerator:
    """
    Generate risk visualization heatmap data (Phase 8.4).
    """

    def __init__(self):
        """Initialize heatmap generator."""
        self._position_history: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        self._volatilities: dict[str, float] = {}
        self._correlations: dict[str, float] = {}

        logger.info("RiskHeatmapGenerator initialized")

    def update_position(
        self,
        symbol: str,
        value: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Update position value history."""
        ts = timestamp or datetime.now(timezone.utc)
        self._position_history[symbol].append((ts, value))

        # Keep 30 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        self._position_history[symbol] = [
            (t, v) for t, v in self._position_history[symbol]
            if t > cutoff
        ]

    def update_volatility(self, symbol: str, volatility: float) -> None:
        """Update symbol volatility."""
        self._volatilities[symbol] = volatility

    def update_correlation(self, symbol: str, correlation: float) -> None:
        """Update correlation to portfolio."""
        self._correlations[symbol] = correlation

    def calculate_risk_score(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
        var_contribution: float,
        entry_date: datetime,
    ) -> PositionRiskScore:
        """
        Calculate comprehensive risk score for a position.

        Score components:
        - Concentration risk (weight in portfolio)
        - Volatility percentile
        - Correlation to portfolio
        - VaR contribution
        - Days in position (time risk)
        - Current drawdown
        """
        risk_factors = []
        score = 0

        # Concentration (max 25 points)
        concentration = position_value / portfolio_value * 100 if portfolio_value > 0 else 0
        if concentration > 20:
            score += 25
            risk_factors.append("high_concentration")
        elif concentration > 10:
            score += 15
        elif concentration > 5:
            score += 8
        else:
            score += 3

        # Volatility (max 25 points)
        vol = self._volatilities.get(symbol, 0.2)
        vol_percentile = min(100, int(vol / 0.5 * 100))  # Assume 50% is max
        if vol_percentile > 80:
            score += 25
            risk_factors.append("high_volatility")
        elif vol_percentile > 60:
            score += 18
        elif vol_percentile > 40:
            score += 12
        else:
            score += 5

        # Correlation (max 20 points)
        corr = self._correlations.get(symbol, 0.5)
        if corr > 0.8:
            score += 20
            risk_factors.append("high_correlation")
        elif corr > 0.6:
            score += 14
        elif corr > 0.4:
            score += 8
        else:
            score += 3

        # VaR contribution (max 20 points)
        if var_contribution > 30:
            score += 20
            risk_factors.append("high_var_contribution")
        elif var_contribution > 20:
            score += 15
        elif var_contribution > 10:
            score += 10
        else:
            score += 5

        # Time in position (max 10 points)
        days_held = (datetime.now(timezone.utc) - entry_date).days
        if days_held > 30:
            score += 10
            risk_factors.append("long_holding_period")
        elif days_held > 14:
            score += 6
        elif days_held > 7:
            score += 3
        else:
            score += 1

        # Calculate drawdown
        history = self._position_history.get(symbol, [])
        if history:
            values = [v for _, v in history]
            peak = max(values)
            drawdown = (peak - position_value) / peak * 100 if peak > 0 else 0
        else:
            drawdown = 0

        return PositionRiskScore(
            symbol=symbol,
            risk_score=min(100, score),
            var_contribution_pct=var_contribution,
            correlation_to_portfolio=corr,
            concentration_pct=concentration,
            volatility_percentile=vol_percentile,
            drawdown_pct=drawdown,
            days_in_position=days_held,
            risk_factors=risk_factors,
        )

    def get_heatmap_data(
        self,
        positions: list[dict[str, Any]],
        portfolio_value: float,
    ) -> list[dict]:
        """
        Generate heatmap data for all positions.

        Args:
            positions: List of position dicts with symbol, value, entry_date, var_contribution
            portfolio_value: Total portfolio value

        Returns:
            List of PositionRiskScore dicts
        """
        return [
            self.calculate_risk_score(
                symbol=p["symbol"],
                position_value=p["value"],
                portfolio_value=portfolio_value,
                var_contribution=p.get("var_contribution", 0),
                entry_date=p.get("entry_date", datetime.now(timezone.utc) - timedelta(days=1)),
            ).to_dict()
            for p in positions
        ]


# =============================================================================
# 8.5 TRADE JOURNAL INTEGRATION
# =============================================================================

class TradeQuality(Enum):
    """Trade quality rating."""
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    POOR = 2
    TERRIBLE = 1


class EmotionalState(Enum):
    """Trader emotional state."""
    CONFIDENT = "confident"
    NEUTRAL = "neutral"
    ANXIOUS = "anxious"
    FEARFUL = "fearful"
    GREEDY = "greedy"
    FRUSTRATED = "frustrated"


@dataclass
class TradeJournalEntry:
    """Trade journal entry."""
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    strategy: str
    quality_rating: TradeQuality
    emotional_state: EmotionalState
    setup_notes: str
    execution_notes: str
    lessons_learned: str
    tags: list[str]
    screenshots: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 2),
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "holding_period_hours": round(
                (self.exit_time - self.entry_time).total_seconds() / 3600, 1
            ),
            "strategy": self.strategy,
            "quality_rating": self.quality_rating.value,
            "emotional_state": self.emotional_state.value,
            "setup_notes": self.setup_notes,
            "execution_notes": self.execution_notes,
            "lessons_learned": self.lessons_learned,
            "tags": self.tags,
            "screenshots": self.screenshots,
            "timestamp": self.timestamp.isoformat(),
        }


class TradeJournal:
    """
    Trade journal for tracking and reviewing trades (Phase 8.5).
    """

    def __init__(self):
        """Initialize trade journal."""
        self._entries: list[TradeJournalEntry] = []
        self._tags: set[str] = set()

        logger.info("TradeJournal initialized")

    def add_entry(self, entry: TradeJournalEntry) -> None:
        """Add journal entry."""
        self._entries.append(entry)
        self._tags.update(entry.tags)

        # Keep last 2 years
        cutoff = datetime.now(timezone.utc) - timedelta(days=730)
        self._entries = [e for e in self._entries if e.timestamp > cutoff]

    def create_entry(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
        strategy: str,
        quality_rating: int = 3,
        emotional_state: str = "neutral",
        setup_notes: str = "",
        execution_notes: str = "",
        lessons_learned: str = "",
        tags: list[str] | None = None,
    ) -> TradeJournalEntry:
        """Create and add a journal entry."""
        pnl = exit_price - entry_price if direction == "LONG" else entry_price - exit_price
        pnl_pct = pnl / entry_price * 100

        entry = TradeJournalEntry(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_time=entry_time,
            exit_time=exit_time,
            strategy=strategy,
            quality_rating=TradeQuality(quality_rating),
            emotional_state=EmotionalState(emotional_state),
            setup_notes=setup_notes,
            execution_notes=execution_notes,
            lessons_learned=lessons_learned,
            tags=tags or [],
            screenshots=[],
        )

        self.add_entry(entry)
        return entry

    def get_entries(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
        tag: str | None = None,
        min_quality: int | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get filtered journal entries."""
        entries = self._entries

        if symbol:
            entries = [e for e in entries if e.symbol == symbol]
        if strategy:
            entries = [e for e in entries if e.strategy == strategy]
        if tag:
            entries = [e for e in entries if tag in e.tags]
        if min_quality:
            entries = [e for e in entries if e.quality_rating.value >= min_quality]

        # Sort by timestamp descending
        entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)

        return [e.to_dict() for e in entries[:limit]]

    def get_quality_stats(self) -> dict[str, Any]:
        """Get statistics by quality rating."""
        stats = {q.value: {"count": 0, "total_pnl": 0, "avg_pnl": 0} for q in TradeQuality}

        for entry in self._entries:
            rating = entry.quality_rating.value
            stats[rating]["count"] += 1
            stats[rating]["total_pnl"] += entry.pnl

        for rating in stats:
            if stats[rating]["count"] > 0:
                stats[rating]["avg_pnl"] = round(
                    stats[rating]["total_pnl"] / stats[rating]["count"], 2
                )

        return stats

    def get_emotional_stats(self) -> dict[str, Any]:
        """Get statistics by emotional state."""
        stats = {e.value: {"count": 0, "total_pnl": 0, "win_rate": 0} for e in EmotionalState}

        for entry in self._entries:
            state = entry.emotional_state.value
            stats[state]["count"] += 1
            stats[state]["total_pnl"] += entry.pnl
            if entry.pnl > 0:
                stats[state]["win_rate"] += 1

        for state in stats:
            if stats[state]["count"] > 0:
                stats[state]["win_rate"] = round(
                    stats[state]["win_rate"] / stats[state]["count"] * 100, 1
                )

        return stats

    def get_tags(self) -> list[str]:
        """Get all available tags."""
        return sorted(self._tags)


# =============================================================================
# 8.6 SIGNAL CONSENSUS HEATMAP
# =============================================================================

@dataclass
class SignalConsensus:
    """Signal consensus analysis."""
    symbol: str
    market_bias: float  # -1 (bearish) to +1 (bullish)
    consensus_strength: float  # 0-1
    bullish_count: int
    bearish_count: int
    neutral_count: int
    agent_signals: dict[str, str]  # {agent: direction}
    disagreement_level: float  # 0-1 (higher = more disagreement)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market_bias": round(self.market_bias, 2),
            "consensus_strength": round(self.consensus_strength, 2),
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "agent_signals": self.agent_signals,
            "disagreement_level": round(self.disagreement_level, 2),
            "timestamp": self.timestamp.isoformat(),
        }


class SignalConsensusTracker:
    """
    Track and visualize signal consensus (Phase 8.6).
    """

    def __init__(self):
        """Initialize consensus tracker."""
        self._current_signals: dict[str, dict[str, str]] = defaultdict(dict)
        self._signal_history: dict[str, list[dict]] = defaultdict(list)

        logger.info("SignalConsensusTracker initialized")

    def record_signal(
        self,
        symbol: str,
        agent: str,
        direction: str,  # "LONG", "SHORT", "NEUTRAL"
        conviction: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Record signal from an agent."""
        ts = timestamp or datetime.now(timezone.utc)

        self._current_signals[symbol][agent] = direction

        self._signal_history[symbol].append({
            "agent": agent,
            "direction": direction,
            "conviction": conviction,
            "timestamp": ts,
        })

        # Keep last 24 hours of signal history
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        self._signal_history[symbol] = [
            s for s in self._signal_history[symbol]
            if s["timestamp"] > cutoff
        ]

    def get_consensus(self, symbol: str) -> SignalConsensus:
        """Calculate consensus for a symbol."""
        signals = self._current_signals.get(symbol, {})

        if not signals:
            return SignalConsensus(
                symbol=symbol,
                market_bias=0,
                consensus_strength=0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                agent_signals={},
                disagreement_level=0,
            )

        bullish = sum(1 for d in signals.values() if d == "LONG")
        bearish = sum(1 for d in signals.values() if d == "SHORT")
        neutral = sum(1 for d in signals.values() if d == "NEUTRAL")
        total = len(signals)

        # Market bias: +1 if all bullish, -1 if all bearish
        market_bias = (bullish - bearish) / total if total > 0 else 0

        # Consensus strength: How aligned are signals
        max_agreement = max(bullish, bearish, neutral)
        consensus_strength = max_agreement / total if total > 0 else 0

        # Disagreement: entropy-based
        probs = [bullish / total, bearish / total, neutral / total] if total > 0 else [0, 0, 0]
        probs = [p for p in probs if p > 0]
        if probs:
            entropy = -sum(p * np.log2(p) for p in probs)
            max_entropy = np.log2(3)  # Max with 3 categories
            disagreement = entropy / max_entropy
        else:
            disagreement = 0

        return SignalConsensus(
            symbol=symbol,
            market_bias=market_bias,
            consensus_strength=consensus_strength,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            agent_signals=signals,
            disagreement_level=disagreement,
        )

    def get_all_symbols(self) -> dict[str, dict]:
        """Get consensus for all symbols."""
        return {
            symbol: self.get_consensus(symbol).to_dict()
            for symbol in self._current_signals
        }

    def get_high_disagreement_alerts(self, threshold: float = 0.7) -> list[dict]:
        """Get symbols with high agent disagreement."""
        alerts = []
        for symbol in self._current_signals:
            consensus = self.get_consensus(symbol)
            if consensus.disagreement_level >= threshold:
                alerts.append({
                    "symbol": symbol,
                    "disagreement_level": consensus.disagreement_level,
                    "agent_signals": consensus.agent_signals,
                })
        return sorted(alerts, key=lambda x: x["disagreement_level"], reverse=True)

    def clear_old_signals(self) -> None:
        """Clear stale signals (older than 1 hour without update)."""
        # This would be called periodically
        pass


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_rolling_metrics_calculator(risk_free_rate: float = 0.05) -> RollingMetricsCalculator:
    """Create RollingMetricsCalculator instance."""
    return RollingMetricsCalculator(risk_free_rate)


def create_session_performance_tracker() -> SessionPerformanceTracker:
    """Create SessionPerformanceTracker instance."""
    return SessionPerformanceTracker()


def create_strategy_comparison_tracker() -> StrategyComparisonTracker:
    """Create StrategyComparisonTracker instance."""
    return StrategyComparisonTracker()


def create_risk_heatmap_generator() -> RiskHeatmapGenerator:
    """Create RiskHeatmapGenerator instance."""
    return RiskHeatmapGenerator()


def create_trade_journal() -> TradeJournal:
    """Create TradeJournal instance."""
    return TradeJournal()


def create_signal_consensus_tracker() -> SignalConsensusTracker:
    """Create SignalConsensusTracker instance."""
    return SignalConsensusTracker()


def create_all_analytics_components(risk_free_rate: float = 0.05) -> dict:
    """
    Create all analytics components in one call.

    Returns a dictionary with all analytics instances:
    - rolling_metrics: RollingMetricsCalculator
    - session_performance: SessionPerformanceTracker
    - strategy_comparison: StrategyComparisonTracker
    - risk_heatmap: RiskHeatmapGenerator
    - trade_journal: TradeJournal
    - signal_consensus: SignalConsensusTracker
    """
    return {
        "rolling_metrics": create_rolling_metrics_calculator(risk_free_rate),
        "session_performance": create_session_performance_tracker(),
        "strategy_comparison": create_strategy_comparison_tracker(),
        "risk_heatmap": create_risk_heatmap_generator(),
        "trade_journal": create_trade_journal(),
        "signal_consensus": create_signal_consensus_tracker(),
    }

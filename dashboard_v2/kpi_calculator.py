"""
KPI Calculator Engine
=====================

Pure math KPI calculations for the Bloomberg V2 dashboard.
No dependencies on trading system - takes raw data as input.

KPIs implemented:
- Sharpe Ratio (annualized)
- Sortino Ratio (downside deviation)
- Calmar Ratio (CAGR / Max Drawdown)
- Information Ratio (vs benchmark)
- Expectancy (expected $ per trade)
- Profit Factor (gross profit / gross loss)
- Recovery Factor (net profit / max drawdown)
- Ulcer Index (RMS of drawdown %)
- Max Drawdown (peak-to-trough %)
- Win Rate
- Average R-Multiple
- Payoff Ratio (avg win / avg loss)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


# Trading days per year for annualization
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05  # 5% annual (2026 rates)


@dataclass
class TradeRecord:
    """Single closed trade for KPI calculation."""
    pnl: float
    r_multiple: float = 0.0
    strategy: str = ""
    symbol: str = ""
    hold_time_hours: float = 0.0


@dataclass
class KPIResult:
    """All calculated KPIs."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    avg_r_multiple: float = 0.0
    payoff_ratio: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_hold_time_hours: float = 0.0
    # Strategy breakdown
    strategy_pnl: dict[str, float] = field(default_factory=dict)
    strategy_win_rate: dict[str, float] = field(default_factory=dict)
    strategy_trades: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "information_ratio": round(self.information_ratio, 3),
            "expectancy": round(self.expectancy, 2),
            "profit_factor": round(self.profit_factor, 3),
            "recovery_factor": round(self.recovery_factor, 3),
            "ulcer_index": round(self.ulcer_index, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_r_multiple": round(self.avg_r_multiple, 3),
            "payoff_ratio": round(self.payoff_ratio, 3),
            "total_trades": self.total_trades,
            "total_pnl": round(self.total_pnl, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "best_trade": round(self.best_trade, 2),
            "worst_trade": round(self.worst_trade, 2),
            "avg_hold_time_hours": round(self.avg_hold_time_hours, 2),
            "strategy_pnl": {k: round(v, 2) for k, v in self.strategy_pnl.items()},
            "strategy_win_rate": {k: round(v, 4) for k, v in self.strategy_win_rate.items()},
            "strategy_trades": self.strategy_trades,
        }


class KPICalculator:
    """
    Calculates professional trading KPIs from equity curve and trade history.

    All methods are pure functions - no side effects, no state mutation.
    """

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        self._rf = risk_free_rate

    def calculate_all(
        self,
        equity_curve: Sequence[float],
        trades: Sequence[TradeRecord],
        benchmark_returns: Sequence[float] | None = None,
    ) -> KPIResult:
        """Calculate all KPIs from equity curve and trade history."""
        result = KPIResult()

        if not trades:
            return result

        # Basic trade stats
        pnls = [t.pnl for t in trades]
        result.total_trades = len(trades)
        result.total_pnl = sum(pnls)
        result.best_trade = max(pnls)
        result.worst_trade = min(pnls)

        # Win/loss breakdown
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        result.win_rate = len(winners) / len(pnls) if pnls else 0.0
        result.avg_win = sum(winners) / len(winners) if winners else 0.0
        result.avg_loss = sum(losers) / len(losers) if losers else 0.0

        # Hold time
        hold_times = [t.hold_time_hours for t in trades if t.hold_time_hours > 0]
        result.avg_hold_time_hours = sum(hold_times) / len(hold_times) if hold_times else 0.0

        # R-Multiple
        r_multiples = [t.r_multiple for t in trades if t.r_multiple != 0.0]
        result.avg_r_multiple = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0

        # KPIs from trade data
        result.expectancy = self.expectancy(pnls)
        result.profit_factor = self.profit_factor(pnls)
        result.payoff_ratio = self.payoff_ratio(pnls)

        # KPIs from equity curve
        if len(equity_curve) >= 2:
            returns = self.returns_from_equity(equity_curve)
            result.sharpe_ratio = self.sharpe(returns)
            result.sortino_ratio = self.sortino(returns)
            result.max_drawdown_pct = self.max_drawdown(equity_curve)
            result.calmar_ratio = self.calmar(equity_curve)
            result.ulcer_index = self.ulcer_index(equity_curve)
            result.recovery_factor = self.recovery_factor(
                result.total_pnl, result.max_drawdown_pct, equity_curve[0]
            )

            if benchmark_returns and len(benchmark_returns) == len(returns):
                result.information_ratio = self.information_ratio(returns, benchmark_returns)

        # Strategy breakdown
        result.strategy_pnl, result.strategy_win_rate, result.strategy_trades = (
            self.strategy_attribution(trades)
        )

        return result

    # =========================================================================
    # Individual KPI Methods
    # =========================================================================

    @staticmethod
    def returns_from_equity(equity_curve: Sequence[float]) -> list[float]:
        """Convert equity curve to daily returns."""
        if len(equity_curve) < 2:
            return []
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] != 0:
                returns.append((equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1])
            else:
                returns.append(0.0)
        return returns

    def sharpe(self, returns: Sequence[float]) -> float:
        """
        Sharpe Ratio = (Rp - Rf) / sigma_p, annualized.

        Args:
            returns: Daily returns series

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        daily_rf = self._rf / TRADING_DAYS_PER_YEAR
        excess = [r - daily_rf for r in returns]
        mean_excess = sum(excess) / len(excess)
        variance = sum((r - mean_excess) ** 2 for r in excess) / (len(excess) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0:
            return 0.0
        return (mean_excess / std) * math.sqrt(TRADING_DAYS_PER_YEAR)

    def sortino(self, returns: Sequence[float]) -> float:
        """
        Sortino Ratio = (Rp - Rf) / downside_deviation, annualized.

        Only uses negative returns for risk calculation.
        """
        if len(returns) < 2:
            return 0.0
        daily_rf = self._rf / TRADING_DAYS_PER_YEAR
        excess = [r - daily_rf for r in returns]
        mean_excess = sum(excess) / len(excess)
        downside = [min(r, 0) ** 2 for r in excess]
        downside_var = sum(downside) / len(downside)
        downside_dev = math.sqrt(downside_var) if downside_var > 0 else 0.0
        if downside_dev == 0:
            return 0.0
        return (mean_excess / downside_dev) * math.sqrt(TRADING_DAYS_PER_YEAR)

    @staticmethod
    def max_drawdown(equity_curve: Sequence[float]) -> float:
        """
        Maximum Drawdown as a percentage (0.0 to 1.0).

        Peak-to-trough decline over the equity curve.
        """
        if len(equity_curve) < 2:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def calmar(self, equity_curve: Sequence[float]) -> float:
        """
        Calmar Ratio = CAGR / Max Drawdown.

        Uses the equity curve to compute annualized return.
        """
        if len(equity_curve) < 2:
            return 0.0
        dd = self.max_drawdown(equity_curve)
        if dd == 0:
            return 0.0
        # Annualized return
        total_return = (equity_curve[-1] / equity_curve[0]) if equity_curve[0] != 0 else 1.0
        n_years = len(equity_curve) / TRADING_DAYS_PER_YEAR
        if n_years <= 0 or total_return <= 0:
            return 0.0
        cagr = total_return ** (1.0 / n_years) - 1.0
        return cagr / dd

    def information_ratio(
        self,
        returns: Sequence[float],
        benchmark_returns: Sequence[float],
    ) -> float:
        """
        Information Ratio = (Rp - Rb) / tracking_error.

        Measures excess return per unit of tracking error vs benchmark.
        """
        if len(returns) < 2 or len(returns) != len(benchmark_returns):
            return 0.0
        excess = [r - b for r, b in zip(returns, benchmark_returns)]
        mean_excess = sum(excess) / len(excess)
        variance = sum((e - mean_excess) ** 2 for e in excess) / (len(excess) - 1)
        te = math.sqrt(variance) if variance > 0 else 0.0
        if te == 0:
            return 0.0
        return (mean_excess / te) * math.sqrt(TRADING_DAYS_PER_YEAR)

    @staticmethod
    def expectancy(pnls: Sequence[float]) -> float:
        """
        Expectancy = (Win% x AvgWin) - (Loss% x AvgLoss).

        Expected dollar amount per trade.
        """
        if not pnls:
            return 0.0
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        n = len(pnls)
        win_rate = len(winners) / n
        loss_rate = len(losers) / n
        avg_win = sum(winners) / len(winners) if winners else 0.0
        avg_loss = abs(sum(losers) / len(losers)) if losers else 0.0
        return (win_rate * avg_win) - (loss_rate * avg_loss)

    @staticmethod
    def profit_factor(pnls: Sequence[float]) -> float:
        """
        Profit Factor = Gross Profit / Gross Loss.

        Values > 1.0 indicate profitable system.
        """
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def recovery_factor(
        net_profit: float,
        max_drawdown_pct: float,
        initial_equity: float,
    ) -> float:
        """
        Recovery Factor = Net Profit / Max Drawdown ($).

        Higher is better - indicates how well the system recovers.
        """
        if initial_equity <= 0 or max_drawdown_pct <= 0:
            return 0.0
        max_dd_dollars = max_drawdown_pct * initial_equity
        if max_dd_dollars == 0:
            return 0.0
        return net_profit / max_dd_dollars

    @staticmethod
    def ulcer_index(equity_curve: Sequence[float]) -> float:
        """
        Ulcer Index = sqrt(mean(drawdown_pct^2)).

        Measures depth and duration of drawdowns. Lower is better.
        """
        if len(equity_curve) < 2:
            return 0.0
        peak = equity_curve[0]
        dd_sq_sum = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd_pct = ((peak - value) / peak * 100) if peak > 0 else 0.0
            dd_sq_sum += dd_pct ** 2
        mean_dd_sq = dd_sq_sum / len(equity_curve)
        return math.sqrt(mean_dd_sq)

    @staticmethod
    def payoff_ratio(pnls: Sequence[float]) -> float:
        """
        Payoff Ratio = Average Win / Average Loss.

        Also known as reward-to-risk ratio.
        """
        winners = [p for p in pnls if p > 0]
        losers = [abs(p) for p in pnls if p < 0]
        if not winners or not losers:
            return 0.0
        avg_win = sum(winners) / len(winners)
        avg_loss = sum(losers) / len(losers)
        if avg_loss == 0:
            return 0.0
        return avg_win / avg_loss

    @staticmethod
    def strategy_attribution(
        trades: Sequence[TradeRecord],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
        """
        Break down performance by strategy.

        Returns:
            (strategy_pnl, strategy_win_rate, strategy_trades)
        """
        pnl_by_strat: dict[str, float] = {}
        wins_by_strat: dict[str, int] = {}
        count_by_strat: dict[str, int] = {}

        for t in trades:
            strat = t.strategy or "unknown"
            pnl_by_strat[strat] = pnl_by_strat.get(strat, 0.0) + t.pnl
            count_by_strat[strat] = count_by_strat.get(strat, 0) + 1
            if t.pnl > 0:
                wins_by_strat[strat] = wins_by_strat.get(strat, 0) + 1

        wr_by_strat = {}
        for strat, count in count_by_strat.items():
            wr_by_strat[strat] = wins_by_strat.get(strat, 0) / count if count > 0 else 0.0

        return pnl_by_strat, wr_by_strat, count_by_strat


def create_kpi_calculator(risk_free_rate: float = RISK_FREE_RATE) -> KPICalculator:
    """Factory function for KPICalculator."""
    return KPICalculator(risk_free_rate=risk_free_rate)

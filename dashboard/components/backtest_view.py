"""
Backtest View
=============

Strategy backtesting results display and analysis.

Features:
- BacktestResult dataclass for storing backtest outcomes
- BacktestComparison for side-by-side strategy comparison
- TradeRecord for individual trade analysis
- Performance attribution by symbol, time period, market condition
- Monte Carlo simulation results display
- Export for strategy comparison charts

Usage:
    viewer = BacktestViewer()

    # Load a backtest
    result = viewer.load_backtest("backtest_123")

    # Compare strategies
    comparison = viewer.compare_strategies(["strategy_a", "strategy_b"])

    # Get trade analysis
    analysis = viewer.get_trade_analysis("backtest_123")

    # Export for charting
    chart_data = viewer.export_report("backtest_123")
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np


logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Market condition classification for attribution."""
    BULL = "bull"           # Trending up
    BEAR = "bear"           # Trending down
    SIDEWAYS = "sideways"   # Range-bound
    HIGH_VOL = "high_vol"   # High volatility
    LOW_VOL = "low_vol"     # Low volatility
    UNKNOWN = "unknown"


class TradeDirection(Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


@dataclass
class TradeRecord:
    """
    Record of a single trade in a backtest.

    Captures entry/exit details and performance metrics.
    """
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: TradeDirection
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    return_pct: float
    # Optional metadata
    trade_id: str = ""
    strategy_name: str = ""
    commission: float = 0.0
    slippage: float = 0.0
    holding_period_hours: float = 0.0
    market_condition: MarketCondition = MarketCondition.UNKNOWN
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate derived fields after initialization."""
        if self.holding_period_hours == 0.0 and self.exit_time and self.entry_time:
            delta = self.exit_time - self.entry_time
            self.holding_period_hours = delta.total_seconds() / 3600

    @property
    def net_pnl(self) -> float:
        """Net P&L after costs."""
        return self.pnl - self.commission - abs(self.slippage)

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.net_pnl > 0

    @property
    def risk_reward(self) -> float:
        """Calculate risk/reward ratio (assuming entry risk equals actual loss)."""
        if self.is_winner:
            # For winners, compare gain to average loss would be ideal
            # Here we just return the return percentage as proxy
            return self.return_pct
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trade_id": self.trade_id,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "net_pnl": self.net_pnl,
            "commission": self.commission,
            "slippage": self.slippage,
            "holding_period_hours": self.holding_period_hours,
            "market_condition": self.market_condition.value,
            "strategy_name": self.strategy_name,
            "is_winner": self.is_winner,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TradeRecord:
        """Create TradeRecord from dictionary."""
        entry_time = data.get("entry_time")
        exit_time = data.get("exit_time")

        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time)

        direction_str = data.get("direction", "long")
        direction = TradeDirection(direction_str) if direction_str else TradeDirection.LONG

        market_condition_str = data.get("market_condition", "unknown")
        market_condition = MarketCondition(market_condition_str) if market_condition_str else MarketCondition.UNKNOWN

        return cls(
            trade_id=data.get("trade_id", ""),
            entry_time=entry_time,
            exit_time=exit_time,
            symbol=data.get("symbol", ""),
            direction=direction,
            entry_price=data.get("entry_price", 0.0),
            exit_price=data.get("exit_price", 0.0),
            quantity=data.get("quantity", 0),
            pnl=data.get("pnl", 0.0),
            return_pct=data.get("return_pct", 0.0),
            commission=data.get("commission", 0.0),
            slippage=data.get("slippage", 0.0),
            holding_period_hours=data.get("holding_period_hours", 0.0),
            market_condition=market_condition,
            strategy_name=data.get("strategy_name", ""),
            tags=data.get("tags", {}),
        )


@dataclass
class EquityCurvePoint:
    """Single point on the equity curve."""
    timestamp: datetime
    equity: float
    drawdown: float = 0.0
    drawdown_pct: float = 0.0


@dataclass
class BacktestResult:
    """
    Complete backtest result for a strategy.

    Contains all performance metrics and trade history.
    """
    backtest_id: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float         # Percentage return
    sharpe_ratio: float
    max_drawdown: float         # Percentage drawdown
    win_rate: float             # Percentage of winning trades
    profit_factor: float        # Gross profit / gross loss
    total_trades: int
    equity_curve: list[EquityCurvePoint] = field(default_factory=list)
    # Additional metrics
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0   # Annual return / max drawdown
    avg_trade_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    largest_winner: float = 0.0
    largest_loser: float = 0.0
    avg_holding_hours: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    # Risk metrics
    volatility: float = 0.0     # Annualized
    var_95: float = 0.0         # 95% Value at Risk
    cvar_95: float = 0.0        # Conditional VaR (Expected Shortfall)
    # Trade records
    trades: list[TradeRecord] = field(default_factory=list)
    # Monte Carlo results (if available)
    monte_carlo_results: dict[str, Any] = field(default_factory=dict)
    # Metadata
    parameters: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""

    @property
    def net_profit(self) -> float:
        """Net profit in currency terms."""
        return self.final_capital - self.initial_capital

    @property
    def cagr(self) -> float:
        """Compound Annual Growth Rate."""
        if self.initial_capital <= 0 or self.final_capital <= 0:
            return 0.0
        years = (self.end_date - self.start_date).days / 365.25
        if years <= 0:
            return 0.0
        return ((self.final_capital / self.initial_capital) ** (1 / years)) - 1

    @property
    def expectancy(self) -> float:
        """Expected value per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.net_profit / self.total_trades

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "backtest_id": self.backtest_id,
            "strategy_name": self.strategy_name,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "net_profit": self.net_profit,
            "cagr": self.cagr,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor if self.profit_factor != float("inf") else None,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_trade_pnl": self.avg_trade_pnl,
            "avg_winner": self.avg_winner,
            "avg_loser": self.avg_loser,
            "largest_winner": self.largest_winner,
            "largest_loser": self.largest_loser,
            "avg_holding_hours": self.avg_holding_hours,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "long_win_rate": self.long_win_rate,
            "short_win_rate": self.short_win_rate,
            "volatility": self.volatility,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "expectancy": self.expectancy,
            "equity_curve": [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "equity": p.equity,
                    "drawdown": p.drawdown,
                    "drawdown_pct": p.drawdown_pct,
                }
                for p in self.equity_curve
            ],
            "trades": [t.to_dict() for t in self.trades],
            "monte_carlo_results": self.monte_carlo_results,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BacktestResult:
        """Create BacktestResult from dictionary."""
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        created_at = data.get("created_at")

        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        # Parse equity curve
        equity_curve = []
        for point in data.get("equity_curve", []):
            ts = point.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            equity_curve.append(EquityCurvePoint(
                timestamp=ts,
                equity=point.get("equity", 0.0),
                drawdown=point.get("drawdown", 0.0),
                drawdown_pct=point.get("drawdown_pct", 0.0),
            ))

        # Parse trades
        trades = [TradeRecord.from_dict(t) for t in data.get("trades", [])]

        return cls(
            backtest_id=data.get("backtest_id", ""),
            strategy_name=data.get("strategy_name", ""),
            start_date=start_date,
            end_date=end_date,
            initial_capital=data.get("initial_capital", 0.0),
            final_capital=data.get("final_capital", 0.0),
            total_return=data.get("total_return", 0.0),
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            sortino_ratio=data.get("sortino_ratio", 0.0),
            calmar_ratio=data.get("calmar_ratio", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            win_rate=data.get("win_rate", 0.0),
            profit_factor=data.get("profit_factor", 0.0) or 0.0,
            total_trades=data.get("total_trades", 0),
            winning_trades=data.get("winning_trades", 0),
            losing_trades=data.get("losing_trades", 0),
            avg_trade_pnl=data.get("avg_trade_pnl", 0.0),
            avg_winner=data.get("avg_winner", 0.0),
            avg_loser=data.get("avg_loser", 0.0),
            largest_winner=data.get("largest_winner", 0.0),
            largest_loser=data.get("largest_loser", 0.0),
            avg_holding_hours=data.get("avg_holding_hours", 0.0),
            long_trades=data.get("long_trades", 0),
            short_trades=data.get("short_trades", 0),
            long_win_rate=data.get("long_win_rate", 0.0),
            short_win_rate=data.get("short_win_rate", 0.0),
            volatility=data.get("volatility", 0.0),
            var_95=data.get("var_95", 0.0),
            cvar_95=data.get("cvar_95", 0.0),
            equity_curve=equity_curve,
            trades=trades,
            monte_carlo_results=data.get("monte_carlo_results", {}),
            parameters=data.get("parameters", {}),
            created_at=created_at or datetime.now(timezone.utc),
            notes=data.get("notes", ""),
        )


@dataclass
class BacktestComparison:
    """
    Side-by-side comparison of multiple backtest results.

    Enables strategy selection and ranking.
    """
    comparison_id: str
    strategies: list[str]
    results: list[BacktestResult]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def strategy_count(self) -> int:
        """Number of strategies being compared."""
        return len(self.results)

    def get_ranking(self, metric: str = "sharpe_ratio", ascending: bool = False) -> list[tuple[str, float]]:
        """
        Rank strategies by a specific metric.

        Args:
            metric: Metric to rank by (sharpe_ratio, total_return, max_drawdown, etc.)
            ascending: If True, lower values are better (e.g., for max_drawdown)

        Returns:
            List of (strategy_name, metric_value) tuples, sorted
        """
        rankings = []
        for result in self.results:
            value = getattr(result, metric, 0.0)
            if value is None:
                value = 0.0
            rankings.append((result.strategy_name, value))

        rankings.sort(key=lambda x: x[1], reverse=not ascending)
        return rankings

    def get_best_strategy(self, metric: str = "sharpe_ratio", ascending: bool = False) -> str | None:
        """Get the best strategy by a specific metric."""
        rankings = self.get_ranking(metric, ascending)
        return rankings[0][0] if rankings else None

    def get_comparison_table(self) -> dict[str, dict[str, Any]]:
        """
        Generate a comparison table of all strategies.

        Returns:
            Dict mapping strategy names to their metrics
        """
        table = {}
        for result in self.results:
            table[result.strategy_name] = {
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor if result.profit_factor != float("inf") else None,
                "total_trades": result.total_trades,
                "avg_trade_pnl": result.avg_trade_pnl,
                "volatility": result.volatility,
                "cagr": result.cagr,
            }
        return table

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "comparison_id": self.comparison_id,
            "strategies": self.strategies,
            "strategy_count": self.strategy_count,
            "comparison_table": self.get_comparison_table(),
            "rankings": {
                "by_sharpe": self.get_ranking("sharpe_ratio"),
                "by_return": self.get_ranking("total_return"),
                "by_drawdown": self.get_ranking("max_drawdown", ascending=True),
                "by_win_rate": self.get_ranking("win_rate"),
            },
            "best_sharpe": self.get_best_strategy("sharpe_ratio"),
            "best_return": self.get_best_strategy("total_return"),
            "lowest_drawdown": self.get_best_strategy("max_drawdown", ascending=True),
            "results": [r.to_dict() for r in self.results],
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TradeAnalysis:
    """Detailed analysis of trades from a backtest."""
    backtest_id: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    expectancy: float
    # By direction
    long_stats: dict[str, Any] = field(default_factory=dict)
    short_stats: dict[str, Any] = field(default_factory=dict)
    # By symbol
    symbol_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    # By time period
    monthly_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    daily_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    # By market condition
    condition_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Trade distribution
    pnl_distribution: dict[str, Any] = field(default_factory=dict)
    holding_time_distribution: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backtest_id": self.backtest_id,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_winner": self.avg_winner,
            "avg_loser": self.avg_loser,
            "profit_factor": self.profit_factor if self.profit_factor != float("inf") else None,
            "expectancy": self.expectancy,
            "long_stats": self.long_stats,
            "short_stats": self.short_stats,
            "symbol_stats": self.symbol_stats,
            "monthly_stats": self.monthly_stats,
            "daily_stats": self.daily_stats,
            "condition_stats": self.condition_stats,
            "pnl_distribution": self.pnl_distribution,
            "holding_time_distribution": self.holding_time_distribution,
        }


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    simulation_id: str
    num_simulations: int
    original_return: float
    original_sharpe: float
    original_max_drawdown: float
    # Simulation statistics
    mean_return: float
    median_return: float
    std_return: float
    percentile_5_return: float
    percentile_95_return: float
    mean_sharpe: float
    mean_max_drawdown: float
    percentile_95_drawdown: float
    # Probability metrics
    probability_profit: float       # % of simulations profitable
    probability_beat_original: float  # % of simulations that beat original
    # Equity curves (sampled)
    sample_curves: list[list[float]] = field(default_factory=list)
    confidence_bands: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "simulation_id": self.simulation_id,
            "num_simulations": self.num_simulations,
            "original_return": self.original_return,
            "original_sharpe": self.original_sharpe,
            "original_max_drawdown": self.original_max_drawdown,
            "mean_return": self.mean_return,
            "median_return": self.median_return,
            "std_return": self.std_return,
            "percentile_5_return": self.percentile_5_return,
            "percentile_95_return": self.percentile_95_return,
            "mean_sharpe": self.mean_sharpe,
            "mean_max_drawdown": self.mean_max_drawdown,
            "percentile_95_drawdown": self.percentile_95_drawdown,
            "probability_profit": self.probability_profit,
            "probability_beat_original": self.probability_beat_original,
            "sample_curves": self.sample_curves,
            "confidence_bands": self.confidence_bands,
        }


class BacktestViewer:
    """
    Strategy backtesting results viewer and analyzer.

    Features:
    - Load and display backtest results
    - Compare multiple strategies side-by-side
    - Detailed trade analysis with attribution
    - Monte Carlo simulation display
    - Export for charting and reporting
    """

    # Directory for backtest results storage
    DEFAULT_RESULTS_DIR = "backtests"

    def __init__(
        self,
        results_dir: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize backtest viewer.

        Args:
            results_dir: Directory containing backtest result files
            config: Configuration options
        """
        self._results_dir = Path(results_dir or self.DEFAULT_RESULTS_DIR)
        self._config = config or {}

        # Cache of loaded results
        self._results_cache: dict[str, BacktestResult] = {}

        # Ensure results directory exists
        self._results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"BacktestViewer initialized with results_dir={self._results_dir}")

    def load_backtest(self, backtest_id: str) -> BacktestResult | None:
        """
        Load a backtest result by ID.

        Args:
            backtest_id: Unique backtest identifier

        Returns:
            BacktestResult if found, None otherwise
        """
        # Check cache first
        if backtest_id in self._results_cache:
            return self._results_cache[backtest_id]

        # Try to load from file
        result_file = self._results_dir / f"{backtest_id}.json"
        if not result_file.exists():
            logger.warning(f"Backtest not found: {backtest_id}")
            return None

        try:
            with open(result_file, "r") as f:
                data = json.load(f)
            result = BacktestResult.from_dict(data)
            self._results_cache[backtest_id] = result
            logger.debug(f"Loaded backtest: {backtest_id}")
            return result
        except Exception as e:
            logger.exception(f"Error loading backtest {backtest_id}: {e}")
            return None

    def save_backtest(self, result: BacktestResult) -> bool:
        """
        Save a backtest result to file.

        Args:
            result: BacktestResult to save

        Returns:
            True if saved successfully
        """
        try:
            result_file = self._results_dir / f"{result.backtest_id}.json"
            with open(result_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            self._results_cache[result.backtest_id] = result
            logger.info(f"Saved backtest: {result.backtest_id}")
            return True
        except Exception as e:
            logger.exception(f"Error saving backtest {result.backtest_id}: {e}")
            return False

    def get_recent_backtests(
        self,
        limit: int = 20,
        strategy_filter: str | None = None,
    ) -> list[BacktestResult]:
        """
        Get recent backtest results.

        Args:
            limit: Maximum number of results to return
            strategy_filter: Optional strategy name filter

        Returns:
            List of BacktestResult, sorted by creation time (newest first)
        """
        results = []

        # Scan results directory
        for result_file in self._results_dir.glob("*.json"):
            try:
                backtest_id = result_file.stem
                result = self.load_backtest(backtest_id)
                if result:
                    if strategy_filter and result.strategy_name != strategy_filter:
                        continue
                    results.append(result)
            except Exception as e:
                logger.warning(f"Error loading {result_file}: {e}")

        # Sort by creation time (newest first)
        results.sort(key=lambda r: r.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

        return results[:limit]

    def compare_strategies(
        self,
        strategy_names: list[str] | None = None,
        backtest_ids: list[str] | None = None,
    ) -> BacktestComparison | None:
        """
        Compare multiple strategies side-by-side.

        Args:
            strategy_names: List of strategy names to compare (uses most recent backtest for each)
            backtest_ids: List of specific backtest IDs to compare

        Returns:
            BacktestComparison with comparison data
        """
        results = []

        if backtest_ids:
            # Load specific backtests
            for backtest_id in backtest_ids:
                result = self.load_backtest(backtest_id)
                if result:
                    results.append(result)
        elif strategy_names:
            # Get most recent backtest for each strategy
            all_results = self.get_recent_backtests(limit=100)
            for strategy in strategy_names:
                for result in all_results:
                    if result.strategy_name == strategy:
                        results.append(result)
                        break

        if not results:
            logger.warning("No results found for comparison")
            return None

        comparison = BacktestComparison(
            comparison_id=f"CMP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            strategies=[r.strategy_name for r in results],
            results=results,
            created_at=datetime.now(timezone.utc),
        )

        logger.info(f"Created comparison of {len(results)} strategies")
        return comparison

    def get_trade_analysis(self, backtest_id: str) -> TradeAnalysis | None:
        """
        Perform detailed analysis of trades from a backtest.

        Args:
            backtest_id: Backtest ID to analyze

        Returns:
            TradeAnalysis with detailed metrics
        """
        result = self.load_backtest(backtest_id)
        if not result:
            return None

        trades = result.trades
        if not trades:
            return TradeAnalysis(
                backtest_id=backtest_id,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_winner=0.0,
                avg_loser=0.0,
                profit_factor=0.0,
                expectancy=0.0,
            )

        # Basic statistics
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]

        total_trades = len(trades)
        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(t.net_pnl for t in winners) if winners else 0.0
        gross_loss = abs(sum(t.net_pnl for t in losers)) if losers else 0.0

        avg_winner = gross_profit / winning_trades if winning_trades > 0 else 0.0
        avg_loser = gross_loss / losing_trades if losing_trades > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
        expectancy = (gross_profit - gross_loss) / total_trades if total_trades > 0 else 0.0

        # By direction
        long_trades = [t for t in trades if t.direction == TradeDirection.LONG]
        short_trades = [t for t in trades if t.direction == TradeDirection.SHORT]

        long_stats = self._calculate_direction_stats(long_trades, "long")
        short_stats = self._calculate_direction_stats(short_trades, "short")

        # By symbol
        symbol_stats = self._calculate_symbol_attribution(trades)

        # By time period
        monthly_stats = self._calculate_time_attribution(trades, "monthly")
        daily_stats = self._calculate_time_attribution(trades, "daily")

        # By market condition
        condition_stats = self._calculate_condition_attribution(trades)

        # Distributions
        pnl_distribution = self._calculate_pnl_distribution(trades)
        holding_time_distribution = self._calculate_holding_time_distribution(trades)

        return TradeAnalysis(
            backtest_id=backtest_id,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            profit_factor=profit_factor,
            expectancy=expectancy,
            long_stats=long_stats,
            short_stats=short_stats,
            symbol_stats=symbol_stats,
            monthly_stats=monthly_stats,
            daily_stats=daily_stats,
            condition_stats=condition_stats,
            pnl_distribution=pnl_distribution,
            holding_time_distribution=holding_time_distribution,
        )

    def _calculate_direction_stats(
        self,
        trades: list[TradeRecord],
        direction: str,
    ) -> dict[str, Any]:
        """Calculate statistics for trades in a specific direction."""
        if not trades:
            return {
                "direction": direction,
                "total": 0,
                "winners": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
            }

        winners = [t for t in trades if t.is_winner]
        total_pnl = sum(t.net_pnl for t in trades)

        return {
            "direction": direction,
            "total": len(trades),
            "winners": len(winners),
            "win_rate": len(winners) / len(trades) if trades else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(trades) if trades else 0.0,
        }

    def _calculate_symbol_attribution(
        self,
        trades: list[TradeRecord],
    ) -> dict[str, dict[str, Any]]:
        """Calculate performance attribution by symbol."""
        symbol_trades: dict[str, list[TradeRecord]] = defaultdict(list)
        for trade in trades:
            symbol_trades[trade.symbol].append(trade)

        stats = {}
        for symbol, sym_trades in symbol_trades.items():
            winners = [t for t in sym_trades if t.is_winner]
            total_pnl = sum(t.net_pnl for t in sym_trades)

            stats[symbol] = {
                "total_trades": len(sym_trades),
                "winning_trades": len(winners),
                "win_rate": len(winners) / len(sym_trades) if sym_trades else 0.0,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(sym_trades) if sym_trades else 0.0,
                "pct_of_total_trades": len(sym_trades) / len(trades) if trades else 0.0,
            }

        return stats

    def _calculate_time_attribution(
        self,
        trades: list[TradeRecord],
        period: str = "monthly",
    ) -> dict[str, dict[str, Any]]:
        """Calculate performance attribution by time period."""
        period_trades: dict[str, list[TradeRecord]] = defaultdict(list)

        for trade in trades:
            if trade.entry_time:
                if period == "monthly":
                    key = trade.entry_time.strftime("%Y-%m")
                elif period == "daily":
                    key = trade.entry_time.strftime("%Y-%m-%d")
                else:
                    key = trade.entry_time.strftime("%Y")
                period_trades[key].append(trade)

        stats = {}
        for period_key, period_trd in sorted(period_trades.items()):
            winners = [t for t in period_trd if t.is_winner]
            total_pnl = sum(t.net_pnl for t in period_trd)

            stats[period_key] = {
                "total_trades": len(period_trd),
                "winning_trades": len(winners),
                "win_rate": len(winners) / len(period_trd) if period_trd else 0.0,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(period_trd) if period_trd else 0.0,
            }

        return stats

    def _calculate_condition_attribution(
        self,
        trades: list[TradeRecord],
    ) -> dict[str, dict[str, Any]]:
        """Calculate performance attribution by market condition."""
        condition_trades: dict[str, list[TradeRecord]] = defaultdict(list)
        for trade in trades:
            condition_trades[trade.market_condition.value].append(trade)

        stats = {}
        for condition, cond_trades in condition_trades.items():
            winners = [t for t in cond_trades if t.is_winner]
            total_pnl = sum(t.net_pnl for t in cond_trades)

            stats[condition] = {
                "total_trades": len(cond_trades),
                "winning_trades": len(winners),
                "win_rate": len(winners) / len(cond_trades) if cond_trades else 0.0,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(cond_trades) if cond_trades else 0.0,
            }

        return stats

    def _calculate_pnl_distribution(
        self,
        trades: list[TradeRecord],
    ) -> dict[str, Any]:
        """Calculate P&L distribution statistics."""
        if not trades:
            return {}

        pnls = [t.net_pnl for t in trades]
        pnl_array = np.array(pnls)

        return {
            "mean": float(np.mean(pnl_array)),
            "median": float(np.median(pnl_array)),
            "std": float(np.std(pnl_array)),
            "min": float(np.min(pnl_array)),
            "max": float(np.max(pnl_array)),
            "percentile_5": float(np.percentile(pnl_array, 5)),
            "percentile_25": float(np.percentile(pnl_array, 25)),
            "percentile_75": float(np.percentile(pnl_array, 75)),
            "percentile_95": float(np.percentile(pnl_array, 95)),
            "skewness": float(self._calculate_skewness(pnl_array)),
            "kurtosis": float(self._calculate_kurtosis(pnl_array)),
        }

    def _calculate_holding_time_distribution(
        self,
        trades: list[TradeRecord],
    ) -> dict[str, Any]:
        """Calculate holding time distribution statistics."""
        if not trades:
            return {}

        holding_times = [t.holding_period_hours for t in trades if t.holding_period_hours > 0]
        if not holding_times:
            return {}

        ht_array = np.array(holding_times)

        return {
            "mean_hours": float(np.mean(ht_array)),
            "median_hours": float(np.median(ht_array)),
            "std_hours": float(np.std(ht_array)),
            "min_hours": float(np.min(ht_array)),
            "max_hours": float(np.max(ht_array)),
            "percentile_5": float(np.percentile(ht_array, 5)),
            "percentile_95": float(np.percentile(ht_array, 95)),
        }

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of a distribution."""
        if len(data) < 3:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std < 1e-10:
            return 0.0
        return (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis of a distribution."""
        if len(data) < 4:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std < 1e-10:
            return 0.0
        m4 = np.mean((data - mean) ** 4)
        return (m4 / (std ** 4)) - 3  # Excess kurtosis

    def run_monte_carlo(
        self,
        backtest_id: str,
        num_simulations: int = 1000,
        num_sample_curves: int = 10,
    ) -> MonteCarloResult | None:
        """
        Run Monte Carlo simulation on backtest trades.

        Args:
            backtest_id: Backtest to simulate
            num_simulations: Number of simulations to run
            num_sample_curves: Number of sample curves to keep for visualization

        Returns:
            MonteCarloResult with simulation statistics
        """
        result = self.load_backtest(backtest_id)
        if not result or not result.trades:
            logger.warning(f"No trades found for Monte Carlo: {backtest_id}")
            return None

        # Extract trade returns
        trade_returns = [t.return_pct for t in result.trades]
        if not trade_returns:
            return None

        returns_array = np.array(trade_returns)
        num_trades = len(returns_array)

        # Run simulations (bootstrap resampling)
        simulated_returns = []
        simulated_sharpes = []
        simulated_drawdowns = []
        sample_curves = []

        for i in range(num_simulations):
            # Randomly resample trades with replacement
            resampled = np.random.choice(returns_array, size=num_trades, replace=True)

            # Calculate cumulative equity curve
            equity = np.cumprod(1 + resampled / 100)

            # Calculate total return
            total_return = (equity[-1] - 1) * 100
            simulated_returns.append(total_return)

            # Calculate Sharpe (annualized, assuming daily)
            if len(resampled) > 1 and np.std(resampled) > 0:
                sharpe = np.mean(resampled) / np.std(resampled) * np.sqrt(252)
            else:
                sharpe = 0.0
            simulated_sharpes.append(sharpe)

            # Calculate max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / running_max * 100
            max_dd = abs(np.min(drawdowns))
            simulated_drawdowns.append(max_dd)

            # Keep sample curves
            if i < num_sample_curves:
                sample_curves.append(equity.tolist())

        # Calculate statistics
        sim_returns = np.array(simulated_returns)
        sim_sharpes = np.array(simulated_sharpes)
        sim_drawdowns = np.array(simulated_drawdowns)

        # Confidence bands (5th, 25th, 50th, 75th, 95th percentiles at each point)
        confidence_bands = {}
        if sample_curves:
            # Reconstruct all curves for confidence bands
            all_curves = []
            for i in range(min(100, num_simulations)):
                resampled = np.random.choice(returns_array, size=num_trades, replace=True)
                equity = np.cumprod(1 + resampled / 100)
                all_curves.append(equity)

            all_curves = np.array(all_curves)
            confidence_bands = {
                "p5": np.percentile(all_curves, 5, axis=0).tolist(),
                "p25": np.percentile(all_curves, 25, axis=0).tolist(),
                "p50": np.percentile(all_curves, 50, axis=0).tolist(),
                "p75": np.percentile(all_curves, 75, axis=0).tolist(),
                "p95": np.percentile(all_curves, 95, axis=0).tolist(),
            }

        return MonteCarloResult(
            simulation_id=f"MC-{backtest_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            num_simulations=num_simulations,
            original_return=result.total_return,
            original_sharpe=result.sharpe_ratio,
            original_max_drawdown=result.max_drawdown,
            mean_return=float(np.mean(sim_returns)),
            median_return=float(np.median(sim_returns)),
            std_return=float(np.std(sim_returns)),
            percentile_5_return=float(np.percentile(sim_returns, 5)),
            percentile_95_return=float(np.percentile(sim_returns, 95)),
            mean_sharpe=float(np.mean(sim_sharpes)),
            mean_max_drawdown=float(np.mean(sim_drawdowns)),
            percentile_95_drawdown=float(np.percentile(sim_drawdowns, 95)),
            probability_profit=float(np.mean(sim_returns > 0)),
            probability_beat_original=float(np.mean(sim_returns > result.total_return)),
            sample_curves=sample_curves,
            confidence_bands=confidence_bands,
        )

    def export_report(
        self,
        backtest_id: str,
        include_trades: bool = True,
        include_monte_carlo: bool = False,
    ) -> dict[str, Any] | None:
        """
        Export comprehensive backtest report.

        Args:
            backtest_id: Backtest to export
            include_trades: Include individual trade records
            include_monte_carlo: Run and include Monte Carlo analysis

        Returns:
            Complete report as dictionary
        """
        result = self.load_backtest(backtest_id)
        if not result:
            return None

        # Get trade analysis
        trade_analysis = self.get_trade_analysis(backtest_id)

        report = {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "backtest": result.to_dict(),
            "trade_analysis": trade_analysis.to_dict() if trade_analysis else None,
            "summary": {
                "strategy_name": result.strategy_name,
                "period": f"{result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}",
                "total_return_pct": result.total_return,
                "net_profit": result.net_profit,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown,
                "total_trades": result.total_trades,
                "win_rate_pct": result.win_rate * 100,
            },
            "chart_data": self._prepare_chart_data(result),
        }

        if not include_trades:
            report["backtest"]["trades"] = []

        if include_monte_carlo:
            mc_result = self.run_monte_carlo(backtest_id)
            if mc_result:
                report["monte_carlo"] = mc_result.to_dict()

        return report

    def _prepare_chart_data(self, result: BacktestResult) -> dict[str, Any]:
        """Prepare data formatted for charting libraries."""
        # Equity curve data
        equity_data = []
        for point in result.equity_curve:
            equity_data.append({
                "x": point.timestamp.isoformat(),
                "y": point.equity,
                "drawdown": point.drawdown_pct,
            })

        # Trade PnL scatter data
        trade_scatter = []
        for trade in result.trades:
            if trade.entry_time and trade.exit_time:
                trade_scatter.append({
                    "x": trade.exit_time.isoformat(),
                    "y": trade.net_pnl,
                    "symbol": trade.symbol,
                    "direction": trade.direction.value,
                    "is_winner": trade.is_winner,
                })

        # Monthly returns bar chart data
        monthly_returns = defaultdict(float)
        for trade in result.trades:
            if trade.entry_time:
                month_key = trade.entry_time.strftime("%Y-%m")
                monthly_returns[month_key] += trade.net_pnl

        monthly_bar = [
            {"x": k, "y": v}
            for k, v in sorted(monthly_returns.items())
        ]

        # Win/Loss distribution
        win_loss_data = {
            "labels": ["Winners", "Losers"],
            "values": [result.winning_trades, result.losing_trades],
        }

        return {
            "equity_curve": equity_data,
            "trade_scatter": trade_scatter,
            "monthly_returns": monthly_bar,
            "win_loss_distribution": win_loss_data,
            "drawdown_chart": [
                {"x": p.timestamp.isoformat(), "y": p.drawdown_pct}
                for p in result.equity_curve
            ],
        }

    def export_comparison_report(
        self,
        comparison: BacktestComparison,
    ) -> dict[str, Any]:
        """
        Export comparison report for multiple strategies.

        Args:
            comparison: BacktestComparison to export

        Returns:
            Comparison report as dictionary
        """
        # Prepare comparative equity curves
        equity_curves = {}
        for result in comparison.results:
            curves_data = []
            for point in result.equity_curve:
                curves_data.append({
                    "timestamp": point.timestamp.isoformat(),
                    "equity": point.equity,
                })
            equity_curves[result.strategy_name] = curves_data

        return {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "comparison": comparison.to_dict(),
            "chart_data": {
                "equity_curves": equity_curves,
                "metrics_radar": self._prepare_radar_data(comparison),
                "metrics_bar": self._prepare_bar_comparison(comparison),
            },
        }

    def _prepare_radar_data(self, comparison: BacktestComparison) -> dict[str, Any]:
        """Prepare data for radar/spider chart comparing strategies."""
        metrics = ["sharpe_ratio", "win_rate", "profit_factor", "total_return"]

        # Normalize metrics for radar chart (0-1 scale)
        series = []
        for result in comparison.results:
            values = []
            for metric in metrics:
                value = getattr(result, metric, 0.0)
                if value is None or value == float("inf"):
                    value = 0.0
                values.append(value)
            series.append({
                "name": result.strategy_name,
                "values": values,
            })

        return {
            "labels": metrics,
            "series": series,
        }

    def _prepare_bar_comparison(self, comparison: BacktestComparison) -> dict[str, Any]:
        """Prepare data for bar chart comparison."""
        return {
            "strategies": [r.strategy_name for r in comparison.results],
            "metrics": {
                "total_return": [r.total_return for r in comparison.results],
                "sharpe_ratio": [r.sharpe_ratio for r in comparison.results],
                "max_drawdown": [r.max_drawdown for r in comparison.results],
                "win_rate": [r.win_rate * 100 for r in comparison.results],
            },
        }

    def list_available_backtests(self) -> list[dict[str, Any]]:
        """
        List all available backtests with summary info.

        Returns:
            List of backtest summaries
        """
        summaries = []
        for result_file in self._results_dir.glob("*.json"):
            try:
                backtest_id = result_file.stem
                result = self.load_backtest(backtest_id)
                if result:
                    summaries.append({
                        "backtest_id": result.backtest_id,
                        "strategy_name": result.strategy_name,
                        "start_date": result.start_date.isoformat() if result.start_date else None,
                        "end_date": result.end_date.isoformat() if result.end_date else None,
                        "total_return": result.total_return,
                        "sharpe_ratio": result.sharpe_ratio,
                        "total_trades": result.total_trades,
                        "created_at": result.created_at.isoformat() if result.created_at else None,
                    })
            except Exception as e:
                logger.warning(f"Error summarizing {result_file}: {e}")

        # Sort by creation time
        summaries.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return summaries

    def delete_backtest(self, backtest_id: str) -> bool:
        """
        Delete a backtest result.

        Args:
            backtest_id: Backtest to delete

        Returns:
            True if deleted successfully
        """
        try:
            result_file = self._results_dir / f"{backtest_id}.json"
            if result_file.exists():
                result_file.unlink()
            self._results_cache.pop(backtest_id, None)
            logger.info(f"Deleted backtest: {backtest_id}")
            return True
        except Exception as e:
            logger.exception(f"Error deleting backtest {backtest_id}: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """Get viewer status for monitoring."""
        return {
            "results_dir": str(self._results_dir),
            "cached_results": len(self._results_cache),
            "available_backtests": len(list(self._results_dir.glob("*.json"))),
        }

    def to_dict(self) -> dict[str, Any]:
        """Export viewer state to dictionary."""
        return {
            "status": self.get_status(),
            "available_backtests": self.list_available_backtests(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

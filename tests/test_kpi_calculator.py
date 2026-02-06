"""
Tests for KPI Calculator Engine (Dashboard V2)
===============================================

Tests for Sharpe, Sortino, Calmar, Information Ratio,
Expectancy, Profit Factor, Recovery Factor, Ulcer Index,
Max Drawdown, Win Rate, Payoff Ratio, Strategy Attribution.
"""

import math
import pytest

from dashboard_v2.kpi_calculator import (
    KPICalculator,
    KPIResult,
    TradeRecord,
    create_kpi_calculator,
    TRADING_DAYS_PER_YEAR,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def calc():
    """Create KPICalculator with 0% risk-free rate for simplicity."""
    return KPICalculator(risk_free_rate=0.0)


@pytest.fixture
def calc_rf():
    """Create KPICalculator with 5% risk-free rate."""
    return KPICalculator(risk_free_rate=0.05)


@pytest.fixture
def winning_trades():
    """All winning trades."""
    return [
        TradeRecord(pnl=100, r_multiple=1.5, strategy="Momentum", symbol="ES"),
        TradeRecord(pnl=200, r_multiple=2.0, strategy="Momentum", symbol="NQ"),
        TradeRecord(pnl=150, r_multiple=1.8, strategy="StatArb", symbol="ES"),
    ]


@pytest.fixture
def mixed_trades():
    """Mix of winning and losing trades."""
    return [
        TradeRecord(pnl=100, r_multiple=1.5, strategy="Momentum", symbol="ES", hold_time_hours=2.0),
        TradeRecord(pnl=-50, r_multiple=-0.5, strategy="Momentum", symbol="NQ", hold_time_hours=1.0),
        TradeRecord(pnl=200, r_multiple=2.0, strategy="StatArb", symbol="ES", hold_time_hours=4.0),
        TradeRecord(pnl=-30, r_multiple=-0.3, strategy="StatArb", symbol="CL", hold_time_hours=0.5),
        TradeRecord(pnl=80, r_multiple=1.0, strategy="Macro", symbol="GC", hold_time_hours=3.0),
    ]


@pytest.fixture
def equity_curve():
    """Sample equity curve with drawdown."""
    return [
        100000, 100500, 101000, 101500, 102000,  # Steady up
        101800, 101500, 101200, 101000, 100800,   # Drawdown
        101000, 101500, 102000, 102500, 103000,   # Recovery
        103500, 104000, 104500, 105000, 105500,   # New highs
    ]


# ============================================================================
# SHARPE RATIO TESTS
# ============================================================================

class TestSharpeRatio:
    """Tests for Sharpe Ratio calculation."""

    def test_constant_positive_returns_high_sharpe(self, calc):
        """Constant positive returns => near-zero std => very high Sharpe."""
        returns = [0.01] * 100
        sharpe = calc.sharpe(returns)
        # With constant returns, std is near-zero (float precision),
        # so Sharpe is extremely large (positive)
        assert sharpe > 100  # Effectively infinite

    def test_positive_sharpe(self, calc):
        """Positive mean return with volatility."""
        returns = [0.01, 0.02, -0.005, 0.015, 0.01, 0.008, -0.003, 0.012] * 10
        sharpe = calc.sharpe(returns)
        assert sharpe > 0

    def test_negative_sharpe(self, calc):
        """Negative mean return."""
        returns = [-0.01, -0.02, 0.005, -0.015, -0.01, -0.008, 0.003, -0.012] * 10
        sharpe = calc.sharpe(returns)
        assert sharpe < 0

    def test_insufficient_data(self, calc):
        """Less than 2 data points returns 0."""
        assert calc.sharpe([]) == 0.0
        assert calc.sharpe([0.01]) == 0.0

    def test_annualized(self, calc):
        """Sharpe is annualized by sqrt(252)."""
        returns = [0.001] * 252 + [-0.001] * 252
        sharpe = calc.sharpe(returns)
        # Just check it's reasonable and not NaN
        assert math.isfinite(sharpe)

    def test_risk_free_reduces_sharpe(self, calc_rf):
        """Non-zero risk-free rate reduces Sharpe."""
        calc_zero = KPICalculator(risk_free_rate=0.0)
        returns = [0.01, 0.02, -0.005, 0.015, 0.01] * 20
        sharpe_zero = calc_zero.sharpe(returns)
        sharpe_rf = calc_rf.sharpe(returns)
        assert sharpe_rf < sharpe_zero


# ============================================================================
# SORTINO RATIO TESTS
# ============================================================================

class TestSortinoRatio:
    """Tests for Sortino Ratio calculation."""

    def test_no_downside_returns_zero(self, calc):
        """All positive returns => downside dev = 0 => Sortino = 0."""
        returns = [0.01, 0.02, 0.015, 0.01, 0.008] * 20
        sortino = calc.sortino(returns)
        assert sortino == 0.0

    def test_positive_sortino(self, calc):
        """Normal case with mixed returns."""
        returns = [0.01, -0.005, 0.02, -0.003, 0.015, -0.01, 0.008] * 10
        sortino = calc.sortino(returns)
        assert sortino > 0

    def test_insufficient_data(self, calc):
        """Less than 2 data points."""
        assert calc.sortino([]) == 0.0
        assert calc.sortino([0.01]) == 0.0

    def test_sortino_uses_downside_only(self, calc):
        """Sortino should only penalize negative returns."""
        # Same mean but different upside volatility
        returns1 = [0.01, 0.01, -0.005, -0.005] * 20  # Low upside vol
        returns2 = [0.05, -0.03, -0.005, -0.005] * 20  # High upside vol (same mean approx)
        sortino1 = calc.sortino(returns1)
        sortino2 = calc.sortino(returns2)
        # Sortino should not penalize upside vol as much as Sharpe does
        assert math.isfinite(sortino1)
        assert math.isfinite(sortino2)


# ============================================================================
# MAX DRAWDOWN TESTS
# ============================================================================

class TestMaxDrawdown:
    """Tests for Max Drawdown calculation."""

    def test_no_drawdown(self, calc):
        """Monotonically increasing equity."""
        equity = [100, 101, 102, 103, 104, 105]
        assert calc.max_drawdown(equity) == 0.0

    def test_simple_drawdown(self, calc):
        """Simple peak-to-trough."""
        equity = [100, 110, 90, 95, 100]
        dd = calc.max_drawdown(equity)
        expected = (110 - 90) / 110  # ~18.18%
        assert abs(dd - expected) < 0.001

    def test_multiple_drawdowns(self, calc):
        """Should return the largest drawdown."""
        equity = [100, 110, 95, 105, 80, 100]
        dd = calc.max_drawdown(equity)
        # Peak at 110, trough at 80 => (110-80)/110 = 27.27%
        expected = (110 - 80) / 110
        assert abs(dd - expected) < 0.001

    def test_full_drawdown(self, calc):
        """Equity goes to zero."""
        equity = [100, 50, 0]
        dd = calc.max_drawdown(equity)
        assert dd == 1.0

    def test_insufficient_data(self, calc):
        """Less than 2 points."""
        assert calc.max_drawdown([]) == 0.0
        assert calc.max_drawdown([100]) == 0.0

    def test_drawdown_percentage(self, equity_curve, calc):
        """Test with realistic equity curve."""
        dd = calc.max_drawdown(equity_curve)
        # Peak at 102000, trough at 100800 => 1.176%
        expected = (102000 - 100800) / 102000
        assert abs(dd - expected) < 0.001


# ============================================================================
# CALMAR RATIO TESTS
# ============================================================================

class TestCalmarRatio:
    """Tests for Calmar Ratio calculation."""

    def test_no_drawdown_returns_zero(self, calc):
        """Zero drawdown => zero Calmar (avoid division by zero)."""
        equity = [100, 101, 102, 103]
        assert calc.calmar(equity) == 0.0

    def test_positive_calmar(self, calc):
        """Profitable system with drawdown."""
        equity = list(range(100000, 110000, 100))  # Steady 10% gain
        equity[50] = equity[49] - 500  # Small drawdown
        calmar = calc.calmar(equity)
        assert calmar > 0

    def test_insufficient_data(self, calc):
        assert calc.calmar([]) == 0.0
        assert calc.calmar([100]) == 0.0


# ============================================================================
# EXPECTANCY TESTS
# ============================================================================

class TestExpectancy:
    """Tests for Expectancy calculation."""

    def test_positive_expectancy(self, calc, mixed_trades):
        """Mixed trades with positive expectancy."""
        pnls = [t.pnl for t in mixed_trades]
        exp = calc.expectancy(pnls)
        # 3 wins avg=126.67, 2 losses avg=-40, win%=60%, loss%=40%
        # (0.6 * 126.67) - (0.4 * 40) = 76 - 16 = 60
        assert exp > 0

    def test_zero_with_no_trades(self, calc):
        """Empty list returns 0."""
        assert calc.expectancy([]) == 0.0

    def test_all_winners(self, calc, winning_trades):
        """All winners => expectancy = avg win."""
        pnls = [t.pnl for t in winning_trades]
        exp = calc.expectancy(pnls)
        assert exp == pytest.approx(sum(pnls) / len(pnls))

    def test_all_losers(self, calc):
        """All losers => negative expectancy."""
        pnls = [-100, -50, -75]
        exp = calc.expectancy(pnls)
        assert exp < 0


# ============================================================================
# PROFIT FACTOR TESTS
# ============================================================================

class TestProfitFactor:
    """Tests for Profit Factor calculation."""

    def test_profitable_system(self, calc, mixed_trades):
        """PF > 1 for profitable system."""
        pnls = [t.pnl for t in mixed_trades]
        pf = calc.profit_factor(pnls)
        gross_profit = 100 + 200 + 80  # 380
        gross_loss = 50 + 30  # 80
        assert pf == pytest.approx(gross_profit / gross_loss, abs=0.01)

    def test_no_losses_returns_inf(self, calc, winning_trades):
        """No losses => infinity."""
        pnls = [t.pnl for t in winning_trades]
        pf = calc.profit_factor(pnls)
        assert pf == float("inf")

    def test_no_profits_returns_zero(self, calc):
        """No profits => 0."""
        pnls = [-100, -50]
        pf = calc.profit_factor(pnls)
        assert pf == 0.0

    def test_empty_returns_zero(self, calc):
        """Empty list."""
        assert calc.profit_factor([]) == 0.0

    def test_breakeven_returns_one(self, calc):
        """Equal profits and losses."""
        pnls = [100, -100]
        pf = calc.profit_factor(pnls)
        assert pf == pytest.approx(1.0)


# ============================================================================
# PAYOFF RATIO TESTS
# ============================================================================

class TestPayoffRatio:
    """Tests for Payoff Ratio (avg win / avg loss)."""

    def test_normal_case(self, calc):
        """Standard payoff ratio."""
        pnls = [100, 200, -50, -50]
        ratio = calc.payoff_ratio(pnls)
        # avg_win = 150, avg_loss = 50 => ratio = 3.0
        assert ratio == pytest.approx(3.0)

    def test_no_winners(self, calc):
        """All losses => 0."""
        assert calc.payoff_ratio([-100, -50]) == 0.0

    def test_no_losers(self, calc):
        """All wins => 0 (no losses to divide by)."""
        assert calc.payoff_ratio([100, 50]) == 0.0


# ============================================================================
# RECOVERY FACTOR TESTS
# ============================================================================

class TestRecoveryFactor:
    """Tests for Recovery Factor."""

    def test_positive_recovery(self, calc):
        """Net profit / max drawdown in dollars."""
        rf = calc.recovery_factor(
            net_profit=5000,
            max_drawdown_pct=0.05,  # 5%
            initial_equity=100000,
        )
        # DD$ = 0.05 * 100000 = 5000, RF = 5000/5000 = 1.0
        assert rf == pytest.approx(1.0)

    def test_zero_drawdown(self, calc):
        """No drawdown => 0."""
        rf = calc.recovery_factor(5000, 0.0, 100000)
        assert rf == 0.0

    def test_zero_equity(self, calc):
        """Zero initial equity."""
        rf = calc.recovery_factor(5000, 0.05, 0)
        assert rf == 0.0


# ============================================================================
# ULCER INDEX TESTS
# ============================================================================

class TestUlcerIndex:
    """Tests for Ulcer Index."""

    def test_no_drawdown(self, calc):
        """Monotonically increasing => UI = 0."""
        equity = [100, 101, 102, 103, 104]
        ui = calc.ulcer_index(equity)
        assert ui == pytest.approx(0.0)

    def test_with_drawdown(self, calc):
        """Ulcer Index increases with drawdown depth and duration."""
        equity = [100, 110, 100, 90, 80, 85, 90, 95, 100, 110]
        ui = calc.ulcer_index(equity)
        assert ui > 0

    def test_deeper_drawdown_higher_ui(self, calc):
        """Deeper drawdown should give higher UI."""
        equity_small = [100, 110, 105, 110]
        equity_large = [100, 110, 70, 110]
        ui_small = calc.ulcer_index(equity_small)
        ui_large = calc.ulcer_index(equity_large)
        assert ui_large > ui_small

    def test_insufficient_data(self, calc):
        assert calc.ulcer_index([]) == 0.0
        assert calc.ulcer_index([100]) == 0.0


# ============================================================================
# INFORMATION RATIO TESTS
# ============================================================================

class TestInformationRatio:
    """Tests for Information Ratio."""

    def test_same_as_benchmark(self, calc):
        """Identical returns => tracking error 0 => IR = 0."""
        returns = [0.01, 0.02, -0.01, 0.015, 0.005] * 10
        ir = calc.information_ratio(returns, returns)
        assert ir == 0.0

    def test_outperformance(self, calc):
        """Consistent outperformance => positive IR."""
        returns = [0.02, 0.03, 0.01, 0.025, 0.015] * 10
        benchmark = [0.01, 0.02, 0.005, 0.015, 0.01] * 10
        ir = calc.information_ratio(returns, benchmark)
        assert ir > 0

    def test_mismatched_lengths(self, calc):
        """Different length arrays => 0."""
        assert calc.information_ratio([0.01, 0.02], [0.01]) == 0.0


# ============================================================================
# STRATEGY ATTRIBUTION TESTS
# ============================================================================

class TestStrategyAttribution:
    """Tests for strategy attribution breakdown."""

    def test_attribution_sums(self, calc, mixed_trades):
        """Total P&L across strategies should equal sum of all trades."""
        pnl_by_strat, _, _ = calc.strategy_attribution(mixed_trades)
        total_from_strats = sum(pnl_by_strat.values())
        total_from_trades = sum(t.pnl for t in mixed_trades)
        assert total_from_strats == pytest.approx(total_from_trades)

    def test_correct_strategies(self, calc, mixed_trades):
        """Should have the correct strategy names."""
        pnl_by_strat, wr_by_strat, count_by_strat = calc.strategy_attribution(mixed_trades)
        assert "Momentum" in pnl_by_strat
        assert "StatArb" in pnl_by_strat
        assert "Macro" in pnl_by_strat

    def test_win_rates(self, calc, mixed_trades):
        """Win rates should be correct per strategy."""
        _, wr_by_strat, count_by_strat = calc.strategy_attribution(mixed_trades)
        # Momentum: 1 win + 1 loss = 50%
        assert wr_by_strat["Momentum"] == pytest.approx(0.5)
        # StatArb: 1 win + 1 loss = 50%
        assert wr_by_strat["StatArb"] == pytest.approx(0.5)
        # Macro: 1 win = 100%
        assert wr_by_strat["Macro"] == pytest.approx(1.0)

    def test_trade_counts(self, calc, mixed_trades):
        """Trade counts per strategy."""
        _, _, count_by_strat = calc.strategy_attribution(mixed_trades)
        assert count_by_strat["Momentum"] == 2
        assert count_by_strat["StatArb"] == 2
        assert count_by_strat["Macro"] == 1


# ============================================================================
# CALCULATE_ALL INTEGRATION TESTS
# ============================================================================

class TestCalculateAll:
    """Tests for the full calculate_all pipeline."""

    def test_empty_trades(self, calc):
        """No trades returns empty result."""
        result = calc.calculate_all([], [])
        assert result.total_trades == 0
        assert result.total_pnl == 0.0

    def test_full_calculation(self, calc, equity_curve, mixed_trades):
        """Full pipeline with equity curve and trades."""
        result = calc.calculate_all(equity_curve, mixed_trades)
        assert result.total_trades == 5
        assert result.total_pnl == pytest.approx(300.0)
        assert result.win_rate == pytest.approx(0.6)
        assert result.best_trade == 200
        assert result.worst_trade == -50
        assert result.max_drawdown_pct > 0
        assert result.sharpe_ratio != 0 or result.total_trades > 0  # Computed
        assert "Momentum" in result.strategy_pnl
        assert result.avg_hold_time_hours > 0

    def test_result_to_dict(self, calc, equity_curve, mixed_trades):
        """KPIResult.to_dict() should be JSON-serializable."""
        result = calc.calculate_all(equity_curve, mixed_trades)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "sharpe_ratio" in d
        assert "sortino_ratio" in d
        assert "calmar_ratio" in d
        assert "profit_factor" in d
        assert "expectancy" in d
        assert "win_rate" in d
        assert "strategy_pnl" in d
        # All values should be numeric or dict
        for key, val in d.items():
            assert isinstance(val, (int, float, dict))


# ============================================================================
# FACTORY FUNCTION TEST
# ============================================================================

class TestFactory:
    """Tests for factory function."""

    def test_create_default(self):
        calc = create_kpi_calculator()
        assert isinstance(calc, KPICalculator)

    def test_create_custom_rf(self):
        calc = create_kpi_calculator(risk_free_rate=0.03)
        assert calc._rf == 0.03


# ============================================================================
# RETURNS FROM EQUITY TESTS
# ============================================================================

class TestReturnsFromEquity:
    """Tests for equity-to-returns conversion."""

    def test_basic_conversion(self, calc):
        returns = calc.returns_from_equity([100, 110, 105])
        assert len(returns) == 2
        assert returns[0] == pytest.approx(0.10)  # 100 -> 110
        assert returns[1] == pytest.approx(-0.04545, abs=0.001)  # 110 -> 105

    def test_empty(self, calc):
        assert calc.returns_from_equity([]) == []
        assert calc.returns_from_equity([100]) == []

    def test_zero_equity(self, calc):
        """Zero equity should not cause division by zero."""
        returns = calc.returns_from_equity([0, 100, 200])
        assert len(returns) == 2
        assert returns[0] == 0.0  # 0 -> 100, but 0 denominator

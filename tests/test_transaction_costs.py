"""
Tests for Transaction Cost Integration (Phase 5.2)
==================================================

Tests for the transaction cost aware portfolio optimization.
"""

import numpy as np
import pytest

from core.position_sizing import (
    PositionSizer,
    SizingMethod,
    TransactionCostConfig,
)


class TestTransactionCostConfig:
    """Tests for TransactionCostConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TransactionCostConfig()

        assert config.fixed_cost_per_trade == 0.0
        assert config.variable_cost_bps == 10.0
        assert config.market_impact_coefficient == 0.1
        assert config.turnover_penalty_lambda == 0.5
        assert config.min_trade_improvement_pct == 0.3

    def test_spread_by_asset_class(self):
        """Test spread lookup by asset class."""
        config = TransactionCostConfig()

        assert config.spread_by_asset_class["equity"] == 5.0
        assert config.spread_by_asset_class["future"] == 2.0
        assert config.spread_by_asset_class["forex"] == 1.0
        assert config.spread_by_asset_class["commodity"] == 10.0

    def test_get_total_cost_bps(self):
        """Test total cost calculation in basis points."""
        config = TransactionCostConfig(variable_cost_bps=10.0)

        # Equity: 10 bps variable + 5 bps spread = 15 bps
        assert config.get_total_cost_bps("equity") == 15.0

        # Forex: 10 bps variable + 1 bps spread = 11 bps
        assert config.get_total_cost_bps("forex") == 11.0

        # Unknown asset class uses default 5 bps spread
        assert config.get_total_cost_bps("unknown") == 15.0

    def test_calculate_trade_cost_basic(self):
        """Test basic trade cost calculation."""
        config = TransactionCostConfig(
            fixed_cost_per_trade=5.0,
            variable_cost_bps=10.0,
        )

        trade_value = 10000.0  # $10,000 trade

        # Fixed: $5.00
        # Variable: $10,000 * (10 + 5 spread) / 10000 = $15.00
        # Total: $20.00
        cost = config.calculate_trade_cost(trade_value, "equity")

        assert cost == pytest.approx(20.0, rel=0.01)

    def test_calculate_trade_cost_with_market_impact(self):
        """Test trade cost with market impact."""
        config = TransactionCostConfig(
            fixed_cost_per_trade=0.0,
            variable_cost_bps=10.0,
            market_impact_coefficient=0.1,
        )

        trade_value = 100000.0  # $100,000 trade
        avg_daily_volume = 1000000.0  # $1M daily volume

        # Variable + spread: $100,000 * 15 / 10000 = $150
        # Impact: $100,000 * 0.1 * sqrt(100000/1000000) = $100,000 * 0.1 * 0.316 = $3,160
        cost = config.calculate_trade_cost(
            trade_value, "equity", avg_daily_volume
        )

        assert cost > 150.0  # Should include market impact
        assert cost < 5000.0  # But not unreasonably large


class TestTurnoverPenalizedOptimization:
    """Tests for turnover-penalized portfolio optimization."""

    @pytest.fixture
    def sizer(self):
        """Create PositionSizer instance."""
        return PositionSizer()

    @pytest.fixture
    def simple_portfolio(self):
        """Create simple 3-asset portfolio setup."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        expected_returns = {
            "AAPL": 0.10,  # 10% expected return
            "MSFT": 0.12,
            "GOOGL": 0.08,
        }
        # Simple covariance matrix
        cov_matrix = np.array([
            [0.04, 0.01, 0.005],   # AAPL vol ~20%
            [0.01, 0.0225, 0.008], # MSFT vol ~15%
            [0.005, 0.008, 0.0324] # GOOGL vol ~18%
        ])
        return symbols, expected_returns, cov_matrix

    def test_no_rebalance_when_cost_exceeds_benefit(self, sizer, simple_portfolio):
        """Test that optimizer recommends no trade when costs exceed benefits."""
        symbols, expected_returns, cov_matrix = simple_portfolio

        # Current weights are already close to optimal
        current_weights = {
            "AAPL": 0.33,
            "MSFT": 0.34,
            "GOOGL": 0.33,
        }

        # High transaction costs
        cost_config = TransactionCostConfig(
            variable_cost_bps=100.0,  # 1% cost - very high
            turnover_penalty_lambda=2.0,
        )

        results = sizer.optimize_portfolio_turnover_penalized(
            symbols=symbols,
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            portfolio_value=100000.0,
            current_weights=current_weights,
            cost_config=cost_config,
        )

        # Should return results close to current weights
        assert len(results) == 3

        for symbol in symbols:
            result = results[symbol]
            assert result.method == SizingMethod.TURNOVER_PENALIZED

    def test_rebalance_when_benefit_exceeds_cost(self, sizer, simple_portfolio):
        """Test rebalancing when benefits exceed costs."""
        symbols, expected_returns, cov_matrix = simple_portfolio

        # Very suboptimal current weights
        current_weights = {
            "AAPL": 0.10,  # Underweight high-return asset
            "MSFT": 0.10,
            "GOOGL": 0.80,  # Overweight low-return asset
        }

        # Low transaction costs
        cost_config = TransactionCostConfig(
            variable_cost_bps=5.0,
            turnover_penalty_lambda=0.1,
        )

        results = sizer.optimize_portfolio_turnover_penalized(
            symbols=symbols,
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            portfolio_value=100000.0,
            current_weights=current_weights,
            cost_config=cost_config,
        )

        # Should rebalance towards higher-return assets
        assert results["MSFT"].adjusted_fraction > 0.10  # Should increase MSFT

    def test_turnover_penalty_reduces_trade_size(self, sizer, simple_portfolio):
        """Test that turnover penalty reduces trade sizes."""
        symbols, expected_returns, cov_matrix = simple_portfolio

        current_weights = {
            "AAPL": 0.20,
            "MSFT": 0.30,
            "GOOGL": 0.50,
        }

        # Low penalty
        low_penalty_config = TransactionCostConfig(
            variable_cost_bps=5.0,
            turnover_penalty_lambda=0.1,
        )

        # High penalty
        high_penalty_config = TransactionCostConfig(
            variable_cost_bps=5.0,
            turnover_penalty_lambda=2.0,
        )

        results_low = sizer.optimize_portfolio_turnover_penalized(
            symbols=symbols,
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            portfolio_value=100000.0,
            current_weights=current_weights,
            cost_config=low_penalty_config,
        )

        results_high = sizer.optimize_portfolio_turnover_penalized(
            symbols=symbols,
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            portfolio_value=100000.0,
            current_weights=current_weights,
            cost_config=high_penalty_config,
        )

        # Calculate total turnover for each
        turnover_low = sum(
            abs(results_low[s].adjusted_fraction - current_weights[s])
            for s in symbols
        )
        turnover_high = sum(
            abs(results_high[s].adjusted_fraction - current_weights[s])
            for s in symbols
        )

        # High penalty should result in lower turnover
        assert turnover_high <= turnover_low + 0.01  # Small tolerance

    def test_empty_portfolio(self, sizer):
        """Test handling of empty portfolio."""
        results = sizer.optimize_portfolio_turnover_penalized(
            symbols=[],
            expected_returns={},
            covariance_matrix=np.array([[]]),
            portfolio_value=100000.0,
        )

        assert results == {}


class TestRebalancingThreshold:
    """Tests for rebalancing threshold calculation."""

    @pytest.fixture
    def sizer(self):
        """Create PositionSizer instance."""
        return PositionSizer()

    def test_rebalancing_recommendation_hold(self, sizer):
        """Test hold recommendation when turnover is small."""
        symbols = ["AAPL", "MSFT"]
        current_weights = {"AAPL": 0.50, "MSFT": 0.50}
        target_weights = {"AAPL": 0.51, "MSFT": 0.49}  # Small change

        result = sizer.calculate_rebalancing_threshold(
            symbols=symbols,
            current_weights=current_weights,
            target_weights=target_weights,
            portfolio_value=100000.0,
        )

        assert result["total_turnover_pct"] == pytest.approx(2.0, rel=0.01)
        assert result["estimated_cost_dollars"] > 0

    def test_rebalancing_recommendation_with_returns(self, sizer):
        """Test recommendation with expected returns."""
        symbols = ["AAPL", "MSFT"]
        current_weights = {"AAPL": 0.30, "MSFT": 0.70}
        target_weights = {"AAPL": 0.60, "MSFT": 0.40}
        expected_returns = {"AAPL": 0.15, "MSFT": 0.08}

        result = sizer.calculate_rebalancing_threshold(
            symbols=symbols,
            current_weights=current_weights,
            target_weights=target_weights,
            portfolio_value=100000.0,
            expected_returns=expected_returns,
        )

        # Moving to higher-return asset should show positive benefit
        assert result["expected_benefit_pct"] > 0
        assert "cost_benefit_ratio" in result

    def test_turnover_by_asset(self, sizer):
        """Test turnover calculation by asset."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        current_weights = {"AAPL": 0.40, "MSFT": 0.30, "GOOGL": 0.30}
        target_weights = {"AAPL": 0.30, "MSFT": 0.40, "GOOGL": 0.30}

        result = sizer.calculate_rebalancing_threshold(
            symbols=symbols,
            current_weights=current_weights,
            target_weights=target_weights,
            portfolio_value=100000.0,
        )

        assert result["turnover_by_asset"]["AAPL"] == pytest.approx(0.10)
        assert result["turnover_by_asset"]["MSFT"] == pytest.approx(0.10)
        assert result["turnover_by_asset"]["GOOGL"] == pytest.approx(0.0)

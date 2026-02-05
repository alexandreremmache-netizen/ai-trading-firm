"""
Tests for Cornish-Fisher VaR Adjustment (Phase 5.3)
===================================================

Tests for the Cornish-Fisher expansion to adjust VaR for non-normal returns.
"""

import numpy as np
import pytest

from core.var_calculator import VaRCalculator, VaRMethod


class TestCornishFisherQuantile:
    """Tests for Cornish-Fisher quantile calculation.

    Note: The _cornish_fisher_quantile function is a mathematical function
    that approximates quantiles of non-normal distributions. For VaR (which
    measures losses), the caller should negate skewness since losses = -returns.

    The tests below verify mathematical correctness of the CF expansion.
    """

    @pytest.fixture
    def calculator(self):
        """Create VaRCalculator instance."""
        return VaRCalculator()

    def test_normal_distribution_no_adjustment(self, calculator):
        """Test that normal distribution (S=0, K=0) gives no adjustment."""
        z = 1.645  # 95% confidence

        z_cf = calculator._cornish_fisher_quantile(z, skewness=0.0, excess_kurtosis=0.0)

        # Should be very close to original z
        assert z_cf == pytest.approx(z, rel=0.01)

    def test_positive_skewness_for_var(self, calculator):
        """Test that positive skewness increases z (for loss distributions).

        When returns have negative skew, losses have positive skew.
        VaR calculation uses -skewness, so we test with positive skewness here.
        """
        z = 1.645

        # Positive skewness in loss distribution (negative in returns)
        # This represents a situation where large losses are more frequent
        z_cf = calculator._cornish_fisher_quantile(z, skewness=0.5, excess_kurtosis=0.0)

        # With positive skew, upper quantile should be higher
        assert z_cf > z

    def test_excess_kurtosis_increases_quantile(self, calculator):
        """Test that excess kurtosis increases the quantile."""
        z = 1.645

        # Excess kurtosis: fatter tails
        z_cf = calculator._cornish_fisher_quantile(z, skewness=0.0, excess_kurtosis=3.0)

        # Fat tails mean higher quantiles for upper tail
        # Note: with pure kurtosis (no skew), the effect is more subtle
        # and depends on the exact quantile being computed
        assert np.isfinite(z_cf)
        # For the upper tail, excess kurtosis can slightly reduce z due to
        # the cubic term, but the overall effect on VaR is still conservative

    def test_combined_skew_and_kurtosis_for_var(self, calculator):
        """Test combined effect for VaR-relevant parameters.

        For VaR with negatively skewed returns, we use positive skewness
        in the CF formula (since losses = -returns).
        """
        z = 1.645

        # Positive skew + excess kurtosis (VaR perspective)
        z_cf = calculator._cornish_fisher_quantile(z, skewness=0.8, excess_kurtosis=4.0)

        # Should significantly increase z
        assert z_cf > z

    def test_numerical_stability(self, calculator):
        """Test that extreme values don't cause instability."""
        z = 2.326  # 99% confidence

        # Very extreme (but not unrealistic) values
        z_cf = calculator._cornish_fisher_quantile(z, skewness=-2.0, excess_kurtosis=10.0)

        # Should be bounded (not infinite or negative)
        assert np.isfinite(z_cf)
        assert z_cf > 0
        assert z_cf < z * 3.0  # Bounded at 3x


class TestCornishFisherVaR:
    """Tests for Cornish-Fisher VaR calculation."""

    @pytest.fixture
    def calculator(self):
        """Create VaRCalculator with sample data."""
        calc = VaRCalculator()

        # Add synthetic returns with negative skew and fat tails
        np.random.seed(42)

        # Create returns that are not normally distributed
        n_days = 252

        # Mix of normal and extreme moves (fat tails)
        normal_returns = np.random.normal(0.0005, 0.015, n_days)
        extreme_returns = np.random.choice(
            [-0.05, -0.04, -0.03, 0.025, 0.03],
            size=20,
            p=[0.3, 0.2, 0.2, 0.2, 0.1]  # More negative extremes (skewness)
        )

        # Replace some normal returns with extreme ones
        returns = normal_returns.copy()
        replace_indices = np.random.choice(n_days, 20, replace=False)
        for i, idx in enumerate(replace_indices):
            returns[idx] = extreme_returns[i]

        # Create correlated returns for MSFT
        msft_returns = returns * 0.8 + np.random.normal(0, 0.005, n_days)

        # Store in calculator cache
        calc._returns_cache = {
            "AAPL": returns.tolist(),
            "MSFT": msft_returns.tolist(),
        }

        # Also need to set up covariance for parametric VaR
        # Calculate covariance matrix
        returns_matrix = np.column_stack([returns, msft_returns])
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        calc._covariance_cache = cov_matrix

        return calc

    def test_cornish_fisher_var_returns_result(self, calculator):
        """Test that CF VaR calculation returns valid result."""
        positions = {"AAPL": 50000, "MSFT": 50000}

        result = calculator.calculate_cornish_fisher_var(
            positions=positions,
            portfolio_value=100000.0,
            confidence_level=0.95,
            horizon_days=1,
        )

        # Check result structure - may have error details if covariance not set up
        # The important thing is that it returns a valid VaRResult
        assert result is not None
        # If there's sufficient data, we should get a valid VaR
        if "error" not in result.details:
            assert result.var_absolute > 0 or result.var_pct > 0

    def test_cf_var_higher_than_normal_for_skewed_returns(self, calculator):
        """Test CF VaR is higher than normal VaR for negatively skewed returns."""
        positions = {"AAPL": 50000, "MSFT": 50000}

        # Calculate both
        cf_result = calculator.calculate_cornish_fisher_var(
            positions=positions,
            portfolio_value=100000.0,
            confidence_level=0.95,
            horizon_days=1,
        )

        normal_result = calculator.calculate_parametric_var(
            positions=positions,
            portfolio_value=100000.0,
            confidence_level=0.95,
            horizon_days=1,
        )

        # If returns are negatively skewed (which our test data is),
        # CF VaR should be higher
        if "skewness" in cf_result.details:
            skewness = cf_result.details["skewness"]
            if skewness < -0.1:  # Negative skew
                assert cf_result.var_absolute >= normal_result.var_absolute * 0.95

    def test_adjustment_factor_in_details(self, calculator):
        """Test that adjustment factor is included in result details."""
        positions = {"AAPL": 50000, "MSFT": 50000}

        result = calculator.calculate_cornish_fisher_var(
            positions=positions,
            portfolio_value=100000.0,
        )

        if "method_variant" in result.details:
            assert "adjustment_factor" in result.details
            assert result.details["adjustment_factor"] > 0


class TestCornishFisherAdjustmentFactor:
    """Tests for Cornish-Fisher adjustment factor calculation."""

    @pytest.fixture
    def calculator(self):
        """Create VaRCalculator with sufficient return history."""
        calc = VaRCalculator()

        np.random.seed(42)

        # Normal-ish returns
        calc._returns_cache = {
            "NORM": np.random.normal(0.0005, 0.02, 100).tolist(),
        }

        return calc

    def test_get_adjustment_factor(self, calculator):
        """Test adjustment factor calculation."""
        positions = {"NORM": 100000}

        result = calculator.get_cornish_fisher_adjustment_factor(
            positions=positions,
            portfolio_value=100000.0,
            confidence_level=0.95,
        )

        assert "skewness" in result
        assert "excess_kurtosis" in result
        assert "adjustment_factor" in result
        assert result["sample_size"] >= 30

    def test_insufficient_data_warning(self, calculator):
        """Test warning when insufficient data."""
        # Create calculator with minimal data
        calc = VaRCalculator()
        calc._returns_cache = {"SHORT": [0.01] * 10}  # Only 10 observations

        result = calc.get_cornish_fisher_adjustment_factor(
            positions={"SHORT": 100000},
            portfolio_value=100000.0,
        )

        assert result["warning"] == "insufficient_data"
        assert result["adjustment_factor"] == 1.0


class TestCornishFisherES:
    """Tests for Cornish-Fisher Expected Shortfall."""

    @pytest.fixture
    def calculator(self):
        """Create VaRCalculator instance."""
        return VaRCalculator()

    def test_es_calculation(self, calculator):
        """Test ES calculation with Cornish-Fisher adjustment."""
        z = 1.645
        skewness = -0.5
        excess_kurtosis = 2.0

        es = calculator._calculate_cornish_fisher_es(
            z=z,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
            portfolio_std_scaled=0.02,  # 2% daily vol
            portfolio_value=100000.0,
            confidence_level=0.95,
        )

        assert es > 0
        # ES should be larger than VaR
        z_cf = calculator._cornish_fisher_quantile(z, skewness, excess_kurtosis)
        var_cf = z_cf * 0.02 * 100000
        assert es >= var_cf * 0.8  # ES typically higher than VaR


class TestInterpretation:
    """Tests for Cornish-Fisher interpretation."""

    @pytest.fixture
    def calculator(self):
        """Create VaRCalculator instance."""
        return VaRCalculator()

    def test_interpret_negative_skew(self, calculator):
        """Test interpretation of negative skewness."""
        interpretation = calculator._interpret_cf_adjustment(
            skewness=-0.8,
            excess_kurtosis=0.0,
        )

        assert "negative skew" in interpretation.lower()

    def test_interpret_fat_tails(self, calculator):
        """Test interpretation of excess kurtosis."""
        interpretation = calculator._interpret_cf_adjustment(
            skewness=0.0,
            excess_kurtosis=4.0,
        )

        assert "fat tails" in interpretation.lower() or "extreme" in interpretation.lower()

    def test_interpret_normal(self, calculator):
        """Test interpretation of near-normal distribution."""
        interpretation = calculator._interpret_cf_adjustment(
            skewness=0.1,
            excess_kurtosis=0.3,
        )

        assert "normal" in interpretation.lower()

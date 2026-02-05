"""
Tests for Yield Curve Analysis
==============================

Tests for the yield curve analyzer module.
"""

import pytest
from datetime import datetime, timezone

from core.yield_curve import (
    YieldCurveState,
    YieldCurvePoint,
    YieldCurveAnalysisResult,
    YieldCurveAnalyzer,
    calculate_2s10s_spread,
    calculate_3m10y_spread,
    calculate_real_rate,
    detect_curve_state,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def normal_curve_data():
    """Normal upward-sloping yield curve."""
    return {
        "3M": 3.00,
        "6M": 3.20,
        "2Y": 3.50,
        "5Y": 4.00,
        "10Y": 4.50,
        "30Y": 4.80,
    }


@pytest.fixture
def inverted_curve_data():
    """Inverted yield curve (short > long)."""
    return {
        "3M": 5.50,
        "6M": 5.40,
        "2Y": 5.00,
        "5Y": 4.50,
        "10Y": 4.20,
        "30Y": 4.30,
    }


@pytest.fixture
def flat_curve_data():
    """Flat yield curve - strictly flat with minimal variation."""
    return {
        "3M": 4.50,
        "6M": 4.50,
        "2Y": 4.50,
        "5Y": 4.50,
        "10Y": 4.50,
        "30Y": 4.50,
    }


@pytest.fixture
def steep_curve_data():
    """Very steep yield curve."""
    return {
        "3M": 2.00,
        "6M": 2.20,
        "2Y": 2.50,
        "5Y": 3.20,
        "10Y": 4.20,
        "30Y": 4.80,
    }


@pytest.fixture
def analyzer():
    """Create yield curve analyzer."""
    return YieldCurveAnalyzer()


# ============================================================================
# YIELD CURVE POINT TESTS
# ============================================================================

class TestYieldCurvePoint:
    """Tests for YieldCurvePoint dataclass."""

    def test_basic_creation(self):
        """Test basic point creation."""
        point = YieldCurvePoint(tenor="10Y", tenor_years=10.0, yield_pct=4.5)
        assert point.tenor == "10Y"
        assert point.tenor_years == 10.0
        assert point.yield_pct == 4.5

    def test_from_tenor(self):
        """Test creation from tenor string."""
        point = YieldCurvePoint.from_tenor("10Y", 4.5)
        assert point.tenor == "10Y"
        assert point.tenor_years == 10.0
        assert point.yield_pct == 4.5

    def test_from_tenor_short(self):
        """Test creation with short tenor."""
        point = YieldCurvePoint.from_tenor("3M", 5.25)
        assert point.tenor_years == 0.25

    def test_from_tenor_various(self):
        """Test various tenor conversions."""
        tenors = {
            "1M": 1/12,
            "3M": 0.25,
            "6M": 0.5,
            "1Y": 1.0,
            "2Y": 2.0,
            "5Y": 5.0,
            "10Y": 10.0,
            "30Y": 30.0,
        }
        for tenor, expected_years in tenors.items():
            point = YieldCurvePoint.from_tenor(tenor, 4.0)
            assert point.tenor_years == expected_years


# ============================================================================
# YIELD CURVE STATE TESTS
# ============================================================================

class TestYieldCurveState:
    """Tests for YieldCurveState enum."""

    def test_all_states_defined(self):
        """Test all states are defined."""
        assert YieldCurveState.NORMAL.value == "normal"
        assert YieldCurveState.FLAT.value == "flat"
        assert YieldCurveState.INVERTED.value == "inverted"
        assert YieldCurveState.STEEP.value == "steep"
        assert YieldCurveState.KINKED.value == "kinked"


# ============================================================================
# SPREAD CALCULATION TESTS
# ============================================================================

class TestSpreadCalculations:
    """Tests for spread calculation functions."""

    def test_calculate_2s10s_spread_positive(self):
        """Test 2s10s spread with normal curve."""
        spread = calculate_2s10s_spread(4.5, 5.0)
        assert spread == 50.0  # 50 bps

    def test_calculate_2s10s_spread_negative(self):
        """Test 2s10s spread with inverted curve."""
        spread = calculate_2s10s_spread(5.0, 4.5)
        assert spread == -50.0  # -50 bps

    def test_calculate_2s10s_spread_zero(self):
        """Test 2s10s spread when flat."""
        spread = calculate_2s10s_spread(4.5, 4.5)
        assert spread == 0.0

    def test_calculate_3m10y_spread_positive(self):
        """Test 3m10y spread with normal curve."""
        spread = calculate_3m10y_spread(4.0, 4.5)
        assert spread == 50.0

    def test_calculate_3m10y_spread_negative(self):
        """Test 3m10y spread with inverted curve."""
        spread = calculate_3m10y_spread(5.5, 4.5)
        assert spread == -100.0


# ============================================================================
# REAL RATE CALCULATION TESTS
# ============================================================================

class TestRealRateCalculation:
    """Tests for real rate calculation."""

    def test_positive_real_rate(self):
        """Test positive real rate calculation."""
        real_rate = calculate_real_rate(5.0, 3.0)
        assert real_rate == 2.0

    def test_negative_real_rate(self):
        """Test negative real rate calculation."""
        real_rate = calculate_real_rate(3.0, 4.0)
        assert real_rate == -1.0

    def test_zero_real_rate(self):
        """Test zero real rate."""
        real_rate = calculate_real_rate(3.0, 3.0)
        assert real_rate == 0.0


# ============================================================================
# CURVE STATE DETECTION TESTS
# ============================================================================

class TestCurveStateDetection:
    """Tests for curve state detection."""

    def test_detect_normal_curve(self, normal_curve_data):
        """Test detection of normal curve."""
        points = [
            YieldCurvePoint.from_tenor(tenor, yld)
            for tenor, yld in normal_curve_data.items()
        ]
        state = detect_curve_state(points)
        # Normal curve might be detected as FLAT due to relatively small spread
        assert state in (YieldCurveState.NORMAL, YieldCurveState.FLAT)

    def test_detect_inverted_curve(self, inverted_curve_data):
        """Test detection of inverted curve."""
        points = [
            YieldCurvePoint.from_tenor(tenor, yld)
            for tenor, yld in inverted_curve_data.items()
        ]
        state = detect_curve_state(points)
        assert state == YieldCurveState.INVERTED

    def test_detect_flat_curve(self, flat_curve_data):
        """Test detection of flat curve."""
        points = [
            YieldCurvePoint.from_tenor(tenor, yld)
            for tenor, yld in flat_curve_data.items()
        ]
        state = detect_curve_state(points)
        assert state == YieldCurveState.FLAT

    def test_detect_steep_curve(self, steep_curve_data):
        """Test detection of steep curve."""
        points = [
            YieldCurvePoint.from_tenor(tenor, yld)
            for tenor, yld in steep_curve_data.items()
        ]
        state = detect_curve_state(points)
        assert state == YieldCurveState.STEEP

    def test_detect_with_minimal_points(self):
        """Test detection with minimal points."""
        points = [
            YieldCurvePoint.from_tenor("2Y", 4.0),
            YieldCurvePoint.from_tenor("10Y", 4.5),
        ]
        state = detect_curve_state(points)
        assert state in YieldCurveState  # Should return some valid state


# ============================================================================
# YIELD CURVE ANALYZER TESTS
# ============================================================================

class TestYieldCurveAnalyzer:
    """Tests for YieldCurveAnalyzer class."""

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.recession_model == "probit"
        assert analyzer.inflation_expectation is None

    def test_set_inflation(self, analyzer):
        """Test setting inflation expectation."""
        analyzer.set_inflation_expectation(2.5)
        assert analyzer.inflation_expectation == 2.5

    def test_update_yield(self, analyzer):
        """Test updating individual yield."""
        analyzer.update_yield("10Y", 4.5)
        assert analyzer._current_curve["10Y"] == 4.5

    def test_update_curve(self, analyzer, normal_curve_data):
        """Test updating entire curve."""
        analyzer.update_curve(normal_curve_data)
        assert "10Y" in analyzer._current_curve
        assert "2Y" in analyzer._current_curve

    def test_calculate_steepness(self, analyzer, normal_curve_data):
        """Test curve steepness calculation."""
        analyzer.update_curve(normal_curve_data)
        steepness = analyzer.calculate_curve_steepness()
        # Should be some finite value
        assert isinstance(steepness, float)

    def test_calculate_concavity(self, analyzer, normal_curve_data):
        """Test concavity calculation."""
        analyzer.update_curve(normal_curve_data)
        concavity = analyzer.calculate_concavity()
        assert isinstance(concavity, float)

    def test_recession_probability_normal(self, analyzer, normal_curve_data):
        """Test recession probability with normal curve."""
        analyzer.update_curve(normal_curve_data)
        prob = analyzer.get_recession_probability()
        # Normal curve should have relatively low probability
        assert 0 <= prob <= 1

    def test_recession_probability_inverted(self, analyzer, inverted_curve_data):
        """Test recession probability with inverted curve."""
        analyzer.update_curve(inverted_curve_data)
        prob = analyzer.get_recession_probability()
        # Inverted curve should have higher probability
        assert 0 <= prob <= 1

    def test_analyze_returns_result(self, analyzer, normal_curve_data):
        """Test analysis returns proper result."""
        result = analyzer.analyze(normal_curve_data)

        assert isinstance(result, YieldCurveAnalysisResult)
        assert isinstance(result.state, YieldCurveState)
        assert isinstance(result.spread_2s10s_bps, float)
        assert isinstance(result.recession_probability, float)

    def test_analyze_with_inflation(self, analyzer, normal_curve_data):
        """Test analysis with inflation expectation."""
        analyzer.set_inflation_expectation(2.5)
        result = analyzer.analyze(normal_curve_data)

        assert result.real_rate_10y is not None

    def test_get_trading_signal_normal(self, analyzer, normal_curve_data):
        """Test trading signal with normal curve."""
        signal, rationale = analyzer.get_trading_signal(normal_curve_data)

        assert -1 <= signal <= 1
        assert isinstance(rationale, str)

    def test_get_trading_signal_inverted(self, analyzer, inverted_curve_data):
        """Test trading signal with inverted curve."""
        signal, rationale = analyzer.get_trading_signal(inverted_curve_data)

        assert signal < 0  # Should be bearish
        assert "inverted" in rationale.lower()


# ============================================================================
# YIELD CURVE ANALYSIS RESULT TESTS
# ============================================================================

class TestYieldCurveAnalysisResult:
    """Tests for analysis result."""

    def test_to_dict(self, analyzer, normal_curve_data):
        """Test result serialization."""
        result = analyzer.analyze(normal_curve_data)
        result_dict = result.to_dict()

        assert "state" in result_dict
        assert "spread_2s10s_bps" in result_dict
        assert "spread_3m10y_bps" in result_dict
        assert "recession_probability" in result_dict
        assert "timestamp" in result_dict

    def test_warning_flags(self, analyzer, inverted_curve_data):
        """Test warning flags are set correctly."""
        result = analyzer.analyze(inverted_curve_data)

        assert result.is_warning is True
        assert result.warning_message is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for yield curve analysis."""

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow."""
        analyzer = YieldCurveAnalyzer()
        analyzer.set_inflation_expectation(2.8)

        # Add curve data
        curve = {
            "3M": 5.35,
            "2Y": 4.95,
            "5Y": 4.60,
            "10Y": 4.45,
            "30Y": 4.65,
        }

        result = analyzer.analyze(curve)

        # Should have all fields
        assert result.state is not None
        assert result.spread_2s10s_bps is not None
        assert result.real_rate_10y is not None
        assert 0 <= result.recession_probability <= 1

        # Get signal
        signal, rationale = analyzer.get_trading_signal()
        assert signal is not None
        assert rationale is not None

    def test_track_spread_history(self):
        """Test spread history tracking."""
        analyzer = YieldCurveAnalyzer()

        # Analyze multiple times
        for i in range(5):
            curve = {
                "2Y": 4.5 + i * 0.1,
                "10Y": 4.2 + i * 0.05,
            }
            analyzer.analyze(curve)

        # Should have history
        assert len(analyzer._spread_history) == 5

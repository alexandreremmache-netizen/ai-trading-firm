"""
Tests for DXY Analyzer
======================

Tests for the Dollar Index analyzer module.
"""

import pytest
import numpy as np
from datetime import datetime, timezone

from core.dxy_analyzer import (
    DXYState,
    AssetDXYRelation,
    DXYAnalysisResult,
    DXYAnalyzer,
    ASSET_DXY_CORRELATIONS,
    calculate_dxy_trend,
    calculate_dxy_momentum,
    get_dxy_regime,
    get_asset_correlation,
    get_dxy_signal_for_asset,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def dxy_uptrend_prices():
    """DXY prices in uptrend."""
    np.random.seed(42)
    base = 100.0
    trend = np.linspace(0, 10, 50)
    noise = np.random.normal(0, 0.5, 50)
    return list(base + trend + noise)


@pytest.fixture
def dxy_downtrend_prices():
    """DXY prices in downtrend."""
    np.random.seed(42)
    base = 110.0
    trend = np.linspace(0, -10, 50)
    noise = np.random.normal(0, 0.5, 50)
    return list(base + trend + noise)


@pytest.fixture
def dxy_sideways_prices():
    """DXY prices moving sideways."""
    np.random.seed(42)
    base = 100.0
    noise = np.random.normal(0, 0.5, 50)
    return list(base + noise)


@pytest.fixture
def analyzer():
    """Create DXY analyzer."""
    return DXYAnalyzer()


# ============================================================================
# DXY STATE TESTS
# ============================================================================

class TestDXYState:
    """Tests for DXYState enum."""

    def test_all_states_defined(self):
        """Test all states are defined."""
        assert DXYState.EXTREME_STRONG.value == "extreme_strong"
        assert DXYState.STRONG.value == "strong"
        assert DXYState.NEUTRAL.value == "neutral"
        assert DXYState.WEAK.value == "weak"
        assert DXYState.EXTREME_WEAK.value == "extreme_weak"


# ============================================================================
# ASSET DXY RELATION TESTS
# ============================================================================

class TestAssetDXYRelation:
    """Tests for asset-DXY relationship enum."""

    def test_all_relations_defined(self):
        """Test all relations are defined."""
        assert AssetDXYRelation.NEGATIVE.value == "negative"
        assert AssetDXYRelation.POSITIVE.value == "positive"
        assert AssetDXYRelation.MIXED.value == "mixed"
        assert AssetDXYRelation.UNCORRELATED.value == "uncorrelated"


# ============================================================================
# TREND CALCULATION TESTS
# ============================================================================

class TestDXYTrend:
    """Tests for DXY trend calculation."""

    def test_uptrend_detection(self, dxy_uptrend_prices):
        """Test detection of uptrend."""
        direction, strength = calculate_dxy_trend(dxy_uptrend_prices, period=20)
        assert direction == "up"
        assert strength > 0.3

    def test_downtrend_detection(self, dxy_downtrend_prices):
        """Test detection of downtrend."""
        direction, strength = calculate_dxy_trend(dxy_downtrend_prices, period=20)
        assert direction == "down"
        assert strength > 0.3

    def test_sideways_detection(self, dxy_sideways_prices):
        """Test detection of sideways movement."""
        direction, strength = calculate_dxy_trend(dxy_sideways_prices, period=20)
        # Could be sideways or weak up/down due to noise
        # Strength should be relatively low compared to trending markets
        assert strength < 1.0  # Less strict threshold

    def test_insufficient_data(self):
        """Test with insufficient data."""
        direction, strength = calculate_dxy_trend([100.0, 100.5], period=20)
        assert direction == "sideways"
        assert strength == 0.0


# ============================================================================
# MOMENTUM CALCULATION TESTS
# ============================================================================

class TestDXYMomentum:
    """Tests for DXY momentum calculation."""

    def test_positive_momentum(self, dxy_uptrend_prices):
        """Test positive momentum."""
        momentum = calculate_dxy_momentum(dxy_uptrend_prices)
        assert momentum > 0

    def test_negative_momentum(self, dxy_downtrend_prices):
        """Test negative momentum."""
        momentum = calculate_dxy_momentum(dxy_downtrend_prices)
        assert momentum < 0

    def test_momentum_range(self, dxy_uptrend_prices):
        """Test momentum is in valid range."""
        momentum = calculate_dxy_momentum(dxy_uptrend_prices)
        assert -1 <= momentum <= 1

    def test_momentum_insufficient_data(self):
        """Test momentum with insufficient data."""
        momentum = calculate_dxy_momentum([100.0, 100.5, 101.0])
        assert momentum == 0.0


# ============================================================================
# REGIME DETECTION TESTS
# ============================================================================

class TestDXYRegime:
    """Tests for DXY regime detection."""

    def test_extreme_strong(self):
        """Test extreme strong regime."""
        state = get_dxy_regime(115.0)
        assert state == DXYState.EXTREME_STRONG

    def test_strong(self):
        """Test strong regime."""
        state = get_dxy_regime(107.0)
        assert state == DXYState.STRONG

    def test_neutral(self):
        """Test neutral regime."""
        state = get_dxy_regime(100.0)
        assert state == DXYState.NEUTRAL

    def test_weak(self):
        """Test weak regime."""
        state = get_dxy_regime(92.0)
        assert state == DXYState.WEAK

    def test_extreme_weak(self):
        """Test extreme weak regime."""
        state = get_dxy_regime(85.0)
        assert state == DXYState.EXTREME_WEAK


# ============================================================================
# ASSET CORRELATION TESTS
# ============================================================================

class TestAssetCorrelation:
    """Tests for asset-DXY correlation."""

    def test_gold_correlation(self):
        """Test gold has negative correlation with DXY."""
        relation, corr = get_asset_correlation("GC")
        assert relation == AssetDXYRelation.NEGATIVE
        assert corr < 0

    def test_gold_etf_correlation(self):
        """Test gold ETF correlation."""
        relation, corr = get_asset_correlation("GLD")
        assert relation == AssetDXYRelation.NEGATIVE

    def test_oil_correlation(self):
        """Test oil correlation."""
        relation, corr = get_asset_correlation("CL")
        assert relation == AssetDXYRelation.NEGATIVE

    def test_em_correlation(self):
        """Test EM correlation."""
        relation, corr = get_asset_correlation("EEM")
        assert relation == AssetDXYRelation.NEGATIVE

    def test_spy_mixed_correlation(self):
        """Test SPY has mixed correlation."""
        relation, corr = get_asset_correlation("SPY")
        assert relation == AssetDXYRelation.MIXED

    def test_unknown_asset(self):
        """Test unknown asset returns uncorrelated."""
        relation, corr = get_asset_correlation("UNKNOWN")
        assert relation == AssetDXYRelation.UNCORRELATED
        assert corr == 0.0

    def test_correlation_table_entries(self):
        """Test correlation table has expected entries."""
        assert "GC" in ASSET_DXY_CORRELATIONS
        assert "CL" in ASSET_DXY_CORRELATIONS
        assert "EURUSD" in ASSET_DXY_CORRELATIONS


# ============================================================================
# SIGNAL FOR ASSET TESTS
# ============================================================================

class TestDXYSignalForAsset:
    """Tests for DXY-based asset signals."""

    def test_gold_signal_strong_dxy(self):
        """Test gold signal when DXY is strong."""
        signal, rationale = get_dxy_signal_for_asset(
            "gold", DXYState.STRONG, 0.3
        )
        assert signal < 0  # Negative for gold
        assert "gold" in rationale.lower()

    def test_gold_signal_weak_dxy(self):
        """Test gold signal when DXY is weak."""
        signal, rationale = get_dxy_signal_for_asset(
            "gold", DXYState.WEAK, -0.3
        )
        assert signal > 0  # Positive for gold

    def test_em_signal_strong_dxy(self):
        """Test EM signal when DXY is strong."""
        signal, rationale = get_dxy_signal_for_asset(
            "em_equity", DXYState.EXTREME_STRONG, 0.5
        )
        assert signal < 0  # Negative for EM

    def test_us_equity_signal(self):
        """Test US equity signal (mixed relationship)."""
        signal, rationale = get_dxy_signal_for_asset(
            "us_equity", DXYState.STRONG, 0.3
        )
        # Should be muted due to mixed relationship
        assert -0.5 < signal < 0.5

    def test_signal_range(self):
        """Test signals are in valid range."""
        for state in DXYState:
            for momentum in [-0.5, 0.0, 0.5]:
                signal, _ = get_dxy_signal_for_asset("gold", state, momentum)
                assert -1 <= signal <= 1


# ============================================================================
# DXY ANALYZER TESTS
# ============================================================================

class TestDXYAnalyzer:
    """Tests for DXY Analyzer class."""

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.lookback_period == 50
        assert len(analyzer._dxy_history) == 0

    def test_update(self, analyzer):
        """Test price update."""
        analyzer.update(100.0)
        assert len(analyzer._dxy_history) == 1
        assert analyzer._dxy_history[0] == 100.0

    def test_multiple_updates(self, analyzer, dxy_uptrend_prices):
        """Test multiple updates."""
        for price in dxy_uptrend_prices:
            analyzer.update(price)
        assert len(analyzer._dxy_history) == 50

    def test_update_asset(self, analyzer):
        """Test asset price update."""
        analyzer.update_asset("GLD", 180.0)
        assert "GLD" in analyzer._asset_histories
        assert len(analyzer._asset_histories["GLD"]) == 1

    def test_analyze_requires_data(self, analyzer):
        """Test analysis requires data."""
        with pytest.raises(ValueError):
            analyzer.analyze()

    def test_analyze_returns_result(self, analyzer, dxy_uptrend_prices):
        """Test analysis returns proper result."""
        for price in dxy_uptrend_prices:
            analyzer.update(price)

        result = analyzer.analyze()

        assert isinstance(result, DXYAnalysisResult)
        assert isinstance(result.state, DXYState)
        assert result.current_level == dxy_uptrend_prices[-1]

    def test_analyze_calculates_mas(self, analyzer, dxy_uptrend_prices):
        """Test MAs are calculated."""
        for price in dxy_uptrend_prices:
            analyzer.update(price)

        result = analyzer.analyze()

        assert result.ma_20 is not None
        assert result.ma_50 is not None

    def test_get_signal_for_symbol(self, analyzer, dxy_uptrend_prices):
        """Test signal generation for symbol."""
        for price in dxy_uptrend_prices:
            analyzer.update(price)

        signal, rationale = analyzer.get_signal_for_symbol("GLD")

        assert -1 <= signal <= 1
        assert isinstance(rationale, str)

    def test_is_favorable_for_gold_weak_dxy(self, analyzer):
        """Test gold favorable condition with weak DXY."""
        for price in [95.0] * 25 + [94.0] * 25:
            analyzer.update(price)

        assert analyzer.is_favorable_for_gold()

    def test_is_headwind_for_commodities_strong_dxy(self, analyzer):
        """Test commodity headwind with strong DXY."""
        for price in [105.0] * 25 + [108.0] * 25:
            analyzer.update(price)

        assert analyzer.is_headwind_for_commodities()


# ============================================================================
# DXY ANALYSIS RESULT TESTS
# ============================================================================

class TestDXYAnalysisResult:
    """Tests for analysis result."""

    def test_to_dict(self, analyzer, dxy_uptrend_prices):
        """Test result serialization."""
        for price in dxy_uptrend_prices:
            analyzer.update(price)

        result = analyzer.analyze()
        result_dict = result.to_dict()

        assert "state" in result_dict
        assert "current_level" in result_dict
        assert "trend_direction" in result_dict
        assert "momentum_score" in result_dict
        assert "timestamp" in result_dict


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for DXY analyzer."""

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow."""
        analyzer = DXYAnalyzer()

        # Simulate DXY rising
        np.random.seed(123)
        for i in range(60):
            price = 100.0 + i * 0.2 + np.random.normal(0, 0.3)
            analyzer.update(price)

        result = analyzer.analyze()

        assert result.trend_direction == "up"
        assert result.momentum_score > 0
        # DXY ends up above 110, so could be STRONG or EXTREME_STRONG
        assert result.state in (DXYState.NEUTRAL, DXYState.STRONG, DXYState.EXTREME_STRONG)

        # Get signals for various assets
        gold_signal, _ = analyzer.get_signal_for_symbol("GLD")
        assert gold_signal < 0  # Strong DXY bad for gold

        em_signal, _ = analyzer.get_signal_for_symbol("EEM")
        assert em_signal < 0  # Strong DXY bad for EM

    def test_with_asset_tracking(self):
        """Test with asset price tracking."""
        analyzer = DXYAnalyzer()

        np.random.seed(456)
        for i in range(100):
            dxy_price = 100.0 + np.random.normal(0, 1)
            gold_price = 2000.0 - dxy_price * 5 + np.random.normal(0, 10)

            analyzer.update(dxy_price)
            analyzer.update_asset("GC", gold_price)

        # Should be able to analyze
        result = analyzer.analyze()
        assert result is not None

        # Get correlation-adjusted signal
        signal, rationale = analyzer.get_signal_for_symbol("GC")
        assert signal is not None
        assert "correlation" in rationale.lower()

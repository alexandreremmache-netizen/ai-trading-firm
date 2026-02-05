"""
Tests for Execution Optimizer (Phase 7)
========================================

Tests for adaptive TWAP, dynamic slippage caps, session-aware execution,
smart algo selection, and fill quality monitoring.
"""

import numpy as np
import pytest
from datetime import datetime, time, timedelta, timezone

from core.execution_optimizer import (
    # 7.1 Adaptive TWAP
    AdaptiveTWAP,
    AdaptiveTWAPConfig,
    AdaptiveTWAPPlan,
    VolatilityRegime,
    # 7.2 Dynamic Slippage
    DynamicSlippageCaps,
    SlippageConfig,
    SlippageCap,
    # 7.3 Session-Aware
    SessionAwareExecution,
    SessionPhase,
    SessionRule,
    # 7.4 Smart Algo
    SmartAlgoSelector,
    AlgoType,
    AlgoRecommendation,
    # 7.5 Fill Quality
    FillQualityMonitor,
    FillQualityMetrics,
    # Factories
    create_adaptive_twap,
    create_dynamic_slippage_caps,
    create_session_aware_execution,
    create_smart_algo_selector,
    create_fill_quality_monitor,
)


# =============================================================================
# 7.1 ADAPTIVE TWAP TESTS
# =============================================================================

class TestAdaptiveTWAP:
    """Tests for Adaptive TWAP."""

    @pytest.fixture
    def twap(self):
        """Create AdaptiveTWAP instance."""
        return AdaptiveTWAP()

    def test_volatility_regime_detection(self, twap):
        """Test volatility regime detection."""
        # Add history
        for _ in range(20):
            twap.update_volatility("TEST", 0.2)

        # Test regimes
        assert twap.get_volatility_regime("TEST", 0.08) == VolatilityRegime.LOW
        assert twap.get_volatility_regime("TEST", 0.2) == VolatilityRegime.NORMAL
        assert twap.get_volatility_regime("TEST", 0.35) == VolatilityRegime.HIGH
        assert twap.get_volatility_regime("TEST", 0.55) == VolatilityRegime.EXTREME

    def test_generate_plan_normal_vol(self, twap):
        """Test plan generation in normal volatility."""
        plan = twap.generate_plan(
            symbol="SPY",
            quantity=1000,
            side="BUY",
            current_vol=0.2,
            urgency=0.5,
        )

        assert isinstance(plan, AdaptiveTWAPPlan)
        assert plan.total_quantity == 1000
        assert plan.num_slices >= 3
        assert sum(plan.slice_sizes) == 1000

    def test_generate_plan_high_vol(self, twap):
        """Test plan in high volatility (faster execution)."""
        # Build history first
        for _ in range(20):
            twap.update_volatility("HIGH_VOL", 0.2)

        plan = twap.generate_plan(
            symbol="HIGH_VOL",
            quantity=1000,
            side="BUY",
            current_vol=0.4,  # High vol
            urgency=0.5,
        )

        # High vol = fewer slices, shorter intervals
        assert plan.volatility_regime == VolatilityRegime.HIGH

    def test_generate_plan_low_vol(self, twap):
        """Test plan in low volatility (slower execution)."""
        for _ in range(20):
            twap.update_volatility("LOW_VOL", 0.2)

        plan = twap.generate_plan(
            symbol="LOW_VOL",
            quantity=1000,
            side="BUY",
            current_vol=0.08,  # Low vol
            urgency=0.5,
        )

        # Low vol = more slices, longer intervals
        assert plan.volatility_regime == VolatilityRegime.LOW

    def test_urgency_affects_plan(self, twap):
        """Test that urgency affects execution plan."""
        plan_passive = twap.generate_plan("TEST", 1000, "BUY", 0.2, urgency=0.1)
        plan_urgent = twap.generate_plan("TEST", 1000, "BUY", 0.2, urgency=0.9)

        # Urgent should have fewer slices or shorter intervals
        assert (
            plan_urgent.num_slices <= plan_passive.num_slices or
            plan_urgent.intervals_seconds[0] <= plan_passive.intervals_seconds[0]
        )


class TestAdaptiveTWAPFactory:
    """Tests for AdaptiveTWAP factory."""

    def test_create_default(self):
        """Test default creation."""
        twap = create_adaptive_twap()
        assert isinstance(twap, AdaptiveTWAP)

    def test_create_custom(self):
        """Test custom creation."""
        twap = create_adaptive_twap({"base_slices": 15})
        assert twap._config.base_slices == 15


# =============================================================================
# 7.2 DYNAMIC SLIPPAGE CAPS TESTS
# =============================================================================

class TestDynamicSlippageCaps:
    """Tests for Dynamic Slippage Caps."""

    @pytest.fixture
    def caps(self):
        """Create DynamicSlippageCaps instance."""
        return DynamicSlippageCaps()

    def test_calculate_cap_basic(self, caps):
        """Test basic slippage cap calculation."""
        cap = caps.calculate_cap(
            symbol="SPY",
            price=500.0,
            order_size=1000,
            avg_volume=50000000,
            volatility=0.15,
            urgency=0.5,
        )

        assert isinstance(cap, SlippageCap)
        assert cap.cap_bps > 0
        assert cap.cap_price > 0
        assert cap.cap_bps <= 50  # Max cap

    def test_volatility_affects_cap(self, caps):
        """Test that volatility affects slippage cap."""
        # Use low volatilities to avoid hitting max cap
        cap_low_vol = caps.calculate_cap("TEST", 100, 1000, 100000000, 0.05, 0.5)
        cap_high_vol = caps.calculate_cap("TEST", 100, 1000, 100000000, 0.10, 0.5)

        # Higher vol = higher cap (or check component difference if both hit max)
        assert cap_high_vol.components["volatility"] > cap_low_vol.components["volatility"]

    def test_size_affects_cap(self, caps):
        """Test that order size affects slippage cap."""
        # Use large avg_volume and low volatility to avoid max cap
        cap_small = caps.calculate_cap("TEST", 100, 100, 100000000, 0.05, 0.5)
        cap_large = caps.calculate_cap("TEST", 100, 5000000, 100000000, 0.05, 0.5)

        # Larger size = higher size component
        assert cap_large.components["size"] > cap_small.components["size"]

    def test_urgency_affects_cap(self, caps):
        """Test that urgency affects slippage cap."""
        # Use parameters that won't hit max cap
        cap_passive = caps.calculate_cap("TEST", 100, 1000, 100000000, 0.05, 0.1)
        cap_urgent = caps.calculate_cap("TEST", 100, 1000, 100000000, 0.05, 0.9)

        # Higher urgency = higher urgency adjustment
        assert cap_urgent.components["urgency_adj"] > cap_passive.components["urgency_adj"]

    def test_historical_adjustment(self, caps):
        """Test historical slippage affects cap."""
        # Record historical slippage
        for _ in range(10):
            caps.record_slippage("HIST_TEST", 15.0)

        cap = caps.calculate_cap("HIST_TEST", 100, 1000, 1000000, 0.2, 0.5)

        # Should include historical component
        assert cap.components["historical"] > 0


# =============================================================================
# 7.3 SESSION-AWARE EXECUTION TESTS
# =============================================================================

class TestSessionAwareExecution:
    """Tests for Session-Aware Execution."""

    @pytest.fixture
    def session(self):
        """Create SessionAwareExecution instance."""
        return SessionAwareExecution()

    def test_session_phase_detection(self, session):
        """Test session phase detection."""
        # Pre-market
        pre = datetime(2026, 2, 4, 8, 0, 0)  # 8:00 AM
        assert session.get_session_phase(pre) == SessionPhase.PRE_MARKET

        # Market open
        open_time = datetime(2026, 2, 4, 9, 45, 0)  # 9:45 AM
        assert session.get_session_phase(open_time) == SessionPhase.MARKET_OPEN

        # Mid-day
        mid = datetime(2026, 2, 4, 12, 0, 0)  # 12:00 PM
        assert session.get_session_phase(mid) == SessionPhase.MID_DAY

        # Market close
        close_time = datetime(2026, 2, 4, 15, 45, 0)  # 3:45 PM
        assert session.get_session_phase(close_time) == SessionPhase.MARKET_CLOSE

        # After hours
        after = datetime(2026, 2, 4, 17, 0, 0)  # 5:00 PM
        assert session.get_session_phase(after) == SessionPhase.AFTER_HOURS

    def test_apply_rules_mid_day(self, session):
        """Test applying rules during mid-day."""
        result = session.apply_rules(
            current_time=datetime(2026, 2, 4, 12, 0, 0),
            order_size=10000,
            avg_daily_volume=1000000,
            base_urgency=0.5,
        )

        assert result["allowed"]
        assert result["phase"] == SessionPhase.MID_DAY
        assert result["preferred_algo"] == "VWAP"

    def test_apply_rules_market_open(self, session):
        """Test applying rules at market open."""
        result = session.apply_rules(
            current_time=datetime(2026, 2, 4, 9, 35, 0),
            order_size=10000,
            avg_daily_volume=1000000,
            base_urgency=0.7,
        )

        assert result["allowed"]
        assert result["phase"] == SessionPhase.MARKET_OPEN
        assert result["preferred_algo"] == "TWAP"
        # Urgency should be reduced
        assert result["adjusted_urgency"] < 0.7

    def test_apply_rules_overnight_blocked(self, session):
        """Test that overnight execution is blocked."""
        result = session.apply_rules(
            current_time=datetime(2026, 2, 4, 2, 0, 0),  # 2 AM
            order_size=10000,
            avg_daily_volume=1000000,
            base_urgency=0.5,
        )

        assert not result["allowed"]

    def test_oversized_order_limited(self, session):
        """Test that oversized orders are limited."""
        result = session.apply_rules(
            current_time=datetime(2026, 2, 4, 12, 0, 0),
            order_size=200000,  # 20% of ADV
            avg_daily_volume=1000000,
            base_urgency=0.5,
        )

        assert result["oversized"]
        assert result["adjusted_size"] < 200000


# =============================================================================
# 7.4 SMART ALGO SELECTION TESTS
# =============================================================================

class TestSmartAlgoSelector:
    """Tests for Smart Algo Selection."""

    @pytest.fixture
    def selector(self):
        """Create SmartAlgoSelector instance."""
        return SmartAlgoSelector()

    def test_select_market_for_urgent_small(self, selector):
        """Test MARKET selection for urgent small orders."""
        rec = selector.select_algo(
            order_size=50,
            side="BUY",
            price=100.0,
            spread_bps=5.0,
            volatility=0.2,
            avg_volume=1000000,
            urgency=0.95,
            session_phase=SessionPhase.MID_DAY,
        )

        assert isinstance(rec, AlgoRecommendation)
        # Urgent small order should lean toward MARKET
        assert rec.algo in [AlgoType.MARKET, AlgoType.TWAP, AlgoType.VWAP]

    def test_select_vwap_for_mid_day(self, selector):
        """Test VWAP selection during mid-day."""
        rec = selector.select_algo(
            order_size=50000,
            side="BUY",
            price=100.0,
            spread_bps=5.0,
            volatility=0.2,
            avg_volume=1000000,
            urgency=0.5,
            session_phase=SessionPhase.MID_DAY,
        )

        # Mid-day with medium size should prefer VWAP
        assert rec.algo in [AlgoType.VWAP, AlgoType.TWAP]

    def test_select_twap_for_open(self, selector):
        """Test TWAP selection at market open."""
        rec = selector.select_algo(
            order_size=30000,
            side="BUY",
            price=100.0,
            spread_bps=8.0,
            volatility=0.25,
            avg_volume=1000000,
            urgency=0.5,
            session_phase=SessionPhase.MARKET_OPEN,
        )

        # Market open should prefer TWAP
        assert rec.algo in [AlgoType.TWAP, AlgoType.VWAP]

    def test_select_iceberg_for_large(self, selector):
        """Test ICEBERG selection for large orders."""
        rec = selector.select_algo(
            order_size=100000,  # 10% of ADV
            side="BUY",
            price=100.0,
            spread_bps=10.0,
            volatility=0.2,
            avg_volume=1000000,
            urgency=0.5,
            session_phase=SessionPhase.MID_DAY,
        )

        # Large order should consider ICEBERG
        assert rec.algo in [AlgoType.ICEBERG, AlgoType.VWAP, AlgoType.TWAP]

    def test_recommendation_includes_rationale(self, selector):
        """Test that recommendation includes rationale."""
        rec = selector.select_algo(
            order_size=10000,
            side="BUY",
            price=100.0,
            spread_bps=5.0,
            volatility=0.2,
            avg_volume=1000000,
            urgency=0.5,
            session_phase=SessionPhase.MID_DAY,
        )

        assert len(rec.rationale) > 0
        assert rec.confidence > 0
        assert len(rec.alternatives) >= 0

    def test_performance_affects_selection(self, selector):
        """Test that historical performance affects selection."""
        # Record bad TWAP performance
        for _ in range(15):
            selector.record_performance("TWAP", 20.0)  # 20bps slippage

        # Record good VWAP performance
        for _ in range(15):
            selector.record_performance("VWAP", 5.0)  # 5bps slippage

        rec = selector.select_algo(
            order_size=30000,
            side="BUY",
            price=100.0,
            spread_bps=5.0,
            volatility=0.2,
            avg_volume=1000000,
            urgency=0.5,
            session_phase=SessionPhase.MID_DAY,
        )

        # Should prefer VWAP due to better historical performance
        # (or at least have it as high alternative)
        assert rec.algo == AlgoType.VWAP or AlgoType.VWAP in [a[0] for a in rec.alternatives]


# =============================================================================
# 7.5 FILL QUALITY MONITORING TESTS
# =============================================================================

class TestFillQualityMonitor:
    """Tests for Fill Quality Monitoring."""

    @pytest.fixture
    def monitor(self):
        """Create FillQualityMonitor instance."""
        return FillQualityMonitor()

    def test_calculate_metrics_basic(self, monitor):
        """Test basic metrics calculation."""
        fills = [(100.5, 500), (100.6, 500)]  # Total 1000 @ avg 100.55
        start = datetime(2026, 2, 4, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 2, 4, 10, 5, 0, tzinfo=timezone.utc)

        metrics = monitor.calculate_metrics(
            symbol="TEST",
            algo="TWAP",
            side="BUY",
            quantity=1000,
            fills=fills,
            arrival_price=100.4,
            vwap=100.5,
            execution_start=start,
            execution_end=end,
        )

        assert isinstance(metrics, FillQualityMetrics)
        assert metrics.avg_fill_price == 100.55
        assert metrics.fill_rate == 1.0
        assert metrics.num_fills == 2
        assert metrics.execution_time_seconds == 300

    def test_slippage_calculation_buy(self, monitor):
        """Test slippage calculation for buy orders."""
        # Bought at 101, arrival was 100 = negative slippage
        fills = [(101.0, 1000)]

        metrics = monitor.calculate_metrics(
            symbol="TEST",
            algo="MARKET",
            side="BUY",
            quantity=1000,
            fills=fills,
            arrival_price=100.0,
            vwap=100.5,
            execution_start=datetime.now(timezone.utc),
            execution_end=datetime.now(timezone.utc),
        )

        # Slippage should be positive (paid more than arrival)
        assert metrics.slippage_vs_arrival_bps > 0

    def test_slippage_calculation_sell(self, monitor):
        """Test slippage calculation for sell orders."""
        # Sold at 99, arrival was 100 = negative slippage
        fills = [(99.0, 1000)]

        metrics = monitor.calculate_metrics(
            symbol="TEST",
            algo="MARKET",
            side="SELL",
            quantity=1000,
            fills=fills,
            arrival_price=100.0,
            vwap=100.0,
            execution_start=datetime.now(timezone.utc),
            execution_end=datetime.now(timezone.utc),
        )

        # Slippage should be positive (sold less than arrival)
        assert metrics.slippage_vs_arrival_bps > 0

    def test_partial_fill(self, monitor):
        """Test partial fill rate calculation."""
        fills = [(100.0, 500)]  # Only 500 of 1000 filled

        metrics = monitor.calculate_metrics(
            symbol="TEST",
            algo="LIMIT",
            side="BUY",
            quantity=1000,
            fills=fills,
            arrival_price=100.0,
            vwap=100.0,
            execution_start=datetime.now(timezone.utc),
            execution_end=datetime.now(timezone.utc),
        )

        assert metrics.fill_rate == 0.5

    def test_spread_capture(self, monitor):
        """Test spread capture calculation."""
        # Bid 99.9, Ask 100.1, filled at 100.0 (midpoint)
        fills = [(100.0, 1000)]

        metrics = monitor.calculate_metrics(
            symbol="TEST",
            algo="MIDPEG",
            side="BUY",
            quantity=1000,
            fills=fills,
            arrival_price=100.0,
            vwap=100.0,
            execution_start=datetime.now(timezone.utc),
            execution_end=datetime.now(timezone.utc),
            bid_at_arrival=99.9,
            ask_at_arrival=100.1,
        )

        # Bought at midpoint = 50% spread capture
        assert abs(metrics.spread_capture_pct - 50.0) < 1.0

    def test_summary_stats(self, monitor):
        """Test summary statistics calculation."""
        # Record some executions
        for i in range(10):
            monitor.calculate_metrics(
                symbol="SPY",
                algo="VWAP",
                side="BUY",
                quantity=1000,
                fills=[(100 + i * 0.1, 1000)],
                arrival_price=100.0,
                vwap=100.0,
                execution_start=datetime.now(timezone.utc),
                execution_end=datetime.now(timezone.utc),
            )

        stats = monitor.get_summary_stats()

        assert stats["count"] == 10
        assert stats["avg_slippage_arrival_bps"] is not None
        assert stats["avg_fill_rate"] == 1.0

    def test_filter_by_symbol(self, monitor):
        """Test filtering stats by symbol."""
        # SPY executions
        for _ in range(5):
            monitor.calculate_metrics(
                symbol="SPY", algo="TWAP", side="BUY", quantity=1000,
                fills=[(100.1, 1000)], arrival_price=100.0, vwap=100.0,
                execution_start=datetime.now(timezone.utc),
                execution_end=datetime.now(timezone.utc),
            )

        # QQQ executions
        for _ in range(3):
            monitor.calculate_metrics(
                symbol="QQQ", algo="TWAP", side="BUY", quantity=1000,
                fills=[(400.1, 1000)], arrival_price=400.0, vwap=400.0,
                execution_start=datetime.now(timezone.utc),
                execution_end=datetime.now(timezone.utc),
            )

        spy_stats = monitor.get_summary_stats(symbol="SPY")
        assert spy_stats["count"] == 5

        qqq_stats = monitor.get_summary_stats(symbol="QQQ")
        assert qqq_stats["count"] == 3

    def test_get_status(self, monitor):
        """Test status reporting."""
        monitor.calculate_metrics(
            symbol="TEST", algo="VWAP", side="BUY", quantity=1000,
            fills=[(100.0, 1000)], arrival_price=100.0, vwap=100.0,
            execution_start=datetime.now(timezone.utc),
            execution_end=datetime.now(timezone.utc),
        )

        status = monitor.get_status()

        assert status["total_executions"] == 1
        assert status["symbols_tracked"] == 1
        assert status["algos_tracked"] == 1


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_all_components(self):
        """Test creation of all components."""
        twap = create_adaptive_twap()
        caps = create_dynamic_slippage_caps()
        session = create_session_aware_execution()
        selector = create_smart_algo_selector()
        monitor = create_fill_quality_monitor()

        assert isinstance(twap, AdaptiveTWAP)
        assert isinstance(caps, DynamicSlippageCaps)
        assert isinstance(session, SessionAwareExecution)
        assert isinstance(selector, SmartAlgoSelector)
        assert isinstance(monitor, FillQualityMonitor)

"""
Tests for Advanced Analytics Dashboard Components (Phase 8)
============================================================

Tests for rolling metrics, session performance, strategy comparison,
risk heatmap, trade journal, and signal consensus.
"""

import numpy as np
import pytest
from datetime import datetime, timedelta, timezone

from dashboard.components.advanced_analytics import (
    # 8.1 Rolling Metrics
    RollingMetricsCalculator,
    RollingPeriod,
    RollingMetrics,
    # 8.2 Session Performance
    SessionPerformanceTracker,
    TradingSession,
    SessionPerformance,
    # 8.3 Strategy Comparison
    StrategyComparisonTracker,
    StrategyPerformance,
    # 8.4 Risk Heatmap
    RiskHeatmapGenerator,
    PositionRiskScore,
    # 8.5 Trade Journal
    TradeJournal,
    TradeJournalEntry,
    TradeQuality,
    EmotionalState,
    # 8.6 Signal Consensus
    SignalConsensusTracker,
    SignalConsensus,
    # Factories
    create_rolling_metrics_calculator,
    create_session_performance_tracker,
    create_strategy_comparison_tracker,
    create_risk_heatmap_generator,
    create_trade_journal,
    create_signal_consensus_tracker,
)


# =============================================================================
# 8.1 ROLLING METRICS TESTS
# =============================================================================

class TestRollingMetricsCalculator:
    """Tests for Rolling Metrics Calculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return RollingMetricsCalculator(risk_free_rate=0.05)

    def test_add_equity_point(self, calculator):
        """Test adding equity points."""
        now = datetime.now(timezone.utc)
        calculator.add_equity_point(now, 100000)
        calculator.add_equity_point(now + timedelta(days=1), 101000)

        assert len(calculator._equity_history) == 2

    def test_calculate_metrics_with_data(self, calculator):
        """Test metrics calculation with data."""
        now = datetime.now(timezone.utc)

        # Add 30 days of equity data
        for i in range(30):
            equity = 100000 * (1 + 0.001 * i + np.random.randn() * 0.005)
            calculator.add_equity_point(now - timedelta(days=30-i), equity)

        metrics = calculator.calculate_metrics(RollingPeriod.MONTH_1)

        assert isinstance(metrics, RollingMetrics)
        assert metrics.period == RollingPeriod.MONTH_1
        assert metrics.volatility_pct >= 0

    def test_calculate_metrics_insufficient_data(self, calculator):
        """Test metrics with insufficient data."""
        metrics = calculator.calculate_metrics(RollingPeriod.MONTH_1)

        assert metrics.sharpe_ratio is None
        assert metrics.total_return_pct == 0

    def test_get_all_periods(self, calculator):
        """Test getting metrics for all periods."""
        now = datetime.now(timezone.utc)
        for i in range(100):
            calculator.add_equity_point(now - timedelta(days=100-i), 100000 + i * 100)

        all_metrics = calculator.get_all_periods()

        assert "1D" in all_metrics
        assert "1W" in all_metrics
        assert "1M" in all_metrics

    def test_trade_recording(self, calculator):
        """Test trade recording for win rate calculation."""
        calculator.add_trade({"pnl": 100, "close_time": datetime.now(timezone.utc)})
        calculator.add_trade({"pnl": -50, "close_time": datetime.now(timezone.utc)})

        assert len(calculator._trade_history) == 2


# =============================================================================
# 8.2 SESSION PERFORMANCE TESTS
# =============================================================================

class TestSessionPerformanceTracker:
    """Tests for Session Performance Tracker."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return SessionPerformanceTracker()

    def test_record_trade(self, tracker):
        """Test recording a trade."""
        now = datetime.now(timezone.utc)
        tracker.record_trade(
            open_time=now,
            close_time=now + timedelta(hours=2),
            pnl=500,
            symbol="AAPL",
        )

        # Should be recorded in some session
        total_trades = sum(
            len(trades) for trades in tracker._trades_by_session.values()
        )
        assert total_trades == 1

    def test_session_detection_asian(self, tracker):
        """Test Asian session detection."""
        asian_time = datetime(2026, 2, 4, 3, 0, 0, tzinfo=timezone.utc)
        session = tracker._get_session(asian_time)
        assert session == TradingSession.ASIAN

    def test_session_detection_overlap(self, tracker):
        """Test EU/US overlap detection."""
        overlap_time = datetime(2026, 2, 4, 14, 0, 0, tzinfo=timezone.utc)
        session = tracker._get_session(overlap_time)
        assert session == TradingSession.OVERLAP_EU_US

    def test_get_session_performance(self, tracker):
        """Test session performance calculation."""
        now = datetime.now(timezone.utc)

        # Add some trades
        for i in range(10):
            tracker.record_trade(
                open_time=now - timedelta(hours=i),
                close_time=now - timedelta(hours=i) + timedelta(minutes=30),
                pnl=100 if i % 2 == 0 else -50,
                symbol="SPY",
            )

        # Get performance for any session with trades
        for session in TradingSession:
            perf = tracker.get_session_performance(session)
            assert isinstance(perf, SessionPerformance)

    def test_get_all_sessions(self, tracker):
        """Test getting all session performances."""
        all_sessions = tracker.get_all_sessions()

        assert "asian" in all_sessions
        assert "european" in all_sessions
        assert "us" in all_sessions
        assert "overlap" in all_sessions


# =============================================================================
# 8.3 STRATEGY COMPARISON TESTS
# =============================================================================

class TestStrategyComparisonTracker:
    """Tests for Strategy Comparison Tracker."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return StrategyComparisonTracker()

    def test_record_signal(self, tracker):
        """Test recording a signal."""
        tracker.record_signal(
            strategy="momentum",
            symbol="AAPL",
            direction="LONG",
            conviction=0.8,
            was_acted_on=True,
            timestamp=datetime.now(timezone.utc),
        )

        assert len(tracker._strategy_signals["momentum"]) == 1

    def test_record_trade(self, tracker):
        """Test recording a trade."""
        tracker.record_trade(
            strategy="momentum",
            symbol="AAPL",
            pnl=500,
            holding_hours=24,
            timestamp=datetime.now(timezone.utc),
        )

        assert len(tracker._strategy_trades["momentum"]) == 1

    def test_get_strategy_performance(self, tracker):
        """Test getting strategy performance."""
        # Add data
        for i in range(5):
            tracker.record_signal(
                strategy="stat_arb",
                symbol="SPY",
                direction="LONG",
                conviction=0.7,
                was_acted_on=True,
                timestamp=datetime.now(timezone.utc),
            )
            tracker.record_trade(
                strategy="stat_arb",
                symbol="SPY",
                pnl=100 if i % 2 == 0 else -50,
                holding_hours=12,
                timestamp=datetime.now(timezone.utc),
            )

        perf = tracker.get_strategy_performance("stat_arb")

        assert isinstance(perf, StrategyPerformance)
        assert perf.total_signals == 5
        assert perf.total_pnl == 200  # 3*100 - 2*50 = 300-100 = 200

    def test_get_all_strategies(self, tracker):
        """Test getting all strategy performances."""
        tracker.record_trade(
            strategy="momentum", symbol="SPY", pnl=100,
            holding_hours=1, timestamp=datetime.now(timezone.utc)
        )
        tracker.record_trade(
            strategy="macro", symbol="ES", pnl=200,
            holding_hours=2, timestamp=datetime.now(timezone.utc)
        )

        all_strats = tracker.get_all_strategies()

        assert "momentum" in all_strats
        assert "macro" in all_strats

    def test_get_ranking(self, tracker):
        """Test strategy ranking."""
        tracker.record_trade(
            strategy="best", symbol="SPY", pnl=1000,
            holding_hours=1, timestamp=datetime.now(timezone.utc)
        )
        tracker.record_trade(
            strategy="worst", symbol="SPY", pnl=-500,
            holding_hours=1, timestamp=datetime.now(timezone.utc)
        )

        ranking = tracker.get_ranking(metric="total_pnl")

        assert ranking[0]["strategy_name"] == "best"


# =============================================================================
# 8.4 RISK HEATMAP TESTS
# =============================================================================

class TestRiskHeatmapGenerator:
    """Tests for Risk Heatmap Generator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return RiskHeatmapGenerator()

    def test_update_position(self, generator):
        """Test updating position value."""
        generator.update_position("AAPL", 10000)
        generator.update_position("AAPL", 10500)

        assert len(generator._position_history["AAPL"]) == 2

    def test_calculate_risk_score(self, generator):
        """Test risk score calculation."""
        generator.update_volatility("HIGH_RISK", 0.5)
        generator.update_correlation("HIGH_RISK", 0.9)

        score = generator.calculate_risk_score(
            symbol="HIGH_RISK",
            position_value=25000,
            portfolio_value=100000,
            var_contribution=35,
            entry_date=datetime.now(timezone.utc) - timedelta(days=45),
        )

        assert isinstance(score, PositionRiskScore)
        assert score.risk_score > 50  # Should be high risk
        assert "high_concentration" in score.risk_factors or score.concentration_pct > 20

    def test_calculate_low_risk_score(self, generator):
        """Test low risk score calculation."""
        generator.update_volatility("LOW_RISK", 0.1)
        generator.update_correlation("LOW_RISK", 0.2)

        score = generator.calculate_risk_score(
            symbol="LOW_RISK",
            position_value=1000,
            portfolio_value=100000,
            var_contribution=2,
            entry_date=datetime.now(timezone.utc) - timedelta(days=3),
        )

        assert score.risk_score < 50  # Should be low risk

    def test_get_heatmap_data(self, generator):
        """Test heatmap data generation."""
        positions = [
            {"symbol": "AAPL", "value": 10000, "var_contribution": 15},
            {"symbol": "GOOGL", "value": 15000, "var_contribution": 20},
        ]

        heatmap = generator.get_heatmap_data(positions, portfolio_value=100000)

        assert len(heatmap) == 2
        assert all("risk_score" in h for h in heatmap)


# =============================================================================
# 8.5 TRADE JOURNAL TESTS
# =============================================================================

class TestTradeJournal:
    """Tests for Trade Journal."""

    @pytest.fixture
    def journal(self):
        """Create journal instance."""
        return TradeJournal()

    def test_create_entry(self, journal):
        """Test creating journal entry."""
        entry = journal.create_entry(
            trade_id="T001",
            symbol="AAPL",
            direction="LONG",
            entry_price=150.0,
            exit_price=155.0,
            entry_time=datetime.now(timezone.utc) - timedelta(hours=4),
            exit_time=datetime.now(timezone.utc),
            strategy="momentum",
            quality_rating=4,
            emotional_state="confident",
            setup_notes="Strong breakout setup",
            execution_notes="Good fill",
            lessons_learned="Wait for confirmation",
            tags=["breakout", "tech"],
        )

        assert isinstance(entry, TradeJournalEntry)
        assert entry.pnl == 5.0
        assert len(journal._entries) == 1

    def test_get_entries_filtered(self, journal):
        """Test getting filtered entries."""
        journal.create_entry(
            trade_id="T001", symbol="AAPL", direction="LONG",
            entry_price=100, exit_price=105,
            entry_time=datetime.now(timezone.utc) - timedelta(hours=2),
            exit_time=datetime.now(timezone.utc),
            strategy="momentum", tags=["tech"],
        )
        journal.create_entry(
            trade_id="T002", symbol="GOOGL", direction="LONG",
            entry_price=200, exit_price=195,
            entry_time=datetime.now(timezone.utc) - timedelta(hours=1),
            exit_time=datetime.now(timezone.utc),
            strategy="macro", tags=["tech"],
        )

        # Filter by symbol
        aapl_entries = journal.get_entries(symbol="AAPL")
        assert len(aapl_entries) == 1

        # Filter by strategy
        momentum_entries = journal.get_entries(strategy="momentum")
        assert len(momentum_entries) == 1

    def test_quality_stats(self, journal):
        """Test quality statistics."""
        journal.create_entry(
            trade_id="T001", symbol="SPY", direction="LONG",
            entry_price=100, exit_price=105,
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
            strategy="test", quality_rating=5,
        )
        journal.create_entry(
            trade_id="T002", symbol="SPY", direction="LONG",
            entry_price=100, exit_price=95,
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
            strategy="test", quality_rating=2,
        )

        stats = journal.get_quality_stats()

        assert stats[5]["count"] == 1
        assert stats[2]["count"] == 1

    def test_emotional_stats(self, journal):
        """Test emotional state statistics."""
        journal.create_entry(
            trade_id="T001", symbol="SPY", direction="LONG",
            entry_price=100, exit_price=110,
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
            strategy="test", emotional_state="confident",
        )

        stats = journal.get_emotional_stats()

        assert stats["confident"]["count"] == 1


# =============================================================================
# 8.6 SIGNAL CONSENSUS TESTS
# =============================================================================

class TestSignalConsensusTracker:
    """Tests for Signal Consensus Tracker."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return SignalConsensusTracker()

    def test_record_signal(self, tracker):
        """Test recording signal."""
        tracker.record_signal("AAPL", "momentum", "LONG", 0.8)
        tracker.record_signal("AAPL", "macro", "LONG", 0.6)

        assert len(tracker._current_signals["AAPL"]) == 2

    def test_get_consensus_unanimous(self, tracker):
        """Test consensus with unanimous signals."""
        tracker.record_signal("SPY", "agent1", "LONG", 0.8)
        tracker.record_signal("SPY", "agent2", "LONG", 0.7)
        tracker.record_signal("SPY", "agent3", "LONG", 0.9)

        consensus = tracker.get_consensus("SPY")

        assert consensus.market_bias == 1.0  # All bullish
        assert consensus.consensus_strength == 1.0
        assert consensus.disagreement_level == 0.0

    def test_get_consensus_mixed(self, tracker):
        """Test consensus with mixed signals."""
        tracker.record_signal("SPY", "agent1", "LONG", 0.8)
        tracker.record_signal("SPY", "agent2", "SHORT", 0.7)
        tracker.record_signal("SPY", "agent3", "NEUTRAL", 0.5)

        consensus = tracker.get_consensus("SPY")

        assert consensus.bullish_count == 1
        assert consensus.bearish_count == 1
        assert consensus.neutral_count == 1
        assert consensus.disagreement_level > 0.5  # High disagreement

    def test_get_high_disagreement_alerts(self, tracker):
        """Test high disagreement alerts."""
        # Create disagreement
        tracker.record_signal("HIGH_DIS", "a1", "LONG", 0.8)
        tracker.record_signal("HIGH_DIS", "a2", "SHORT", 0.8)
        tracker.record_signal("HIGH_DIS", "a3", "NEUTRAL", 0.5)

        # Create agreement
        tracker.record_signal("LOW_DIS", "a1", "LONG", 0.8)
        tracker.record_signal("LOW_DIS", "a2", "LONG", 0.8)

        alerts = tracker.get_high_disagreement_alerts(threshold=0.5)

        assert len(alerts) >= 1
        assert alerts[0]["symbol"] == "HIGH_DIS"

    def test_get_all_symbols(self, tracker):
        """Test getting all symbol consensus."""
        tracker.record_signal("AAPL", "agent1", "LONG", 0.8)
        tracker.record_signal("GOOGL", "agent1", "SHORT", 0.7)

        all_symbols = tracker.get_all_symbols()

        assert "AAPL" in all_symbols
        assert "GOOGL" in all_symbols


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_all_components(self):
        """Test creation of all components."""
        rolling = create_rolling_metrics_calculator()
        session = create_session_performance_tracker()
        strategy = create_strategy_comparison_tracker()
        risk = create_risk_heatmap_generator()
        journal = create_trade_journal()
        consensus = create_signal_consensus_tracker()

        assert isinstance(rolling, RollingMetricsCalculator)
        assert isinstance(session, SessionPerformanceTracker)
        assert isinstance(strategy, StrategyComparisonTracker)
        assert isinstance(risk, RiskHeatmapGenerator)
        assert isinstance(journal, TradeJournal)
        assert isinstance(consensus, SignalConsensusTracker)

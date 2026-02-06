"""
CIO (Chief Investment Officer) Agent
====================================

THE SINGLE DECISION-MAKING AUTHORITY.

This agent is the ONLY one authorized to make trading decisions.
It aggregates signals from all strategy agents and decides whether to trade.

Per the constitution:
- One and only one decision authority
- Decisions must include rationale and data sources
- All decisions are logged for compliance

Enhanced features:
- Kelly criterion position sizing
- Dynamic signal weights (regime-dependent, performance-weighted)
- Correlation-adjusted sizing
- Performance attribution integration
"""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, time as dt_time
from enum import Enum
from typing import TYPE_CHECKING, Any

from core.agent_base import DecisionAgent, AgentConfig
from core.events import (
    Event,
    EventType,
    SignalEvent,
    DecisionEvent,
    ValidatedDecisionEvent,
    MarketDataEvent,
    SignalDirection,
    OrderSide,
    OrderType,
    DecisionAction,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger
    from core.position_sizing import PositionSizer, StrategyStats
    from core.attribution import PerformanceAttribution
    from core.correlation_manager import CorrelationManager, CorrelationRegime
    from core.risk_budget import RiskBudgetManager

# Import CONTRACT_SPECS for futures multiplier (FIX-01: position sizing)
from core.contract_specs import CONTRACT_SPECS

# Import Signal Quality Scorer
try:
    from core.signal_quality import SignalQualityScorer, create_signal_quality_scorer
    HAS_SIGNAL_QUALITY = True
except ImportError:
    HAS_SIGNAL_QUALITY = False


logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification for weight adjustment."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    VOLATILE = "volatile"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    NEUTRAL = "neutral"


class DecisionMode(Enum):
    """
    CIO decision-making mode based on market/system conditions.

    NORMAL: Full strategy suite, normal position sizing
    DEFENSIVE: Reduced position sizes, conservative strategies only
    EMERGENCY: Minimal strategies, severely capped position sizes, exit-focused
    HUMAN_CIO: Human-in-the-loop mode - all decisions require human approval
    """
    NORMAL = "normal"
    DEFENSIVE = "defensive"
    EMERGENCY = "emergency"
    HUMAN_CIO = "human_cio"  # Requires human approval for all decisions


@dataclass
class PendingHumanDecision:
    """A decision awaiting human approval in HUMAN_CIO mode."""
    decision_id: str
    symbol: str
    direction: SignalDirection
    quantity: int
    conviction: float
    rationale: str
    signals: dict  # Original signals that led to this decision
    aggregation: Any  # SignalAggregation
    created_at: datetime
    expires_at: datetime  # Auto-reject after this time
    status: str = "pending"  # pending, approved, rejected, expired


# Core strategies allowed in EMERGENCY mode (most battle-tested)
EMERGENCY_MODE_CORE_STRATEGIES = {
    "MacroAgent",
    "MomentumAgent",
    "StatArbAgent",
}

# Position size caps by mode
POSITION_SIZE_CAPS = {
    DecisionMode.NORMAL: 1.0,       # 100% of calculated size
    DecisionMode.DEFENSIVE: 0.5,    # 50% of calculated size
    DecisionMode.EMERGENCY: 0.25,   # 25% of calculated size
    DecisionMode.HUMAN_CIO: 1.0,    # Human decides size, no auto cap
}


@dataclass
class SignalAggregation:
    """Aggregated signals for decision making."""
    symbol: str
    signals: dict[str, SignalEvent]  # agent_name -> signal
    weighted_strength: float = 0.0
    weighted_confidence: float = 0.0
    consensus_direction: SignalDirection = SignalDirection.FLAT
    timestamp: datetime = None
    regime_adjusted: bool = False
    correlation_adjusted: bool = False
    effective_signal_count: float = 0.0  # Effective N after correlation adjustment


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy used in dynamic weighting."""
    strategy: str
    rolling_sharpe: float = 0.0
    win_rate: float = 0.5
    recent_pnl: float = 0.0
    signal_accuracy: float = 0.5
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Kelly criterion inputs - track actual win/loss magnitudes
    avg_win: float = 0.0  # Average profit on winning trades
    avg_loss: float = 0.0  # Average loss on losing trades (positive value)
    total_trades: int = 0  # Number of trades for statistical significance


@dataclass
class DecisionRecord:
    """Record of a CIO decision for accuracy tracking (P2)."""
    decision_id: str
    symbol: str
    direction: SignalDirection
    quantity: int
    conviction_score: float
    timestamp: datetime
    contributing_strategies: list[str]
    regime_at_decision: str
    # Outcome tracking
    outcome_pnl: float | None = None
    outcome_direction_correct: bool | None = None
    outcome_recorded: bool = False


@dataclass
class TrackedPosition:
    """
    CIO-tracked position with performance metrics for autonomous management.

    The CIO monitors all positions and makes autonomous decisions to:
    - Close losing positions that exceed loss threshold
    - Take profits on winning positions
    - Reduce position size when conviction drops
    - Override individual stop-losses in special situations
    - Move stop to breakeven at 1R (Phase 12)
    - Trail stop behind price at 1.5R+ (Phase 12)
    """
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    is_long: bool
    current_price: float = 0.0
    highest_price: float = 0.0  # For tracking peak (trailing stop logic)
    lowest_price: float = float("inf")  # For short positions
    original_conviction: float = 0.0  # Conviction when position was opened
    current_conviction: float = 0.0  # Current conviction from latest signals
    last_signal_time: datetime | None = None  # When we last got a signal for this symbol
    contributing_strategies: list[str] = field(default_factory=list)
    stop_loss_level: float | None = None  # Individual stop-loss price
    take_profit_level: float | None = None  # Take profit target
    stop_loss_overridden: bool = False  # True if CIO has overridden individual stop
    # R-Multiple Tracking (Phase 3 Enhancement)
    initial_risk: float = 0.0  # Distance from entry to stop-loss (1R unit)
    # Phase 12: Active Position Protection
    stop_moved_to_breakeven: bool = False  # True once stop moved to entry price
    trailing_stop_active: bool = False  # True once trailing stop engages
    trailing_stop_level: float | None = None  # Current trailing stop price
    peak_r_multiple: float = 0.0  # Highest R-multiple reached
    partial_exits_taken: int = 0  # Number of partial exits (0, 1, 2, 3)
    strategy_type: str = ""  # For time-based exit rules (intraday, swing, pairs)
    contract_multiplier: float = 1.0  # FIX-09: Futures multiplier for PnL calculation

    @property
    def pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.entry_price <= 0 or self.current_price <= 0:
            return 0.0
        if self.is_long:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100

    @property
    def pnl_dollar(self) -> float:
        """Calculate unrealized P&L in dollars (FIX-09: includes futures multiplier)."""
        if self.is_long:
            return (self.current_price - self.entry_price) * self.quantity * self.contract_multiplier
        else:
            return (self.entry_price - self.current_price) * self.quantity * self.contract_multiplier

    @property
    def holding_hours(self) -> float:
        """Calculate how long position has been held."""
        delta = datetime.now(timezone.utc) - self.entry_time
        return delta.total_seconds() / 3600

    @property
    def drawdown_from_peak_pct(self) -> float:
        """Calculate drawdown from highest price seen (for longs)."""
        if self.highest_price <= 0:
            return 0.0
        return ((self.highest_price - self.current_price) / self.highest_price) * 100

    @property
    def rally_from_trough_pct(self) -> float:
        """Calculate rally from lowest price seen (for shorts)."""
        if self.lowest_price == float("inf") or self.lowest_price <= 0:
            return 0.0
        return ((self.current_price - self.lowest_price) / self.lowest_price) * 100

    @property
    def r_multiple(self) -> float:
        """
        Calculate R-Multiple (risk-adjusted return).

        R-Multiple = PnL / Initial Risk
        - 1R = breakeven on initial risk
        - 2R = made 2x the initial risk
        - -1R = lost exactly the initial risk (stopped out)

        If no stop-loss was set, returns 0.0.
        """
        if self.initial_risk <= 0:
            return 0.0

        # Calculate P&L in price terms
        if self.is_long:
            pnl = self.current_price - self.entry_price
        else:
            pnl = self.entry_price - self.current_price

        return pnl / self.initial_risk

    def update_price(self, price: float) -> None:
        """Update current price and track extremes."""
        self.current_price = price
        if self.is_long:
            self.highest_price = max(self.highest_price, price)
        else:
            self.lowest_price = min(self.lowest_price, price)
        # Track peak R-multiple
        current_r = self.r_multiple
        if current_r > self.peak_r_multiple:
            self.peak_r_multiple = current_r


@dataclass
class PositionManagementConfig:
    """Configuration for autonomous position management."""
    # Loss thresholds - when to close losing positions
    max_loss_pct: float = 5.0  # Close position at -5% loss
    extended_loss_pct: float = 8.0  # Definitely close at -8%
    loss_time_threshold_hours: float = 48.0  # Close if losing for 48+ hours

    # Profit taking thresholds (Phase 12: realistic for intraday futures)
    profit_target_pct: float = 4.0  # Start taking profit at +4% (was 15%)
    trailing_profit_pct: float = 1.5  # Take profit if drops 1.5% from peak (was 3%)
    partial_profit_pct: float = 33.0  # Partial exit: sell 33% at each level

    # Conviction-based position reduction
    conviction_drop_threshold: float = 0.3  # Reduce if conviction drops > 30%
    min_conviction_to_hold: float = 0.4  # Close if conviction below this

    # Time-based rules (Phase 12: per-strategy-type holding limits)
    max_holding_days: float = 2.0  # Default max holding 2 days (was 30)
    stale_signal_hours: float = 12.0  # Signal is stale after 12h (was 24h)
    # Per-strategy-type max holding hours
    max_holding_hours_intraday: float = 4.0  # Session, TTM, MeanReversion
    max_holding_hours_swing: float = 48.0  # Momentum, MACD-v, Macro
    max_holding_hours_pairs: float = 120.0  # StatArb, IndexSpread (5 days)

    # Market regime adjustments
    volatile_regime_loss_pct: float = 3.0  # Tighter stop in volatile markets
    trending_regime_profit_mult: float = 1.5  # Let profits run in trends

    # Stop-loss override rules
    allow_stop_override_in_trending: bool = True  # Can override stops in strong trends
    stop_override_min_conviction: float = 0.8  # Need 80%+ conviction to override

    # Phase 12: Breakeven & Trailing Stop Configuration
    breakeven_r_trigger: float = 1.0  # Move stop to breakeven at 1R profit
    breakeven_buffer_pct: float = 0.1  # Add 0.1% above entry for spread protection
    trailing_activation_r: float = 1.5  # Activate trailing stop at 1.5R
    trailing_distance_r: float = 0.5  # Trail 0.5R behind peak
    # Graduated partial profit levels
    partial_profit_1_r: float = 1.5  # First partial exit at 1.5R (33%)
    partial_profit_2_r: float = 2.5  # Second partial exit at 2.5R (33%)
    partial_profit_3_r: float = 3.5  # Final exit at 3.5R or trailing stop

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for status reporting."""
        return {
            "max_loss_pct": self.max_loss_pct,
            "extended_loss_pct": self.extended_loss_pct,
            "loss_time_threshold_hours": self.loss_time_threshold_hours,
            "profit_target_pct": self.profit_target_pct,
            "trailing_profit_pct": self.trailing_profit_pct,
            "partial_profit_pct": self.partial_profit_pct,
            "conviction_drop_threshold": self.conviction_drop_threshold,
            "min_conviction_to_hold": self.min_conviction_to_hold,
            "max_holding_days": self.max_holding_days,
            "stale_signal_hours": self.stale_signal_hours,
            "volatile_regime_loss_pct": self.volatile_regime_loss_pct,
            "trending_regime_profit_mult": self.trending_regime_profit_mult,
            "allow_stop_override_in_trending": self.allow_stop_override_in_trending,
            "stop_override_min_conviction": self.stop_override_min_conviction,
            "breakeven_r_trigger": self.breakeven_r_trigger,
            "trailing_activation_r": self.trailing_activation_r,
            "trailing_distance_r": self.trailing_distance_r,
            "max_holding_hours_intraday": self.max_holding_hours_intraday,
            "max_holding_hours_swing": self.max_holding_hours_swing,
            "max_holding_hours_pairs": self.max_holding_hours_pairs,
        }


class CIOAgent(DecisionAgent):
    """
    Chief Investment Officer Agent.

    THE ONLY DECISION-MAKING AUTHORITY IN THE SYSTEM.

    Responsibilities:
    1. Wait for signal barrier synchronization (fan-in)
    2. Aggregate signals from all strategy agents
    3. Apply portfolio-level constraints
    4. Make final trading decisions
    5. Log decisions with full rationale

    This agent does NOT:
    - Generate signals (that's strategy agents' job)
    - Execute orders (that's execution agent's job)
    - Validate risk/compliance (that's risk agent's job)
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Base signal weights by strategy (ALL agents must be listed here)
        self._base_weights = {
            "MacroAgent": config.parameters.get("signal_weight_macro", 0.10),
            "StatArbAgent": config.parameters.get("signal_weight_stat_arb", 0.12),
            "MomentumAgent": config.parameters.get("signal_weight_momentum", 0.12),
            "MarketMakingAgent": config.parameters.get("signal_weight_market_making", 0.08),
            "MACDvAgent": config.parameters.get("signal_weight_macdv", 0.10),
            "SessionAgent": config.parameters.get("signal_weight_session", 0.10),
            "IndexSpreadAgent": config.parameters.get("signal_weight_index_spread", 0.10),
            "TTMSqueezeAgent": config.parameters.get("signal_weight_ttm_squeeze", 0.08),
            "EventDrivenAgent": config.parameters.get("signal_weight_event_driven", 0.08),
            "MeanReversionAgent": config.parameters.get("signal_weight_mean_reversion", 0.10),
            "SentimentAgent": config.parameters.get("signal_weight_sentiment", 0.10),
            "ChartAnalysisAgent": config.parameters.get("signal_weight_chart_analysis", 0.10),
            "ForecastingAgent": config.parameters.get("signal_weight_forecasting", 0.10),
        }

        # Current effective weights (may be adjusted dynamically)
        self._weights = dict(self._base_weights)

        # Sector concentration limits (PM-01)
        self._max_sector_concentration = config.parameters.get("max_sector_concentration", 0.25)  # 25% max per sector
        self._symbol_to_sector: dict[str, str] = {}  # symbol -> sector mapping
        self._sector_positions: dict[str, float] = {}  # sector -> current exposure as fraction of portfolio

        # Portfolio drawdown tracking (PM-06)
        self._portfolio_drawdown = 0.0  # Current drawdown from peak (0.0 to 1.0)
        self._portfolio_peak = config.parameters.get("portfolio_value", 1_000_000.0)
        self._drawdown_kelly_threshold = config.parameters.get("drawdown_kelly_threshold", 0.05)  # 5% drawdown threshold
        self._drawdown_kelly_floor = config.parameters.get("drawdown_kelly_floor", 0.5)  # Minimum kelly multiplier at max drawdown

        # Stress correlation cache (PM-12)
        self._stress_correlation_cache: dict[tuple[str, str], float] = {}
        self._in_stress_mode = False

        # Decision thresholds
        self._min_conviction = config.parameters.get("min_conviction_threshold", 0.6)
        self._max_concurrent = config.parameters.get("max_concurrent_decisions", 5)

        # Dynamic weight settings
        self._use_dynamic_weights = config.parameters.get("use_dynamic_weights", True)
        self._performance_weight_factor = config.parameters.get("performance_weight_factor", 0.3)
        self._regime_weight_factor = config.parameters.get("regime_weight_factor", 0.2)

        # Position sizing settings - IMPROVED for better money management
        self._use_kelly_sizing = config.parameters.get("use_kelly_sizing", True)
        # IMPROVED: Use quarter-Kelly (0.25x) for safer position sizing
        self._kelly_fraction = config.parameters.get("kelly_fraction", 0.25)
        self._base_position_size = config.parameters.get("base_position_size", 100)
        # IMPROVED: Reduce max position size from 1000 to 500 shares
        self._max_position_size = config.parameters.get("max_position_size", 500)
        self._portfolio_value = config.parameters.get("portfolio_value", 1_000_000.0)
        # NEW: Maximum position as % of portfolio (default 2.5%)
        self._max_position_pct = config.parameters.get("max_position_pct", 2.5)
        # NEW: Maximum total portfolio exposure (default 50%)
        self._max_total_exposure_pct = config.parameters.get("max_total_exposure_pct", 50.0)
        # NEW: Maximum raw Kelly before applying fraction (default 15%)
        self._max_kelly_raw = config.parameters.get("max_kelly_raw", 0.15)
        # NEW: Per-position daily loss limit as % of portfolio (default 1%)
        self._max_position_daily_loss_pct = config.parameters.get("max_position_daily_loss_pct", 1.0)

        # Intraday protection: max open positions and per-symbol cooldown
        self._max_open_positions = config.parameters.get("max_open_positions", 6)
        self._last_decision_time_per_symbol: dict[str, datetime] = {}
        self._decision_cooldown_seconds = config.parameters.get("decision_cooldown_seconds", 120.0)

        # State
        self._pending_aggregations: dict[str, SignalAggregation] = {}
        self._active_decisions: dict[str, datetime] = {}  # decision_id -> created_time
        self._decision_timeout_seconds = 60.0  # Clean up decisions older than this

        # Current market regime
        self._current_regime = MarketRegime.NEUTRAL

        # Strategy performance tracking
        self._strategy_performance: dict[str, StrategyPerformance] = {}

        # External components (lazy initialization)
        self._position_sizer = None
        self._attribution = None
        self._correlation_manager = None
        self._risk_budget_manager = None  # Cross-strategy risk budget (#P3)

        # Price cache for position sizing (symbol -> latest price)
        self._price_cache: dict[str, float] = {}

        # Broker reference for leverage-aware deleveraging
        self._broker = None
        self._max_leverage = config.parameters.get("max_leverage", 2.0)
        self._deleveraging_active = False
        self._last_deleverage_check = datetime.now(timezone.utc)

        # Market hours filtering - prevent decisions for US stocks/ETFs outside market hours
        self._us_equity_symbols: set[str] = set(
            s.upper() for s in config.parameters.get("_us_equity_symbols", [])
        )
        self._us_etf_symbols: set[str] = set(
            s.upper() for s in config.parameters.get("_us_etf_symbols", [])
        )
        self._forex_symbols: set[str] = set(
            s.upper() for s in config.parameters.get("_forex_symbols", [])
        )
        self._us_stock_symbols = self._us_equity_symbols | self._us_etf_symbols
        self._filter_market_hours = config.parameters.get("filter_market_hours", True)

        # No-trade symbols: ETFs and individual stocks are used for analysis only
        # (MacroAgent regime detection, correlation). We only trade futures.
        self._no_trade_symbols: set[str] = set(self._us_etf_symbols) | set(self._forex_symbols) | set(self._us_equity_symbols)

        # Signal history for correlation tracking (#Q5)
        self._signal_history: dict[str, list[tuple[datetime, float]]] = {}  # agent -> [(time, direction_val)]
        self._signal_correlation_matrix: dict[tuple[str, str], float] = {}  # (agent1, agent2) -> correlation
        self._max_signal_history = config.parameters.get("max_signal_history", 100)
        self._correlation_lookback = config.parameters.get("signal_correlation_lookback", 50)
        self._use_correlation_adjustment = config.parameters.get("use_signal_correlation_adjustment", True)

        # Enhanced monthly correlation tracking (Phase D improvement)
        self._monthly_correlation_lookback = config.parameters.get("monthly_correlation_lookback", 720)  # 30 days hourly
        self._high_correlation_threshold = config.parameters.get("high_correlation_threshold", 0.95)
        self._high_correlation_weight_factor = config.parameters.get("high_correlation_weight_factor", 0.5)
        self._monthly_correlations: dict[tuple[str, str], float] = {}  # (agent1, agent2) -> monthly correlation
        self._last_monthly_correlation_update: datetime | None = None
        self._monthly_correlation_update_interval_hours = config.parameters.get(
            "monthly_correlation_update_interval_hours", 1.0
        )
        self._correlation_halved_count = 0  # Track how often weights are halved due to high correlation

        # P2: Historical decision accuracy tracking (bounded to prevent memory leak)
        self._max_decision_history = config.parameters.get("max_decision_history", 500)
        self._decision_history: deque[DecisionRecord] = deque(maxlen=self._max_decision_history)
        self._decision_accuracy_by_strategy: dict[str, dict[str, float]] = {}  # strategy -> {accuracy, count}
        self._decision_accuracy_by_regime: dict[str, dict[str, float]] = {}  # regime -> {accuracy, count}
        self._overall_decision_accuracy: float = 0.0
        self._total_decisions_tracked: int = 0

        # P2: Signal confidence weighting configuration
        self._min_signal_confidence = config.parameters.get("min_signal_confidence", 0.3)  # Filter low-confidence signals
        self._confidence_weight_power = config.parameters.get("confidence_weight_power", 1.5)  # Apply non-linear weighting

        # P2: Regime-based allocation adjustments
        self._regime_allocation_multipliers = {
            MarketRegime.RISK_ON: config.parameters.get("regime_alloc_risk_on", 1.2),  # Increase allocation
            MarketRegime.RISK_OFF: config.parameters.get("regime_alloc_risk_off", 0.7),  # Decrease allocation
            MarketRegime.VOLATILE: config.parameters.get("regime_alloc_volatile", 0.5),  # Significantly decrease
            MarketRegime.TRENDING: config.parameters.get("regime_alloc_trending", 1.1),  # Slight increase
            MarketRegime.MEAN_REVERTING: config.parameters.get("regime_alloc_mean_rev", 1.0),  # Neutral
            MarketRegime.NEUTRAL: config.parameters.get("regime_alloc_neutral", 1.0),  # Baseline
        }

        # Regime-specific weight adjustments
        self._regime_weights = {
            MarketRegime.RISK_ON: {
                "MomentumAgent": 1.3,
                "StatArbAgent": 0.9,
                "MacroAgent": 0.8,
            },
            MarketRegime.RISK_OFF: {
                "MacroAgent": 1.5,
                "MomentumAgent": 0.7,
                "MarketMakingAgent": 0.8,
            },
            MarketRegime.VOLATILE: {
                "MACDvAgent": 1.4,
                "MarketMakingAgent": 0.7,
                "MomentumAgent": 0.8,
            },
            MarketRegime.TRENDING: {
                "MomentumAgent": 1.4,
                "StatArbAgent": 0.7,
            },
            MarketRegime.MEAN_REVERTING: {
                "StatArbAgent": 1.4,
                "MomentumAgent": 0.6,
            },
        }

        # =====================================================================
        # AUTONOMOUS POSITION MANAGEMENT
        # =====================================================================

        # Tracked positions for CIO-level management
        self._tracked_positions: dict[str, TrackedPosition] = {}

        # Position management configuration
        pm_config = config.parameters.get("position_management", {})
        self._position_management_config = PositionManagementConfig(
            max_loss_pct=pm_config.get("max_loss_pct", 5.0),
            extended_loss_pct=pm_config.get("extended_loss_pct", 8.0),
            loss_time_threshold_hours=pm_config.get("loss_time_threshold_hours", 48.0),
            profit_target_pct=pm_config.get("profit_target_pct", 4.0),
            trailing_profit_pct=pm_config.get("trailing_profit_pct", 1.5),
            partial_profit_pct=pm_config.get("partial_profit_pct", 33.0),
            conviction_drop_threshold=pm_config.get("conviction_drop_threshold", 0.3),
            min_conviction_to_hold=pm_config.get("min_conviction_to_hold", 0.4),
            max_holding_days=pm_config.get("max_holding_days", 2.0),
            stale_signal_hours=pm_config.get("stale_signal_hours", 12.0),
            volatile_regime_loss_pct=pm_config.get("volatile_regime_loss_pct", 3.0),
            trending_regime_profit_mult=pm_config.get("trending_regime_profit_mult", 1.5),
            allow_stop_override_in_trending=pm_config.get("allow_stop_override_in_trending", True),
            stop_override_min_conviction=pm_config.get("stop_override_min_conviction", 0.8),
            # Phase 12: Active Position Protection
            breakeven_r_trigger=pm_config.get("breakeven_r_trigger", 1.0),
            breakeven_buffer_pct=pm_config.get("breakeven_buffer_pct", 0.1),
            trailing_activation_r=pm_config.get("trailing_activation_r", 1.5),
            trailing_distance_r=pm_config.get("trailing_distance_r", 0.5),
            partial_profit_1_r=pm_config.get("partial_profit_1_r", 1.5),
            partial_profit_2_r=pm_config.get("partial_profit_2_r", 2.5),
            partial_profit_3_r=pm_config.get("partial_profit_3_r", 3.5),
            max_holding_hours_intraday=pm_config.get("max_holding_hours_intraday", 4.0),
            max_holding_hours_swing=pm_config.get("max_holding_hours_swing", 48.0),
            max_holding_hours_pairs=pm_config.get("max_holding_hours_pairs", 120.0),
        )

        # Position management state
        self._position_management_enabled = config.parameters.get("position_management_enabled", True)
        self._last_position_review = datetime.now(timezone.utc)
        self._position_review_interval_seconds = config.parameters.get("position_review_interval_seconds", 60.0)

        # Track position management decisions for analytics
        self._position_management_stats = {
            "losers_closed": 0,
            "profits_taken": 0,
            "positions_reduced": 0,
            "stop_overrides": 0,
            "time_exits": 0,
            "conviction_exits": 0,
            # Phase 3: R-Multiple tracking
            "r_multiple_exits_2r": 0,
            "r_multiple_exits_3r": 0,
            "total_r_multiple_closed": 0.0,  # Sum of R-multiples at exit
            "closed_trade_count": 0,
            # Phase 12: Active Position Protection
            "breakeven_stops_set": 0,
            "trailing_stops_triggered": 0,
            "partial_profits_1r5": 0,
            "partial_profits_2r5": 0,
            "full_profit_exits": 0,
        }

        # =====================================================================
        # DECISION MODE (Emergency/Defensive mode support)
        # =====================================================================
        self._decision_mode = DecisionMode.NORMAL
        self._decision_mode_override: DecisionMode | None = None  # Manual override
        self._decision_mode_auto_escalation = config.parameters.get("decision_mode_auto_escalation", True)
        self._defensive_mode_drawdown_threshold = config.parameters.get("defensive_mode_drawdown_threshold", 0.03)  # 3%
        self._emergency_mode_drawdown_threshold = config.parameters.get("emergency_mode_drawdown_threshold", 0.05)  # 5%
        self._mode_change_history: deque[tuple[datetime, DecisionMode, str]] = deque(maxlen=50)

        # =====================================================================
        # HUMAN-IN-THE-LOOP MODE (HUMAN_CIO)
        # =====================================================================
        hitl_config = config.parameters.get("human_in_the_loop", {})
        self._human_decision_timeout_seconds = hitl_config.get("timeout_seconds", 300.0)  # 5 min default
        self._max_pending_human_decisions = hitl_config.get("max_pending_decisions", 100)  # Issue 4.2: Bounded queue
        self._pending_human_decisions: dict[str, PendingHumanDecision] = {}  # decision_id -> pending
        self._human_decision_lock = threading.RLock()  # Issue 4.1: Thread-safe approval/rejection
        self._human_decision_history: deque[dict] = deque(maxlen=500)  # Audit trail
        self._human_cio_enabled = False  # Set via set_human_cio_mode()
        self._human_cio_reason: str = ""  # Why human-in-the-loop is active
        self._human_decision_callback: Any = None  # Optional callback for pending decisions

        # =====================================================================
        # SIGNAL QUALITY SCORING (Phase 2 Enhancement)
        # =====================================================================
        signal_quality_config = config.parameters.get("signal_quality", {})
        self._signal_quality_enabled = signal_quality_config.get("enabled", True)
        self._signal_quality_scorer: SignalQualityScorer | None = None
        if HAS_SIGNAL_QUALITY and self._signal_quality_enabled:
            self._signal_quality_scorer = create_signal_quality_scorer({
                "min_total_score": signal_quality_config.get("min_total_score", 50.0),
                "min_volume_score": signal_quality_config.get("min_volume_score", 5.0),
                "min_trend_score": signal_quality_config.get("min_trend_score", 5.0),
            })
            logger.info("Signal Quality Scorer initialized")

        # Market data cache for quality scoring (bounded deques per symbol)
        self._market_data_cache: dict[str, dict[str, deque]] = {}
        self._market_data_maxlen = 100  # Keep last 100 data points

        # Decision analysis CSV file
        self._decision_csv_path = "logs/decision_analysis.csv"
        self._decision_csv_initialized = False
        self._init_decision_csv()

    def _init_decision_csv(self) -> None:
        """Initialize the decision analysis CSV file with headers."""
        os.makedirs(os.path.dirname(self._decision_csv_path), exist_ok=True)

        # Define all possible agent columns
        self._csv_agents = [
            "MacroAgent", "MomentumAgent", "StatArbAgent", "MarketMakingAgent",
            "SessionAgent", "IndexSpreadAgent", "TTMSqueezeAgent", "EventDrivenAgent",
            "MeanReversionAgent", "MACDvAgent", "SentimentAgent", "ChartAnalysisAgent",
            "ForecastingAgent"
        ]

        # Build headers: timestamp, symbol, then for each agent: direction, confidence, weight
        headers = ["timestamp", "symbol"]
        for agent in self._csv_agents:
            headers.extend([f"{agent}_dir", f"{agent}_conf", f"{agent}_weight"])
        headers.extend([
            "long_votes", "short_votes", "total_weight", "threshold_40pct",
            "consensus_direction", "weighted_confidence", "final_decision", "quantity"
        ])

        # Write headers if file doesn't exist or is empty
        write_headers = True
        if os.path.exists(self._decision_csv_path):
            try:
                with open(self._decision_csv_path, 'r') as f:
                    first_line = f.readline()
                    if first_line.startswith("timestamp"):
                        write_headers = False
            except Exception:
                pass

        if write_headers:
            with open(self._decision_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            logger.info(f"Decision analysis CSV initialized: {self._decision_csv_path}")

        self._decision_csv_initialized = True

    def _log_decision_to_csv(
        self,
        agg: "SignalAggregation",
        filtered_signals: dict[str, SignalEvent],
        adjusted_weights: dict[str, float],
        long_votes: float,
        short_votes: float,
        total_weight: float,
        final_decision: str = "NONE",
        quantity: int = 0
    ) -> None:
        """Log decision details to CSV for analysis."""
        if not self._decision_csv_initialized:
            return

        try:
            row = [
                datetime.now(timezone.utc).isoformat(),
                agg.symbol
            ]

            # Add each agent's data (direction, confidence, weight)
            for agent in self._csv_agents:
                if agent in filtered_signals:
                    signal = filtered_signals[agent]
                    row.extend([
                        signal.direction.value,
                        f"{signal.confidence:.3f}",
                        f"{adjusted_weights.get(agent, 0):.3f}"
                    ])
                else:
                    row.extend(["", "", ""])  # Empty if agent didn't contribute

            # Add aggregation results
            row.extend([
                f"{long_votes:.3f}",
                f"{short_votes:.3f}",
                f"{total_weight:.3f}",
                f"{total_weight * 0.55:.3f}",
                agg.consensus_direction.value,
                f"{agg.weighted_confidence:.3f}",
                final_decision,
                str(quantity)
            ])

            # Append to CSV
            with open(self._decision_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

        except Exception as e:
            logger.warning(f"Failed to write decision to CSV: {e}")

    async def initialize(self) -> None:
        """Initialize CIO agent."""
        logger.info(f"CIOAgent initializing with weights: {self._weights}")
        logger.info(f"Min conviction threshold: {self._min_conviction}")

    def get_subscribed_events(self) -> list[EventType]:
        """CIO subscribes to validated decisions and market data (for quality scoring cache)."""
        return [EventType.VALIDATED_DECISION, EventType.MARKET_DATA]

    async def handle_event(self, event: Event) -> None:
        """Handle subscribed events."""
        if event.event_type == EventType.MARKET_DATA and isinstance(event, MarketDataEvent):
            # Update market data cache for signal quality scoring
            price = event.last or event.bid or event.ask or 0.0
            volume = event.volume or 0.0
            if price > 0:
                self.update_market_data_cache(event.symbol, price, volume)

    async def start(self) -> None:
        """Start CIO agent with barrier monitoring loop."""
        await super().start()
        # Start barrier monitoring as background task
        self._barrier_task = asyncio.create_task(self._barrier_monitoring_loop())
        logger.info("CIO barrier monitoring started")

    async def stop(self) -> None:
        """Stop CIO agent and cancel barrier monitoring."""
        if hasattr(self, '_barrier_task') and self._barrier_task:
            self._barrier_task.cancel()
            try:
                await self._barrier_task
            except asyncio.CancelledError:
                pass
        await super().stop()

    async def _barrier_monitoring_loop(self) -> None:
        """
        Monitor signal barrier for fan-in synchronization.

        This is the correct fan-in implementation per CLAUDE.md:
        - Wait for ALL signal agents to report (or timeout)
        - CHECK BARRIER VALIDITY before making decisions (Architecture Invariant #2)
        - Process signals together after synchronization
        - Avoid making decisions on partial/invalid signals
        """
        while self._running:
            try:
                # Wait for barrier to complete (returns BarrierResult with validity)
                result = await self._event_bus.wait_for_signals()

                if not result.signals:
                    # No barrier active, wait to avoid busy loop
                    await asyncio.sleep(0.5)
                    continue

                # ============================================================
                # Phase 12: CHECK BARRIER VALIDITY (Architecture Invariant #2)
                # ============================================================
                if not result.is_valid:
                    # CRITICAL agents missing - DO NOT make new trading decisions
                    logger.error(
                        f"CIO: BARRIER INVALID - {len(result.missing_critical)} CRITICAL "
                        f"agent(s) missing: {result.missing_critical}. "
                        f"Skipping new signal processing. "
                        f"({result.total_received}/{result.total_expected} agents responded)"
                    )
                    # Still manage existing positions (exits, breakeven, trailing)
                    # but do NOT open new positions based on incomplete signals
                    if self._position_management_enabled:
                        await self._review_and_manage_positions(result.signals)
                    self._position_management_stats.setdefault("barrier_failures", 0)
                    self._position_management_stats["barrier_failures"] += 1
                    continue

                if not result.quorum_met:
                    logger.warning(
                        f"CIO: Barrier quorum NOT met "
                        f"({result.total_received}/{result.total_expected}). "
                        f"Missing: {result.missing_agents}. "
                        f"Proceeding with reduced confidence."
                    )

                logger.info(
                    f"CIO: Barrier complete with {len(result.signals)} signals "
                    f"(valid={result.is_valid}, quorum={result.quorum_met})"
                )
                await self._process_barrier_signals(result.signals)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # CIO decision loop must stay alive - preserve trace for debugging
                logger.exception(f"CIO barrier monitoring error: {e}")
                await asyncio.sleep(1.0)

    async def _process_barrier_signals(self, signals: dict[str, SignalEvent]) -> None:
        """
        Process all signals from completed barrier (fan-in).

        Groups signals by symbol and makes decisions for each.
        Also reviews existing positions for autonomous management.
        """
        # Check for emergency deleveraging BEFORE processing new signals
        deleverage_triggered = await self._check_and_deleverage()
        if deleverage_triggered:
            # Skip normal signal processing when actively deleveraging
            return

        # AUTONOMOUS POSITION MANAGEMENT: Review existing positions first
        if self._position_management_enabled:
            await self._review_and_manage_positions(signals)

        # Group signals by symbol
        by_symbol: dict[str, dict[str, SignalEvent]] = {}
        for agent_name, signal in signals.items():
            symbol = signal.symbol
            if symbol not in by_symbol:
                by_symbol[symbol] = {}
            by_symbol[symbol][agent_name] = signal

        # Process each symbol with its signals
        for symbol, symbol_signals in by_symbol.items():
            agg = SignalAggregation(
                symbol=symbol,
                signals=symbol_signals,
                timestamp=datetime.now(timezone.utc),
            )
            await self._make_decision_from_aggregation(agg)

    async def _check_and_deleverage(self) -> bool:
        """
        Check leverage and generate deleveraging orders if needed.

        Returns True if deleveraging orders were generated, False otherwise.
        """
        if not self._broker:
            return False

        # Only check every 5 seconds to avoid spam
        now = datetime.now(timezone.utc)
        if (now - self._last_deleverage_check).total_seconds() < 5.0:
            return False
        self._last_deleverage_check = now

        try:
            # Get portfolio state from broker
            portfolio_state = await self._broker.get_portfolio_state()
            portfolio_value = portfolio_state.net_liquidation
            if portfolio_value <= 0:
                return False

            positions = portfolio_state.positions
            if not positions:
                return False

            # Calculate current gross exposure
            gross_exposure = sum(
                abs(pos.quantity * (pos.market_value / pos.quantity if pos.quantity != 0 else pos.avg_cost))
                for pos in positions.values()
                if pos.quantity != 0
            )

            current_leverage = gross_exposure / portfolio_value

            # If leverage is OK, nothing to do
            if current_leverage <= self._max_leverage:
                if self._deleveraging_active:
                    logger.info(f"CIO: Leverage normalized ({current_leverage:.2f}x <= {self._max_leverage}x), resuming normal operations")
                    self._deleveraging_active = False
                return False

            # Leverage is too high - need to deleverage
            if not self._deleveraging_active:
                logger.warning(
                    f"CIO: EMERGENCY DELEVERAGING TRIGGERED - Leverage {current_leverage:.2f}x exceeds limit {self._max_leverage}x"
                )
                self._deleveraging_active = True

            # Calculate how much to reduce
            target_exposure = portfolio_value * self._max_leverage * 0.95  # Target 95% of limit
            excess_exposure = gross_exposure - target_exposure

            if excess_exposure <= 0:
                return False

            # Sort positions by absolute market value (largest first)
            sorted_positions = sorted(
                [(sym, pos) for sym, pos in positions.items() if pos.quantity != 0],
                key=lambda x: abs(x[1].market_value),
                reverse=True
            )

            # Generate deleveraging orders for largest positions
            for symbol, pos in sorted_positions:
                if excess_exposure <= 0:
                    break

                # For futures: market_value = price × multiplier × quantity
                # So market_value / quantity = notional per contract (NOT the price!)
                notional_per_contract = abs(pos.market_value / pos.quantity) if pos.quantity != 0 else pos.avg_cost
                position_value = abs(pos.market_value)

                # Determine how much of this position to close
                close_value = min(position_value * 0.5, excess_exposure)  # Close max 50% at a time
                # Calculate contracts to close (minimum 1 to make progress)
                close_qty = max(1, int(close_value / notional_per_contract)) if notional_per_contract > 0 else 1

                # Don't close more than we have
                close_qty = min(close_qty, abs(pos.quantity))

                if close_qty <= 0:
                    continue

                logger.info(f"CIO DELEVERAGING calc: {symbol} notional/contract=${notional_per_contract:,.0f} close_value=${close_value:,.0f} -> close_qty={close_qty}")

                # Determine sell direction based on current position
                action = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY

                # Create deleveraging decision
                decision = DecisionEvent(
                    source_agent=self.name,
                    symbol=symbol,
                    action=action,
                    quantity=close_qty,
                    order_type=OrderType.MARKET,  # Use market orders for urgent deleveraging
                    limit_price=None,
                    rationale=f"EMERGENCY DELEVERAGING: Reducing leverage from {current_leverage:.2f}x to target {self._max_leverage}x",
                    contributing_signals=tuple(),
                    data_sources=("leverage_monitor",),
                    conviction_score=1.0,  # Max conviction for emergency orders
                )

                # Log decision
                logger.warning(
                    f"CIO DELEVERAGING: {action.value} {close_qty} {symbol} "
                    f"(reducing exposure by ${close_value:,.0f})"
                )

                self._audit_logger.log_decision(
                    agent_name=self.name,
                    decision_id=decision.event_id,
                    symbol=symbol,
                    action=decision.action.value,
                    quantity=close_qty,
                    rationale=decision.rationale,
                    data_sources=list(decision.data_sources),
                    contributing_signals=[],
                    conviction_score=1.0,
                )

                # Publish decision
                await self._event_bus.publish(decision)

                excess_exposure -= close_value

                # Track active decision
                self._active_decisions[decision.event_id] = datetime.now(timezone.utc)

            return True

        except Exception as e:
            logger.exception(f"CIO: Error checking leverage for deleveraging: {e}")
            return False

    # =========================================================================
    # AUTONOMOUS POSITION MANAGEMENT
    # =========================================================================

    async def _review_and_manage_positions(
        self,
        signals: dict[str, SignalEvent],
    ) -> None:
        """
        Review all tracked positions and make autonomous management decisions.

        This is the core autonomous decision-making method that:
        1. Closes losing positions exceeding loss thresholds
        2. Takes profits on winning positions
        3. Reduces position size when conviction drops
        4. Can override individual stop-losses in special situations
        5. Adapts rules based on market regime

        Called during each signal barrier cycle.
        """
        now = datetime.now(timezone.utc)

        # Rate limit position reviews
        if (now - self._last_position_review).total_seconds() < self._position_review_interval_seconds:
            return
        self._last_position_review = now

        # Sync tracked positions with broker positions
        await self._sync_tracked_positions()

        # EOD MANAGEMENT: Graduated close before market close (all instruments)
        # Phase 14+: Proportional schedule to avoid order clustering at close
        current_hour, _, hours_remaining = self._get_et_time()
        if self._is_eod_force_close_time:
            # 15:45 ET: FORCE CLOSE everything with market orders
            for symbol, pos in list(self._tracked_positions.items()):
                logger.warning(f"CIO EOD FORCE CLOSE 15:45 ET: Closing {symbol} (P&L: {pos.pnl_pct:.1f}%)")
                decision = self._create_close_loser_decision(
                    pos, reason="EOD FORCE CLOSE: 15:45 ET - market order", is_emergency=True,
                )
                if decision:
                    await self._publish_position_management_decision(decision)
            return
        if self._is_eod_liquidation_time:
            # 15:30 ET: Close all remaining positions with limit orders
            for symbol, pos in list(self._tracked_positions.items()):
                logger.warning(f"CIO EOD LIQUIDATION 15:30 ET: Closing {symbol} (P&L: {pos.pnl_pct:.1f}%)")
                decision = self._create_close_loser_decision(
                    pos, reason=f"EOD LIQUIDATION: 15:30 ET - closing all (P&L: {pos.pnl_pct:.1f}%)",
                )
                if decision:
                    await self._publish_position_management_decision(decision)
            return
        if self._is_past_new_position_cutoff:
            # 15:00-15:30 ET: Graduated close - losers first, then all
            # Close losing positions immediately after 15:00
            for symbol, pos in list(self._tracked_positions.items()):
                if pos.pnl_pct < 0:
                    logger.info(f"CIO EOD GRADUATED 15:00+ ET: Closing loser {symbol} (P&L: {pos.pnl_pct:.1f}%)")
                    decision = self._create_close_loser_decision(
                        pos, reason=f"EOD GRADUATED: 15:00+ ET - closing loser (P&L: {pos.pnl_pct:.1f}%)",
                    )
                    if decision:
                        await self._publish_position_management_decision(decision)
            # After 15:15 ET: close winners too
            if current_hour >= 15.25:
                for symbol, pos in list(self._tracked_positions.items()):
                    if pos.pnl_pct >= 0:
                        logger.info(f"CIO EOD GRADUATED 15:15+ ET: Closing winner {symbol} (P&L: {pos.pnl_pct:.1f}%)")
                        decision = self._create_take_profit_decision(
                            pos, reason=f"EOD GRADUATED: 15:15+ ET - locking in {pos.pnl_pct:.1f}% gain",
                            partial_exit=False,
                        )
                        if decision:
                            await self._publish_position_management_decision(decision)
            return

        # CLEANUP: Close any positions on no-trade symbols (stocks, ETFs, forex)
        # These should only be traded as futures
        if self._broker:
            try:
                portfolio_state = await self._broker.get_portfolio_state()
                for sym, pos_info in portfolio_state.positions.items():
                    if pos_info.quantity != 0 and sym.upper() in self._no_trade_symbols:
                        close_qty = abs(pos_info.quantity)
                        action = OrderSide.SELL if pos_info.quantity > 0 else OrderSide.BUY
                        logger.warning(
                            f"CIO: Closing NO-TRADE symbol {sym} ({pos_info.quantity} shares) - "
                            f"only futures should be traded"
                        )
                        decision = DecisionEvent(
                            source_agent=self.name,
                            symbol=sym,
                            action=action,
                            quantity=close_qty,
                            order_type=OrderType.MARKET,
                            limit_price=None,
                            rationale=f"CLEANUP: {sym} is a no-trade symbol (stock/ETF), closing position",
                            contributing_signals=tuple(),
                            data_sources=("position_management",),
                            conviction_score=1.0,
                        )
                        await self._event_bus.publish(decision)
            except Exception as e:
                logger.debug(f"CIO: Could not check no-trade positions: {e}")

        # Update conviction from latest signals
        self._update_position_convictions(signals)

        # Review each position
        for symbol, pos in list(self._tracked_positions.items()):
            decision = await self._evaluate_position_for_management(pos)
            if decision:
                await self._publish_position_management_decision(decision)

    async def _sync_tracked_positions(self) -> None:
        """
        Sync CIO tracked positions with broker portfolio state.

        Ensures we're tracking all open positions and removes closed ones.
        """
        if not self._broker:
            return

        try:
            portfolio_state = await self._broker.get_portfolio_state()
            broker_positions = portfolio_state.positions

            # Update existing tracked positions with latest prices
            for symbol, pos in broker_positions.items():
                if pos.quantity == 0:
                    # Position closed - remove from tracking
                    if symbol in self._tracked_positions:
                        del self._tracked_positions[symbol]
                    continue

                # FIX: Convert IB notional to actual price using contract multiplier
                # IB market_value = price * multiplier * quantity for futures
                spec = CONTRACT_SPECS.get(symbol)
                multiplier = spec.multiplier if spec else 1.0
                raw_price = (pos.market_value / pos.quantity) if pos.quantity != 0 else pos.avg_cost
                price = raw_price / multiplier if multiplier != 1.0 else raw_price

                if symbol in self._tracked_positions:
                    # Update existing position
                    tracked = self._tracked_positions[symbol]
                    tracked.update_price(price)
                    tracked.quantity = abs(pos.quantity)  # FIX-08: IB gives signed qty, TrackedPosition expects unsigned
                    tracked.is_long = pos.quantity > 0  # Update direction from IB
                else:
                    # New position we're not tracking yet - add it
                    is_long = pos.quantity > 0
                    # IB avg_cost = price * multiplier for ALL futures (including multiplier < 1)
                    entry_price = pos.avg_cost / multiplier if multiplier != 1.0 else pos.avg_cost
                    # Estimate initial risk (2% default for unknown positions)
                    est_initial_risk = entry_price * 0.02
                    self._tracked_positions[symbol] = TrackedPosition(
                        symbol=symbol,
                        quantity=abs(pos.quantity),
                        entry_price=entry_price,
                        entry_time=datetime.now(timezone.utc),  # Approximate
                        is_long=is_long,
                        current_price=price,
                        highest_price=price if is_long else 0.0,
                        lowest_price=price if not is_long else float("inf"),
                        original_conviction=0.5,  # Unknown, use neutral
                        current_conviction=0.5,
                        initial_risk=est_initial_risk,  # Phase 3: R-Multiple tracking
                        contract_multiplier=multiplier,
                    )
                    logger.info(
                        f"CIO: Started tracking existing position {symbol}: "
                        f"{'LONG' if is_long else 'SHORT'} {abs(pos.quantity)} @ ${entry_price:.2f}"
                    )

            # Remove tracked positions that no longer exist
            symbols_to_remove = [
                sym for sym in self._tracked_positions
                if sym not in broker_positions or broker_positions[sym].quantity == 0
            ]
            for sym in symbols_to_remove:
                del self._tracked_positions[sym]
                logger.info(f"CIO: Stopped tracking closed position {sym}")

        except Exception as e:
            logger.exception(f"CIO: Error syncing tracked positions: {e}")

    def _update_position_convictions(self, signals: dict[str, SignalEvent]) -> None:
        """
        Update position convictions based on latest signals.

        Conviction is derived from signal confidence and direction alignment.
        """
        now = datetime.now(timezone.utc)

        for symbol, pos in self._tracked_positions.items():
            # Find signals for this symbol
            symbol_signals = [
                sig for sig in signals.values()
                if sig.symbol == symbol
            ]

            if not symbol_signals:
                continue

            # Calculate weighted conviction from signals
            total_weight = 0.0
            weighted_conviction = 0.0

            for signal in symbol_signals:
                agent_weight = self._weights.get(signal.source_agent, 0.1)

                # Check if signal direction aligns with position direction
                aligned = (
                    (pos.is_long and signal.direction == SignalDirection.LONG) or
                    (not pos.is_long and signal.direction == SignalDirection.SHORT)
                )

                # Confidence contribution: positive if aligned, negative if opposed
                if aligned:
                    contribution = signal.confidence
                elif signal.direction == SignalDirection.FLAT:
                    contribution = 0.3  # Neutral signal = low conviction
                else:
                    contribution = -signal.confidence  # Opposing signal = negative

                weighted_conviction += agent_weight * contribution
                total_weight += agent_weight

            if total_weight > 0:
                new_conviction = (weighted_conviction / total_weight + 1) / 2  # Normalize to 0-1
                new_conviction = max(0.0, min(1.0, new_conviction))
                pos.current_conviction = new_conviction
                pos.last_signal_time = now
                pos.contributing_strategies = [sig.source_agent for sig in symbol_signals]

    async def _evaluate_position_for_management(
        self,
        pos: TrackedPosition,
    ) -> DecisionEvent | None:
        """
        Evaluate a single position and determine if management action is needed.

        Returns a DecisionEvent if action should be taken, None otherwise.

        Decision hierarchy (evaluated in order):
        1. Emergency loss cut (extended_loss_pct)
        2. Standard loss cut (max_loss_pct + time condition)
        3. BREAKEVEN STOP at 1R (Phase 12 - move stop to entry)
        4. TRAILING STOP at 1.5R+ (Phase 12 - trail behind peak)
        5. GRADUATED PARTIAL PROFITS at 1.5R, 2.5R, 3.5R (Phase 12)
        6. Take profit (profit_target_pct + trailing from peak)
        7. Conviction-based reduction
        8. TIME-BASED EXIT per strategy type (Phase 12)
        9. Stale signal exit
        10. Stop-loss override check (trending markets)
        """
        config = self._position_management_config
        regime = self._current_regime

        # Adjust thresholds based on market regime
        effective_loss_pct = self._get_regime_adjusted_loss_threshold(regime)
        effective_profit_pct = self._get_regime_adjusted_profit_threshold(regime)

        # 1. EMERGENCY LOSS CUT - Always trigger at extended loss
        if pos.pnl_pct <= -config.extended_loss_pct:
            return self._create_close_loser_decision(
                pos,
                reason=f"EMERGENCY: Loss exceeds {config.extended_loss_pct}% threshold",
                is_emergency=True,
            )

        # 2. STANDARD LOSS CUT - Loss threshold + time condition
        if pos.pnl_pct <= -effective_loss_pct:
            if pos.holding_hours >= config.loss_time_threshold_hours:
                return self._create_close_loser_decision(
                    pos,
                    reason=(
                        f"Loss {pos.pnl_pct:.1f}% exceeds {effective_loss_pct}% "
                        f"and held for {pos.holding_hours:.1f} hours"
                    ),
                )
            elif pos.current_conviction < config.min_conviction_to_hold:
                return self._create_close_loser_decision(
                    pos,
                    reason=(
                        f"Loss {pos.pnl_pct:.1f}% with low conviction "
                        f"({pos.current_conviction:.1%} < {config.min_conviction_to_hold:.1%})"
                    ),
                )

        # =====================================================================
        # Phase 12: ACTIVE POSITION PROTECTION SYSTEM
        # =====================================================================
        r_mult = pos.r_multiple

        # 3. BREAKEVEN STOP - Move stop to entry price at 1R profit
        if (
            r_mult >= config.breakeven_r_trigger
            and not pos.stop_moved_to_breakeven
            and pos.initial_risk > 0
        ):
            buffer = pos.entry_price * (config.breakeven_buffer_pct / 100)
            if pos.is_long:
                new_stop = pos.entry_price + buffer  # Slightly above entry
            else:
                new_stop = pos.entry_price - buffer  # Slightly below entry
            pos.stop_loss_level = new_stop
            pos.stop_moved_to_breakeven = True
            self._position_management_stats["breakeven_stops_set"] += 1
            logger.info(
                f"CIO: {pos.symbol} reached {r_mult:.1f}R - BREAKEVEN STOP set at "
                f"${new_stop:.2f} (entry ${pos.entry_price:.2f})"
            )

        # 4. TRAILING STOP - Activate at 1.5R, trail 0.5R behind peak
        if (
            r_mult >= config.trailing_activation_r
            and pos.initial_risk > 0
        ):
            trail_distance = pos.initial_risk * config.trailing_distance_r
            if pos.is_long:
                new_trailing = pos.highest_price - trail_distance
            else:
                new_trailing = pos.lowest_price + trail_distance

            # Only update if new trailing level is better than current
            if pos.trailing_stop_level is None or (
                (pos.is_long and new_trailing > pos.trailing_stop_level) or
                (not pos.is_long and new_trailing < pos.trailing_stop_level)
            ):
                pos.trailing_stop_level = new_trailing
                pos.trailing_stop_active = True
                pos.stop_loss_level = new_trailing  # Update actual stop
                logger.debug(
                    f"CIO: {pos.symbol} trailing stop updated to ${new_trailing:.2f} "
                    f"(peak R: {pos.peak_r_multiple:.1f}R, current: {r_mult:.1f}R)"
                )

        # 4b. CHECK TRAILING STOP HIT
        if pos.trailing_stop_active and pos.trailing_stop_level is not None:
            trailing_hit = (
                (pos.is_long and pos.current_price <= pos.trailing_stop_level) or
                (not pos.is_long and pos.current_price >= pos.trailing_stop_level)
            )
            if trailing_hit:
                self._position_management_stats["trailing_stops_triggered"] += 1
                return self._create_take_profit_decision(
                    pos,
                    reason=(
                        f"TRAILING STOP hit at ${pos.trailing_stop_level:.2f} "
                        f"(peak {pos.peak_r_multiple:.1f}R, exit {r_mult:.1f}R)"
                    ),
                    partial_exit=False,
                )

        # 5. GRADUATED PARTIAL PROFIT TAKING (1.5R → 2.5R → 3.5R)
        if r_mult > 0 and pos.initial_risk > 0:
            # First partial: 33% at 1.5R
            if r_mult >= config.partial_profit_1_r and pos.partial_exits_taken == 0:
                pos.partial_exits_taken = 1
                self._position_management_stats["partial_profits_1r5"] += 1
                logger.info(
                    f"CIO: {pos.symbol} reached {r_mult:.1f}R - "
                    f"taking 33% partial profit (1st tranche)"
                )
                return self._create_take_profit_decision(
                    pos,
                    reason=f"R-Multiple {r_mult:.1f}R - partial profit 1/3 at {config.partial_profit_1_r}R",
                    partial_exit=True,
                    partial_pct=33.0,
                )

            # Second partial: 33% at 2.5R
            if r_mult >= config.partial_profit_2_r and pos.partial_exits_taken == 1:
                pos.partial_exits_taken = 2
                self._position_management_stats["partial_profits_2r5"] += 1
                logger.info(
                    f"CIO: {pos.symbol} reached {r_mult:.1f}R - "
                    f"taking 33% partial profit (2nd tranche)"
                )
                return self._create_take_profit_decision(
                    pos,
                    reason=f"R-Multiple {r_mult:.1f}R - partial profit 2/3 at {config.partial_profit_2_r}R",
                    partial_exit=True,
                    partial_pct=50.0,  # 50% of remaining = ~33% of original
                )

            # Final exit: 100% at 3.5R or let trailing stop handle it
            if r_mult >= config.partial_profit_3_r and pos.partial_exits_taken == 2:
                pos.partial_exits_taken = 3
                self._position_management_stats["full_profit_exits"] += 1
                logger.info(
                    f"CIO: {pos.symbol} reached {r_mult:.1f}R - "
                    f"FULL EXIT at {config.partial_profit_3_r}R"
                )
                return self._create_take_profit_decision(
                    pos,
                    reason=f"R-Multiple {r_mult:.1f}R - full exit at {config.partial_profit_3_r}R target",
                    partial_exit=False,
                )

            # Log R-multiple for tracking
            if abs(r_mult) >= 0.5:
                logger.debug(
                    f"CIO: {pos.symbol} R={r_mult:.2f} (peak={pos.peak_r_multiple:.2f}R, "
                    f"partials={pos.partial_exits_taken}, BE={pos.stop_moved_to_breakeven}, "
                    f"trail={pos.trailing_stop_active})"
                )

        # 6. TAKE PROFIT - Percentage-based (fallback for positions without R-multiple)
        if pos.pnl_pct >= effective_profit_pct:
            # Check trailing condition
            pullback = pos.drawdown_from_peak_pct if pos.is_long else pos.rally_from_trough_pct
            if pullback >= config.trailing_profit_pct:
                return self._create_take_profit_decision(
                    pos,
                    reason=(
                        f"Take profit: Gain {pos.pnl_pct:.1f}% with "
                        f"{pullback:.1f}% pullback from peak"
                    ),
                    partial_exit=False,
                )

        # 6.5 TIME DECAY ON CONVICTION (C5 - Intraday urgency)
        # As time progresses, conviction decays to force exits before EOD
        if pos.holding_hours >= 2.0:
            if self._is_past_new_position_cutoff:
                # Past 15:00 ET: zero conviction -> force exit on next check
                pos.current_conviction = 0.0
                logger.info(
                    f"CIO: Time decay zeroed conviction for {pos.symbol} "
                    f"(past 15:00 ET cutoff)"
                )
            elif pos.holding_hours >= 3.0:
                pos.current_conviction *= 0.8
                logger.debug(
                    f"CIO: Time decay 0.8x for {pos.symbol} "
                    f"(held {pos.holding_hours:.1f}h)"
                )
            else:
                pos.current_conviction *= 0.9
                logger.debug(
                    f"CIO: Time decay 0.9x for {pos.symbol} "
                    f"(held {pos.holding_hours:.1f}h)"
                )

        # 7. CONVICTION-BASED REDUCTION
        if pos.original_conviction > 0:
            conviction_drop = (pos.original_conviction - pos.current_conviction) / pos.original_conviction
            if conviction_drop >= config.conviction_drop_threshold:
                return self._create_reduce_position_decision(
                    pos,
                    reduction_pct=50.0,  # Reduce by 50% when conviction drops
                    reason=(
                        f"Conviction dropped {conviction_drop:.0%} "
                        f"(from {pos.original_conviction:.1%} to {pos.current_conviction:.1%})"
                    ),
                )

        # 8. TIME-BASED EXIT (Phase 12: per-strategy-type holding limits)
        max_hours = self._get_max_holding_hours(pos, config)
        if pos.holding_hours >= max_hours:
            # If profitable, take profit; if losing, close loser
            if pos.pnl_pct > 0:
                return self._create_take_profit_decision(
                    pos,
                    reason=(
                        f"Time exit: held {pos.holding_hours:.1f}h (max {max_hours:.0f}h for "
                        f"{pos.strategy_type or 'default'}), locking in {pos.pnl_pct:.1f}% gain"
                    ),
                    partial_exit=False,
                )
            else:
                return self._create_close_loser_decision(
                    pos,
                    reason=(
                        f"Time exit: held {pos.holding_hours:.1f}h (max {max_hours:.0f}h for "
                        f"{pos.strategy_type or 'default'}), cutting {pos.pnl_pct:.1f}% loss"
                    ),
                )

        # 9. STALE SIGNAL EXIT
        if pos.current_conviction < config.min_conviction_to_hold:
            stale_hours = config.stale_signal_hours
            if pos.last_signal_time:
                hours_since_signal = (datetime.now(timezone.utc) - pos.last_signal_time).total_seconds() / 3600
                if hours_since_signal > stale_hours:
                    return self._create_close_loser_decision(
                        pos,
                        reason=(
                            f"Low conviction ({pos.current_conviction:.1%}) with stale signals "
                            f"(last signal {hours_since_signal:.1f}h ago)"
                        ),
                    )

        # 10. STOP-LOSS OVERRIDE CHECK (for trending markets)
        if (
            config.allow_stop_override_in_trending and
            regime == MarketRegime.TRENDING and
            pos.stop_loss_level and
            pos.current_conviction >= config.stop_override_min_conviction and
            not pos.stop_loss_overridden
            and not pos.stop_moved_to_breakeven  # Don't override breakeven stops
        ):
            # Check if price is near stop-loss but position still looks good
            if pos.is_long:
                stop_proximity = (pos.current_price - pos.stop_loss_level) / pos.current_price
                if 0 < stop_proximity < 0.02:  # Within 2% of stop
                    logger.info(
                        f"CIO: Overriding stop-loss for {pos.symbol} in trending market "
                        f"(conviction {pos.current_conviction:.1%})"
                    )
                    pos.stop_loss_overridden = True
                    self._position_management_stats["stop_overrides"] += 1

        return None

    def _get_max_holding_hours(
        self,
        pos: TrackedPosition,
        config: PositionManagementConfig,
    ) -> float:
        """
        Get maximum holding hours based on strategy type.

        Phase 12: Different strategies have different optimal holding periods.
        Intraday strategies (Session, TTM, MeanReversion) should close within 4h.
        Swing strategies (Momentum, MACD-v, Macro) can hold 48h.
        Pairs/spread strategies (StatArb, IndexSpread) can hold 5 days.
        """
        strategy_type = pos.strategy_type.lower() if pos.strategy_type else ""

        # Map contributing strategies to type
        intraday_agents = {"SessionAgent", "TTMSqueezeAgent", "MeanReversionAgent", "EventDrivenAgent"}
        swing_agents = {"MomentumAgent", "MACDvAgent", "MacroAgent", "MarketMakingAgent"}
        pairs_agents = {"StatArbAgent", "IndexSpreadAgent"}

        strategies = set(pos.contributing_strategies)

        if strategy_type == "intraday" or strategies & intraday_agents:
            return config.max_holding_hours_intraday
        elif strategy_type == "pairs" or strategies & pairs_agents:
            return config.max_holding_hours_pairs
        elif strategy_type == "swing" or strategies & swing_agents:
            return config.max_holding_hours_swing
        else:
            return config.max_holding_days * 24

    def _get_regime_adjusted_loss_threshold(self, regime: MarketRegime) -> float:
        """Get loss threshold adjusted for market regime."""
        config = self._position_management_config

        if regime == MarketRegime.VOLATILE:
            return config.volatile_regime_loss_pct
        elif regime == MarketRegime.RISK_OFF:
            return config.max_loss_pct * 0.8  # Tighter stops in risk-off
        else:
            return config.max_loss_pct

    def _get_regime_adjusted_profit_threshold(self, regime: MarketRegime) -> float:
        """Get profit threshold adjusted for market regime."""
        config = self._position_management_config

        if regime == MarketRegime.TRENDING:
            return config.profit_target_pct * config.trending_regime_profit_mult
        elif regime == MarketRegime.VOLATILE:
            return config.profit_target_pct * 0.7  # Take profits earlier
        else:
            return config.profit_target_pct

    def _create_close_loser_decision(
        self,
        pos: TrackedPosition,
        reason: str,
        is_emergency: bool = False,
    ) -> DecisionEvent:
        """Create a decision to close a losing position."""
        action = OrderSide.SELL if pos.is_long else OrderSide.BUY

        decision = DecisionEvent(
            source_agent=self.name,
            symbol=pos.symbol,
            action=action,
            quantity=pos.quantity,
            order_type=OrderType.MARKET if is_emergency else OrderType.LIMIT,
            limit_price=None,
            rationale=f"CIO CLOSE LOSER: {reason}",
            contributing_signals=tuple(pos.contributing_strategies),
            data_sources=("position_management", "portfolio_state"),
            conviction_score=1.0 if is_emergency else 0.8,
            decision_action=DecisionAction.CLOSE_LOSER,
            position_pnl_pct=pos.pnl_pct,
            holding_duration_hours=pos.holding_hours,
            stop_loss_override=pos.stop_loss_overridden,
            regime_context=self._current_regime.value,
        )

        self._position_management_stats["losers_closed"] += 1
        # Phase 3: Track R-multiple at exit
        if pos.r_multiple != 0:
            self._position_management_stats["total_r_multiple_closed"] += pos.r_multiple
            self._position_management_stats["closed_trade_count"] += 1
            logger.info(f"CIO: Closed loser {pos.symbol} at {pos.r_multiple:.2f}R")
        return decision

    def _create_take_profit_decision(
        self,
        pos: TrackedPosition,
        reason: str,
        partial_exit: bool = False,
        partial_pct: float = 33.0,
    ) -> DecisionEvent:
        """Create a decision to take profit on a winning position."""
        action = OrderSide.SELL if pos.is_long else OrderSide.BUY
        if partial_exit:
            quantity = max(1, int(pos.quantity * partial_pct / 100))
        else:
            quantity = pos.quantity

        decision = DecisionEvent(
            source_agent=self.name,
            symbol=pos.symbol,
            action=action,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=None,
            rationale=f"CIO TAKE PROFIT: {reason}",
            contributing_signals=tuple(pos.contributing_strategies),
            data_sources=("position_management", "portfolio_state"),
            conviction_score=0.9,
            decision_action=DecisionAction.TAKE_PROFIT,
            position_pnl_pct=pos.pnl_pct,
            holding_duration_hours=pos.holding_hours,
            regime_context=self._current_regime.value,
        )

        self._position_management_stats["profits_taken"] += 1
        # Phase 3: Track R-multiple at exit
        if pos.r_multiple != 0:
            self._position_management_stats["total_r_multiple_closed"] += pos.r_multiple
            self._position_management_stats["closed_trade_count"] += 1
            if pos.r_multiple >= 3.0:
                self._position_management_stats["r_multiple_exits_3r"] += 1
            elif pos.r_multiple >= 2.0:
                self._position_management_stats["r_multiple_exits_2r"] += 1
            logger.info(f"CIO: Took profit {pos.symbol} at {pos.r_multiple:.2f}R")
        return decision

    def _create_reduce_position_decision(
        self,
        pos: TrackedPosition,
        reduction_pct: float,
        reason: str,
    ) -> DecisionEvent:
        """Create a decision to reduce position size."""
        action = OrderSide.SELL if pos.is_long else OrderSide.BUY
        reduce_qty = max(1, int(pos.quantity * reduction_pct / 100))

        decision = DecisionEvent(
            source_agent=self.name,
            symbol=pos.symbol,
            action=action,
            quantity=reduce_qty,
            order_type=OrderType.LIMIT,
            limit_price=None,
            rationale=f"CIO REDUCE POSITION: {reason}",
            contributing_signals=tuple(pos.contributing_strategies),
            data_sources=("position_management", "signal_analysis"),
            conviction_score=0.7,
            decision_action=DecisionAction.REDUCE_POSITION,
            position_pnl_pct=pos.pnl_pct,
            holding_duration_hours=pos.holding_hours,
            regime_context=self._current_regime.value,
        )

        self._position_management_stats["positions_reduced"] += 1
        return decision

    async def _publish_position_management_decision(
        self,
        decision: DecisionEvent,
    ) -> None:
        """Publish a position management decision with full audit logging."""
        # Log decision (COMPLIANCE REQUIREMENT)
        self._audit_logger.log_decision(
            agent_name=self.name,
            decision_id=decision.event_id,
            symbol=decision.symbol,
            action=decision.action.value if decision.action else "none",
            quantity=decision.quantity,
            rationale=decision.rationale,
            data_sources=list(decision.data_sources),
            contributing_signals=list(decision.contributing_signals),
            conviction_score=decision.conviction_score,
        )

        # Log position management specific event
        self._audit_logger.log_agent_event(
            agent_name=self.name,
            event_type="position_management_decision",
            details={
                "decision_id": decision.event_id,
                "symbol": decision.symbol,
                "action": decision.action.value if decision.action else None,
                "decision_action": decision.decision_action.value if decision.decision_action else None,
                "quantity": decision.quantity,
                "position_pnl_pct": decision.position_pnl_pct,
                "holding_duration_hours": decision.holding_duration_hours,
                "stop_loss_override": decision.stop_loss_override,
                "regime_context": decision.regime_context,
                "rationale": decision.rationale,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Publish decision
        await self._event_bus.publish(decision)

        # Track active decision
        self._active_decisions[decision.event_id] = datetime.now(timezone.utc)

        logger.info(
            f"CIO POSITION MANAGEMENT: {decision.decision_action.value if decision.decision_action else 'ACTION'} "
            f"{decision.quantity} {decision.symbol} "
            f"(PnL: {decision.position_pnl_pct:.1f}%, held: {decision.holding_duration_hours:.1f}h)"
        )

    def register_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        is_long: bool,
        conviction: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        contributing_strategies: list[str] | None = None,
    ) -> None:
        """
        Register a new position for CIO tracking.

        Called by the orchestrator when a position is opened.
        """
        # Phase 3: Calculate initial risk (1R) from stop-loss
        initial_risk = 0.0
        if stop_loss is not None and stop_loss > 0 and entry_price > 0:
            initial_risk = abs(entry_price - stop_loss)
        else:
            # Default to 2% of entry price if no stop-loss defined
            initial_risk = entry_price * 0.02

        # Phase 12: Determine strategy type for time-based exits
        strategies = contributing_strategies or []
        strategy_type = self._classify_strategy_type(strategies)

        self._tracked_positions[symbol] = TrackedPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(timezone.utc),
            is_long=is_long,
            current_price=entry_price,
            highest_price=entry_price if is_long else 0.0,
            lowest_price=entry_price if not is_long else float("inf"),
            original_conviction=conviction,
            current_conviction=conviction,
            last_signal_time=datetime.now(timezone.utc),
            contributing_strategies=strategies,
            stop_loss_level=stop_loss,
            take_profit_level=take_profit,
            initial_risk=initial_risk,  # Phase 3: R-Multiple tracking
            strategy_type=strategy_type,  # Phase 12: for time-based exits
        )

        logger.info(
            f"CIO: Registered new position {symbol}: "
            f"{'LONG' if is_long else 'SHORT'} {quantity} @ ${entry_price:.2f} "
            f"(conviction: {conviction:.1%}, type: {strategy_type}, "
            f"1R=${initial_risk:.2f}, SL=${stop_loss or 0:.2f})"
        )

    def _classify_strategy_type(self, strategies: list[str]) -> str:
        """Classify position as intraday/swing/pairs based on contributing strategies."""
        intraday = {"SessionAgent", "TTMSqueezeAgent", "MeanReversionAgent", "EventDrivenAgent"}
        pairs = {"StatArbAgent", "IndexSpreadAgent"}
        # swing is the default

        strategy_set = set(strategies)
        if strategy_set & intraday:
            return "intraday"
        elif strategy_set & pairs:
            return "pairs"
        else:
            return "swing"

    def get_tracked_positions(self) -> dict[str, dict[str, Any]]:
        """Get all tracked positions with their status."""
        return {
            symbol: {
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "is_long": pos.is_long,
                "pnl_pct": pos.pnl_pct,
                "pnl_dollar": pos.pnl_dollar,
                "holding_hours": pos.holding_hours,
                "original_conviction": pos.original_conviction,
                "current_conviction": pos.current_conviction,
                "stop_loss_level": pos.stop_loss_level,
                "stop_loss_overridden": pos.stop_loss_overridden,
                "highest_price": pos.highest_price,
                "drawdown_from_peak_pct": pos.drawdown_from_peak_pct,
                "contributing_strategies": pos.contributing_strategies,
                # Phase 12: Active Position Protection
                "r_multiple": pos.r_multiple,
                "peak_r_multiple": pos.peak_r_multiple,
                "breakeven_stop_set": pos.stop_moved_to_breakeven,
                "trailing_stop_active": pos.trailing_stop_active,
                "trailing_stop_level": pos.trailing_stop_level,
                "partial_exits_taken": pos.partial_exits_taken,
                "strategy_type": pos.strategy_type,
            }
            for symbol, pos in self._tracked_positions.items()
        }

    def get_position_management_stats(self) -> dict[str, Any]:
        """Get position management statistics."""
        return {
            "tracked_positions": len(self._tracked_positions),
            "management_enabled": self._position_management_enabled,
            "config": self._position_management_config.to_dict(),
            "stats": dict(self._position_management_stats),
            "last_review": self._last_position_review.isoformat(),
        }

    async def process_event(self, event: Event) -> None:
        """
        Process validated decision events.

        Signal processing is handled by the barrier monitoring loop,
        not by individual event subscription (per CLAUDE.md fan-in).
        """
        if isinstance(event, ValidatedDecisionEvent):
            # Clean up decision tracking when decisions are processed
            await self._handle_validated_decision(event)

    async def _handle_validated_decision(self, event: ValidatedDecisionEvent) -> None:
        """Handle validated decision event - clean up tracking."""
        decision_id = event.original_decision_id
        if decision_id in self._active_decisions:
            del self._active_decisions[decision_id]
            if event.approved:
                logger.debug(f"CIO: Decision {decision_id[:8]} approved and cleared")
            else:
                logger.debug(f"CIO: Decision {decision_id[:8]} rejected and cleared")

    def _cleanup_stale_decisions(self) -> None:
        """Remove decisions older than timeout."""
        now = datetime.now(timezone.utc)
        stale_ids = [
            did for did, created in self._active_decisions.items()
            if (now - created).total_seconds() > self._decision_timeout_seconds
        ]
        for did in stale_ids:
            del self._active_decisions[did]
            logger.debug(f"CIO: Cleaned up stale decision {did[:8]}")

    def _check_sector_concentration(self, symbol: str, proposed_size: int) -> tuple[bool, str]:
        """
        Check if adding a position would exceed sector concentration limits (PM-01).

        Args:
            symbol: The symbol to trade
            proposed_size: The proposed position size in shares

        Returns:
            Tuple of (is_allowed, rejection_reason)
            - is_allowed: True if trade is allowed, False if it would exceed limits
            - rejection_reason: Empty string if allowed, explanation if rejected
        """
        sector = self._symbol_to_sector.get(symbol, "Unknown")

        # Get current sector exposure
        current_sector_exposure = self._sector_positions.get(sector, 0.0)

        # Calculate proposed additional exposure
        price = self._price_cache.get(symbol, 0.0)
        if price <= 0:
            # Can't calculate exposure without price, allow trade but log warning
            logger.warning(f"CIO: No price for {symbol}, cannot check sector concentration")
            return True, ""

        proposed_value = proposed_size * price
        proposed_exposure = proposed_value / self._portfolio_value if self._portfolio_value > 0 else 0

        # Total exposure after this trade
        total_sector_exposure = current_sector_exposure + proposed_exposure

        if total_sector_exposure > self._max_sector_concentration:
            rejection_reason = (
                f"Sector concentration limit exceeded: {sector} would be "
                f"{total_sector_exposure:.1%} (max {self._max_sector_concentration:.1%})"
            )
            return False, rejection_reason

        return True, ""

    def update_sector_mapping(self, symbol: str, sector: str) -> None:
        """
        Update the symbol to sector mapping (PM-01).

        Called by orchestrator or data provider with sector classification.
        """
        self._symbol_to_sector[symbol] = sector

    def update_sector_positions(self, sector_positions: dict[str, float]) -> None:
        """
        Update current sector exposures (PM-01).

        Args:
            sector_positions: Dict mapping sector name to exposure as fraction of portfolio
        """
        self._sector_positions = dict(sector_positions)

    def update_portfolio_drawdown(self, current_value: float) -> None:
        """
        Update portfolio drawdown tracking for Kelly adjustment (PM-06).

        Args:
            current_value: Current portfolio value
        """
        # Update peak if current value exceeds it
        if current_value > self._portfolio_peak:
            self._portfolio_peak = current_value
            self._portfolio_drawdown = 0.0
        elif self._portfolio_peak > 0:
            # Calculate drawdown as percentage from peak
            self._portfolio_drawdown = (self._portfolio_peak - current_value) / self._portfolio_peak
        else:
            self._portfolio_drawdown = 0.0

        # Update portfolio value for sizing
        self._portfolio_value = current_value

    def set_stress_mode(self, in_stress: bool) -> None:
        """
        Set stress mode flag for correlation adjustment (PM-12).

        When in stress mode, use stress correlations instead of normal correlations.
        """
        if in_stress != self._in_stress_mode:
            self._in_stress_mode = in_stress
            logger.info(f"CIO: Stress mode {'enabled' if in_stress else 'disabled'}")

    def update_stress_correlations(self, stress_correlations: dict[tuple[str, str], float]) -> None:
        """
        Update stress correlation cache (PM-12).

        Stress correlations are typically higher than normal correlations
        and should be used during market stress periods.
        """
        self._stress_correlation_cache = dict(stress_correlations)

    def _is_market_open_for_symbol(self, symbol: str) -> bool:
        """
        Check if trading is currently allowed for this symbol.

        US stocks/ETFs: only during regular market hours (9:30-16:00 ET, Mon-Fri).
        Futures: always allowed (nearly 24h trading).
        """
        sym = symbol.upper()

        # Only filter US stocks/ETFs by market hours
        if sym not in self._us_stock_symbols:
            return True  # Futures, pairs, etc. - always allowed

        try:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
        except ImportError:
            from datetime import timezone as tz
            # Fallback: approximate ET as UTC-5
            et_tz = timezone(timedelta(hours=-5))

        now_et = datetime.now(timezone.utc).astimezone(et_tz)

        # Weekend check
        if now_et.weekday() >= 5:
            return False

        # Regular US market hours: 9:30 AM - 4:00 PM ET
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        return market_open <= now_et.time() <= market_close

    def _get_et_time(self) -> tuple[float, float, float]:
        """Get current ET time as (hour_decimal, hours_into_session, hours_remaining)."""
        now = datetime.now(timezone.utc)
        try:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
            now_et = now.astimezone(et_tz)
        except ImportError:
            now_et = now.replace(tzinfo=None) - timedelta(hours=5)
        current_hour = now_et.hour + now_et.minute / 60.0
        market_open_hour = 9.5
        market_close_hour = 16.0
        hours_into_session = max(0, current_hour - market_open_hour)
        hours_remaining = max(0.0, market_close_hour - current_hour)
        return current_hour, hours_into_session, hours_remaining

    @property
    def _is_past_new_position_cutoff(self) -> bool:
        """True after 4:45 PM ET -- no new positions allowed (futures trade until 5pm ET)."""
        current_hour, _, _ = self._get_et_time()
        return current_hour >= 16.75

    @property
    def _is_eod_liquidation_time(self) -> bool:
        """True after 4:50 PM ET -- start closing all positions."""
        current_hour, _, _ = self._get_et_time()
        return current_hour >= 16.833

    @property
    def _is_eod_force_close_time(self) -> bool:
        """True after 4:55 PM ET -- force market orders for everything."""
        current_hour, _, _ = self._get_et_time()
        return current_hour >= 16.917

    async def _make_decision_from_aggregation(self, agg: SignalAggregation) -> None:
        """
        Make a trading decision from aggregated signals (post-barrier).

        This is THE decision point - all decisions go through here.
        Called only after barrier synchronization completes (fan-in).
        """
        symbol = agg.symbol

        # No-trade filter: ETFs (data-only for regime detection) and forex (IB paper rejects)
        if symbol.upper() in self._no_trade_symbols:
            return

        # Filter by market hours - prevent decisions for US stocks when market is closed
        if self._filter_market_hours and not self._is_market_open_for_symbol(symbol):
            return

        # INTRADAY: No new positions after 16:45 ET cutoff (futures close 17:00 ET)
        if self._is_past_new_position_cutoff and symbol not in self._tracked_positions:
            logger.info(f"CIO: EOD cutoff - no new positions after 16:45 ET, skipping {symbol}")
            return

        # Per-symbol cooldown: prevent rapid-fire decisions on same symbol
        now = datetime.now(timezone.utc)
        last_decision = self._last_decision_time_per_symbol.get(symbol)
        if last_decision:
            elapsed = (now - last_decision).total_seconds()
            if elapsed < self._decision_cooldown_seconds:
                logger.debug(f"CIO: Cooldown for {symbol} ({elapsed:.0f}s / {self._decision_cooldown_seconds:.0f}s)")
                return

        # Max open positions limit
        if len(self._tracked_positions) >= self._max_open_positions:
            if symbol not in self._tracked_positions:
                logger.warning(f"CIO: Max open positions ({self._max_open_positions}) reached, skipping {symbol}")
                return

        # FIX-04: LEVERAGE GUARD uses config max_leverage instead of hardcoded 1.5x
        # Don't open NEW positions when leverage exceeds configured limit
        if symbol not in self._tracked_positions and self._broker:
            try:
                portfolio_state = await self._broker.get_portfolio_state()
                pv = portfolio_state.net_liquidation
                if pv and pv > 0 and portfolio_state.positions:
                    gross_exp = sum(
                        abs(pos.market_value)
                        for pos in portfolio_state.positions.values()
                        if pos.quantity != 0
                    )
                    lev = gross_exp / pv
                    if lev > self._max_leverage:
                        logger.warning(
                            f"CIO: LEVERAGE GUARD - Blocking new position for {symbol} "
                            f"(leverage={lev:.2f}x > {self._max_leverage}x). Reduce existing positions first."
                        )
                        return
            except Exception:
                pass  # If we can't check leverage, proceed normally

        # Aggregate signals
        self._aggregate_signals(agg)

        # Check conviction threshold
        if agg.weighted_confidence < self._min_conviction:
            rejection_reason = (
                f"Insufficient conviction: {agg.weighted_confidence:.2f} < {self._min_conviction}"
            )
            logger.info(f"CIO: {rejection_reason} for {symbol}")
            # Audit log rejected decision (COMPLIANCE REQUIREMENT)
            self._audit_logger.log_agent_event(
                agent_name=self.name,
                event_type="decision_rejected",
                details={
                    "symbol": symbol,
                    "rejection_reason": rejection_reason,
                    "rejection_code": "INSUFFICIENT_CONVICTION",
                    "weighted_confidence": agg.weighted_confidence,
                    "min_conviction_threshold": self._min_conviction,
                    "contributing_signals": [s for s in agg.signals.keys()],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        # Clean up stale decisions first
        self._cleanup_stale_decisions()

        # Check concurrent decisions limit
        if len(self._active_decisions) >= self._max_concurrent:
            rejection_reason = f"Max concurrent decisions limit reached ({len(self._active_decisions)}/{self._max_concurrent})"
            logger.warning(f"CIO: {rejection_reason}, skipping {symbol}")
            # Audit log rejected decision (COMPLIANCE REQUIREMENT)
            self._audit_logger.log_agent_event(
                agent_name=self.name,
                event_type="decision_rejected",
                details={
                    "symbol": symbol,
                    "rejection_reason": rejection_reason,
                    "rejection_code": "MAX_CONCURRENT_DECISIONS",
                    "active_decisions": len(self._active_decisions),
                    "max_concurrent": self._max_concurrent,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        # Check risk budget availability (#P3)
        if self._risk_budget_manager:
            # Find the primary strategy contributing to this signal
            best_strategy = max(
                agg.signals.keys(),
                key=lambda s: self._weights.get(s, 0) * agg.signals[s].confidence
            )

            budget = self._risk_budget_manager.get_budget(best_strategy)
            if budget and budget.is_frozen:
                rejection_reason = f"Strategy {best_strategy} is FROZEN: {budget.freeze_reason}"
                logger.warning(f"CIO: {rejection_reason}, skipping decision for {symbol}")
                # Audit log rejected decision (COMPLIANCE REQUIREMENT)
                self._audit_logger.log_agent_event(
                    agent_name=self.name,
                    event_type="decision_rejected",
                    details={
                        "symbol": symbol,
                        "rejection_reason": rejection_reason,
                        "rejection_code": "STRATEGY_FROZEN",
                        "strategy": best_strategy,
                        "freeze_reason": budget.freeze_reason,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                return

            available_budget = self._risk_budget_manager.get_available_budget(best_strategy)
            if available_budget <= 0:
                rejection_reason = f"Strategy {best_strategy} has no available risk budget"
                logger.warning(f"CIO: {rejection_reason}, skipping decision for {symbol}")
                # Audit log rejected decision (COMPLIANCE REQUIREMENT)
                self._audit_logger.log_agent_event(
                    agent_name=self.name,
                    event_type="decision_rejected",
                    details={
                        "symbol": symbol,
                        "rejection_reason": rejection_reason,
                        "rejection_code": "NO_RISK_BUDGET",
                        "strategy": best_strategy,
                        "available_budget": available_budget,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                return

            # Check rebalancing triggers (#P4)
            rebalance_events = self._risk_budget_manager.check_rebalance_triggers()
            if rebalance_events:
                for event in rebalance_events:
                    logger.info(
                        f"CIO: Rebalance triggered ({event.trigger.value}): {event.reason}"
                    )
                # Update strategy performance weights based on new allocations
                for strategy, new_alloc in rebalance_events[-1].new_allocations.items():
                    if strategy in self._base_weights:
                        # Adjust base weights to reflect new risk allocation
                        self._base_weights[strategy] = new_alloc
                self._update_dynamic_weights()

        # Determine action
        if agg.consensus_direction == SignalDirection.FLAT:
            logger.info(f"CIO: No consensus for {symbol} (FLAT) - skipping decision")
            # Audit log rejected decision (COMPLIANCE REQUIREMENT)
            self._audit_logger.log_agent_event(
                agent_name=self.name,
                event_type="decision_rejected",
                details={
                    "symbol": symbol,
                    "rejection_reason": "No consensus direction (FLAT)",
                    "rejection_code": "NO_CONSENSUS_DIRECTION",
                    "consensus_direction": agg.consensus_direction.value,
                    "weighted_strength": agg.weighted_strength,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        # Calculate position size using Kelly criterion and market data
        quantity = self._calculate_position_size(agg)

        if quantity == 0:
            logger.info(f"CIO: Position size=0 for {symbol} (conf={agg.weighted_confidence:.2f}, str={agg.weighted_strength:.2f}) - skipping")
            # Audit log rejected decision (COMPLIANCE REQUIREMENT)
            self._audit_logger.log_agent_event(
                agent_name=self.name,
                event_type="decision_rejected",
                details={
                    "symbol": symbol,
                    "rejection_reason": "Position size calculated as zero",
                    "rejection_code": "ZERO_POSITION_SIZE",
                    "weighted_confidence": agg.weighted_confidence,
                    "weighted_strength": agg.weighted_strength,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        # PM-01: Check sector concentration limits
        sector_allowed, sector_rejection = self._check_sector_concentration(symbol, quantity)
        if not sector_allowed:
            logger.warning(f"CIO: {sector_rejection}, rejecting decision for {symbol}")
            self._audit_logger.log_agent_event(
                agent_name=self.name,
                event_type="decision_rejected",
                details={
                    "symbol": symbol,
                    "rejection_reason": sector_rejection,
                    "rejection_code": "SECTOR_CONCENTRATION_EXCEEDED",
                    "proposed_quantity": quantity,
                    "sector": self._symbol_to_sector.get(symbol, "Unknown"),
                    "max_sector_concentration": self._max_sector_concentration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            return

        # Extract best stop_loss and target_price from contributing signals
        # Use the stop_loss from the signal with highest confidence
        best_stop_loss = None
        best_target_price = None
        best_signal_confidence = 0.0
        for signal in agg.signals.values():
            if signal.confidence > best_signal_confidence:
                if signal.stop_loss:
                    best_stop_loss = signal.stop_loss
                    best_signal_confidence = signal.confidence
                if signal.target_price:
                    best_target_price = signal.target_price

        # Create decision event
        decision = DecisionEvent(
            source_agent=self.name,
            symbol=symbol,
            action=OrderSide.BUY if agg.consensus_direction == SignalDirection.LONG else OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.LIMIT,  # Default to limit orders
            limit_price=None,  # Will be set by execution
            stop_price=best_stop_loss,  # Pass stop_loss from signals for automatic stop placement
            rationale=self._build_rationale(agg),
            contributing_signals=tuple(s.event_id for s in agg.signals.values()),
            data_sources=self._collect_data_sources(agg),
            conviction_score=agg.weighted_confidence,
        )

        # Log decision (COMPLIANCE REQUIREMENT)
        self._audit_logger.log_decision(
            agent_name=self.name,
            decision_id=decision.event_id,
            symbol=symbol,
            action=decision.action.value if decision.action else "none",
            quantity=quantity,
            rationale=decision.rationale,
            data_sources=list(decision.data_sources),
            contributing_signals=list(decision.contributing_signals),
            conviction_score=decision.conviction_score,
        )

        # Publish decision
        await self._event_bus.publish(decision)

        # Track active decisions (will be cleared when validated decision comes back)
        self._active_decisions[decision.event_id] = datetime.now(timezone.utc)
        self._last_decision_time_per_symbol[symbol] = datetime.now(timezone.utc)

        # P2: Record decision for accuracy tracking
        self._record_decision(
            decision_id=decision.event_id,
            symbol=symbol,
            direction=agg.consensus_direction,
            quantity=quantity,
            conviction_score=agg.weighted_confidence,
            contributing_strategies=list(agg.signals.keys()),
        )

        logger.info(
            f"CIO DECISION: {decision.action.value if decision.action else 'none'} "
            f"{quantity} {symbol} (conviction={agg.weighted_confidence:.2f})"
        )

    def _aggregate_signals(self, agg: SignalAggregation) -> None:
        """
        Aggregate signals with dynamic weights and correlation adjustment (#Q5).

        Features:
        - Regime-dependent weight adjustment
        - Performance-weighted signals
        - Signal correlation adjustment (NEW)
        - P2: Signal confidence weighting (filter low confidence, apply non-linear weighting)
        - Phase 2: Signal quality scoring
        """
        # Update dynamic weights if enabled
        if self._use_dynamic_weights:
            self._update_dynamic_weights()

        # Record signals to history for correlation tracking (#Q5)
        self._record_signals_to_history(agg.signals)

        # P2: Filter out low-confidence signals
        filtered_signals = {
            agent: signal for agent, signal in agg.signals.items()
            if signal.confidence >= self._min_signal_confidence
        }

        if not filtered_signals:
            # If all signals filtered out, use original but log warning
            logger.warning(
                f"All signals for {agg.symbol} below confidence threshold "
                f"({self._min_signal_confidence}), using unfiltered signals"
            )
            filtered_signals = agg.signals

        # =====================================================================
        # EMERGENCY MODE: Filter to core strategies only
        # =====================================================================
        effective_mode = self._get_effective_decision_mode()
        if effective_mode == DecisionMode.EMERGENCY:
            mode_filtered_signals = {
                agent: signal for agent, signal in filtered_signals.items()
                if self.is_strategy_allowed_in_current_mode(agent)
            }

            if mode_filtered_signals:
                excluded = set(filtered_signals.keys()) - set(mode_filtered_signals.keys())
                if excluded:
                    logger.warning(
                        f"EMERGENCY MODE: Excluded {len(excluded)} non-core strategies: {excluded}. "
                        f"Using only: {list(mode_filtered_signals.keys())}"
                    )
                filtered_signals = mode_filtered_signals
            else:
                logger.warning(
                    f"EMERGENCY MODE: No core strategies have signals for {agg.symbol}, "
                    f"skipping decision"
                )
                return

        # =====================================================================
        # Phase 2: Signal Quality Scoring
        # =====================================================================
        if self._signal_quality_scorer is not None and self._signal_quality_enabled:
            quality_filtered_signals = {}
            market_data = self._get_market_data_for_quality(agg.symbol)

            for agent_name, signal in filtered_signals.items():
                # Get other signals for confluence check
                other_signals = [s for n, s in filtered_signals.items() if n != agent_name]

                quality_result = self._signal_quality_scorer.validate_signal(
                    signal=signal,
                    market_data=market_data,
                    support_levels=[],  # Could be enhanced with S/R detection
                    resistance_levels=[],
                    other_signals=other_signals,
                )

                if quality_result.is_valid:
                    quality_filtered_signals[agent_name] = signal
                    logger.debug(
                        f"Signal {agent_name} quality OK: {quality_result.total_score:.1f} "
                        f"({quality_result.tier.value})"
                    )
                else:
                    logger.info(
                        f"Signal {agent_name} REJECTED by quality filter: "
                        f"score={quality_result.total_score:.1f}, "
                        f"reasons={quality_result.rejection_reasons}"
                    )

            # Use quality-filtered signals if any remain
            if quality_filtered_signals:
                filtered_signals = quality_filtered_signals
            else:
                logger.warning(
                    f"All signals for {agg.symbol} rejected by quality filter, "
                    f"proceeding with confidence-filtered signals"
                )

        # Phase 2: Get VIX-adjusted weights if VIX data available
        base_weights = self.get_vix_adjusted_weights() if hasattr(self, '_vix_current') and self._vix_current else self._weights

        # Get correlation-adjusted weights (#Q5)
        if self._use_correlation_adjustment and len(filtered_signals) > 1:
            adjusted_weights = self._get_correlation_adjusted_weights(filtered_signals, base_weights)
            agg.correlation_adjusted = True
        else:
            adjusted_weights = {agent: base_weights.get(agent, 0.1) for agent in filtered_signals}
            agg.correlation_adjusted = False

        total_weight = 0.0
        weighted_strength = 0.0
        weighted_confidence = 0.0

        long_votes = 0.0
        short_votes = 0.0

        for agent_name, signal in filtered_signals.items():
            weight = adjusted_weights.get(agent_name, 0.1)

            # P2: Apply confidence weighting with non-linear power
            # Higher power gives more weight to high-confidence signals
            confidence_weight = signal.confidence ** self._confidence_weight_power
            effective_weight = weight * confidence_weight

            total_weight += effective_weight

            # Aggregate strength and confidence
            weighted_strength += signal.strength * effective_weight
            weighted_confidence += signal.confidence * effective_weight

            # Count directional votes (also confidence-weighted)
            if signal.direction == SignalDirection.LONG:
                long_votes += effective_weight
            elif signal.direction == SignalDirection.SHORT:
                short_votes += effective_weight

        if total_weight > 0:
            agg.weighted_strength = weighted_strength / total_weight
            agg.weighted_confidence = weighted_confidence / total_weight

        # Calculate effective signal count (#Q5)
        agg.effective_signal_count = self._calculate_effective_signal_count(filtered_signals)

        # Determine consensus direction
        if long_votes > short_votes and long_votes > total_weight * 0.55:
            agg.consensus_direction = SignalDirection.LONG
        elif short_votes > long_votes and short_votes > total_weight * 0.55:
            agg.consensus_direction = SignalDirection.SHORT
        else:
            agg.consensus_direction = SignalDirection.FLAT

        # DEBUG: Log aggregation details
        logger.info(
            f"CIO AGG {agg.symbol}: LONG={long_votes:.2f} SHORT={short_votes:.2f} "
            f"total={total_weight:.2f} (55%={total_weight*0.55:.2f}) "
            f"-> {agg.consensus_direction.value} conf={agg.weighted_confidence:.2f}"
        )

        # Log to CSV for analysis (final_decision and quantity will be updated later if decision is made)
        self._log_decision_to_csv(
            agg=agg,
            filtered_signals=filtered_signals,
            adjusted_weights=adjusted_weights,
            long_votes=long_votes,
            short_votes=short_votes,
            total_weight=total_weight,
            final_decision="PENDING",
            quantity=0
        )

        agg.regime_adjusted = self._use_dynamic_weights

    def _update_dynamic_weights(self) -> None:
        """
        Update signal weights based on regime and performance.

        Combines:
        1. Base weights
        2. Regime-dependent adjustments
        3. Performance-based adjustments
        """
        # Start with base weights
        new_weights = dict(self._base_weights)

        # Apply regime adjustments
        if self._current_regime in self._regime_weights:
            regime_adj = self._regime_weights[self._current_regime]
            for strategy, multiplier in regime_adj.items():
                if strategy in new_weights:
                    adjustment = (multiplier - 1.0) * self._regime_weight_factor
                    new_weights[strategy] *= (1.0 + adjustment)

        # Apply performance adjustments
        for strategy, perf in self._strategy_performance.items():
            if strategy in new_weights:
                # Use Sharpe ratio to adjust weights
                # Positive Sharpe increases weight, negative decreases
                sharpe_adj = perf.rolling_sharpe * self._performance_weight_factor * 0.1
                sharpe_adj = max(-0.3, min(0.3, sharpe_adj))  # Cap adjustment
                new_weights[strategy] *= (1.0 + sharpe_adj)

        # Normalize weights to sum to 1
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        self._weights = new_weights

    # =========================================================================
    # MARKET DATA CACHE FOR SIGNAL QUALITY SCORING
    # =========================================================================

    def update_market_data_cache(self, symbol: str, price: float, volume: float = 0) -> None:
        """
        Update market data cache for signal quality scoring.

        Call this when market data arrives to maintain quality scoring data.
        """
        if symbol not in self._market_data_cache:
            self._market_data_cache[symbol] = {
                'prices': deque(maxlen=self._market_data_maxlen),
                'volumes': deque(maxlen=self._market_data_maxlen),
            }

        if price > 0:
            self._market_data_cache[symbol]['prices'].append(price)
        if volume > 0:
            self._market_data_cache[symbol]['volumes'].append(volume)

    def _get_market_data_for_quality(self, symbol: str) -> dict:
        """
        Get market data dict for signal quality scoring.

        Returns dict with:
        - prices: list of recent prices
        - volumes: list of recent volumes
        - volatility_regime: current volatility regime
        - adx: ADX indicator value (estimated)
        """
        cache = self._market_data_cache.get(symbol, {})
        prices = list(cache.get('prices', []))
        volumes = list(cache.get('volumes', []))

        # Estimate volatility regime from recent returns
        volatility_regime = 'normal'
        if len(prices) >= 20:
            import numpy as np
            returns = np.diff(prices[-20:]) / np.array(prices[-20:-1])
            vol = np.std(returns) * 100  # Percentage volatility
            if vol > 2.0:
                volatility_regime = 'high'
            elif vol < 0.5:
                volatility_regime = 'low'

        # Estimate ADX from trend strength
        adx = 25.0  # Default neutral ADX
        if len(prices) >= 20:
            import numpy as np
            sma_short = np.mean(prices[-10:])
            sma_long = np.mean(prices[-20:])
            trend_diff = abs(sma_short - sma_long) / sma_long * 100
            adx = min(50.0, 15.0 + trend_diff * 5)

        return {
            'prices': prices,
            'volumes': volumes,
            'volatility_regime': volatility_regime,
            'adx': adx,
        }

    # =========================================================================
    # SIGNAL CORRELATION ADJUSTMENT (#Q5)
    # =========================================================================

    def _record_signals_to_history(self, signals: dict[str, SignalEvent]) -> None:
        """
        Record signals to history for correlation calculation (#Q5).

        Converts signal direction to numeric value for correlation calculation:
        - LONG = +1
        - SHORT = -1
        - FLAT = 0
        """
        now = datetime.now(timezone.utc)

        for agent_name, signal in signals.items():
            if agent_name not in self._signal_history:
                self._signal_history[agent_name] = []

            # Convert direction to numeric value
            direction_val = 0.0
            if signal.direction == SignalDirection.LONG:
                direction_val = signal.strength
            elif signal.direction == SignalDirection.SHORT:
                direction_val = -signal.strength

            self._signal_history[agent_name].append((now, direction_val))

            # Trim history to max length
            if len(self._signal_history[agent_name]) > self._max_signal_history:
                self._signal_history[agent_name] = self._signal_history[agent_name][-self._max_signal_history:]

        # Update correlation matrix periodically
        if sum(len(h) for h in self._signal_history.values()) % 10 == 0:
            self._update_signal_correlations()

    def _update_signal_correlations(self) -> None:
        """
        Update the signal correlation matrix (#Q5).

        Calculates pairwise correlations between agent signals over the lookback period.
        """
        agents = list(self._signal_history.keys())

        for i, agent1 in enumerate(agents):
            for agent2 in agents[i + 1:]:
                corr = self._calculate_signal_correlation(agent1, agent2)
                if corr is not None:
                    self._signal_correlation_matrix[(agent1, agent2)] = corr
                    self._signal_correlation_matrix[(agent2, agent1)] = corr

    def _calculate_signal_correlation(self, agent1: str, agent2: str) -> float | None:
        """
        Calculate correlation between two agents' signals (#Q5).

        Uses the last N signals where both agents provided signals.
        Returns None if insufficient data.

        P1-12: Now properly aligns signals by timestamp instead of index.
        """
        history1 = self._signal_history.get(agent1, [])
        history2 = self._signal_history.get(agent2, [])

        if len(history1) < 10 or len(history2) < 10:
            return None

        # P1-12: Create time-aligned series using timestamp matching
        # Convert to dict by timestamp for efficient lookup
        MAX_TIME_DIFF_SECONDS = 60  # Signals within 60s are considered simultaneous

        dict1 = {ts: val for ts, val in history1}
        dict2 = {ts: val for ts, val in history2}

        values1 = []
        values2 = []

        # Find matching timestamps (within tolerance)
        for ts1, val1 in sorted(dict1.items(), reverse=True):
            # Find closest timestamp in history2
            best_match = None
            best_diff = float('inf')

            for ts2, val2 in dict2.items():
                diff = abs((ts1 - ts2).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_match = (ts2, val2)

            # Only include if timestamps are close enough
            if best_match and best_diff <= MAX_TIME_DIFF_SECONDS:
                values1.append(val1)
                values2.append(best_match[1])

            # Stop when we have enough samples
            if len(values1) >= self._correlation_lookback:
                break

        if len(values1) < 10:
            logger.debug(
                f"Insufficient aligned signals for correlation: "
                f"{agent1}-{agent2} only {len(values1)} matches"
            )
            return None

        # Calculate Pearson correlation
        try:
            import numpy as np

            mean1 = sum(values1) / len(values1)
            mean2 = sum(values2) / len(values2)

            numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2))
            denom1 = sum((v1 - mean1) ** 2 for v1 in values1) ** 0.5
            denom2 = sum((v2 - mean2) ** 2 for v2 in values2) ** 0.5

            if denom1 * denom2 == 0:
                return 0.0

            corr = numerator / (denom1 * denom2)
            return max(-1.0, min(1.0, corr))

        except Exception as e:
            logger.debug(f"Error calculating signal correlation: {e}")
            return None

    def _get_correlation_adjusted_weights(
        self,
        signals: dict[str, SignalEvent],
        base_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Adjust signal weights to account for correlation (#Q5, Phase D enhanced).

        Highly correlated signals should not be double-counted.
        Uses a discount factor based on pairwise correlations.

        The adjustment reduces the effective weight of signals that are
        highly correlated with others, preventing overconfidence from
        redundant information.

        Phase 2: Now accepts optional base_weights parameter for VIX-adjusted weights.
        Phase D Enhancement: Monthly rolling correlation with aggressive halving at >0.95.
        """
        agents = list(signals.keys())
        if base_weights is None:
            base_weights = self._weights
        weight_map = {agent: base_weights.get(agent, 0.1) for agent in agents}

        if len(agents) < 2:
            return weight_map

        # Update monthly correlations if needed
        self._maybe_update_monthly_correlations()

        # Calculate correlation discount for each agent
        adjusted_weights = {}

        for agent in agents:
            # Find maximum correlation with other signals in this set
            # Use monthly correlation if available, fall back to short-term
            max_corr = 0.0
            correlated_agents = []
            halved_due_to_high_corr = False

            for other_agent in agents:
                if other_agent == agent:
                    continue

                # Prefer monthly correlation for stability (Phase D)
                corr = self._monthly_correlations.get((agent, other_agent))
                if corr is None:
                    corr = self._signal_correlation_matrix.get((agent, other_agent))
                if corr is None:
                    continue

                if abs(corr) > 0.5:  # Significant correlation
                    correlated_agents.append((other_agent, corr))
                max_corr = max(max_corr, abs(corr))

            # Phase D: Apply enhanced discount based on correlation
            # Very high correlation (>0.95) halves weight (Phase D improvement)
            # High correlation (>0.8) reduces weight by 50%
            # Moderate-high correlation (0.6-0.8) reduces weight by 25%
            # Moderate correlation (0.5-0.6) reduces weight by 10%
            if max_corr >= self._high_correlation_threshold:  # Default 0.95
                discount = self._high_correlation_weight_factor  # Default 0.5
                halved_due_to_high_corr = True
                self._correlation_halved_count += 1
            elif max_corr > 0.8:
                discount = 0.5  # 50% weight reduction
            elif max_corr > 0.6:
                discount = 0.75  # 25% weight reduction
            elif max_corr > 0.5:
                discount = 0.9  # 10% weight reduction
            else:
                discount = 1.0  # No discount

            adjusted_weights[agent] = weight_map[agent] * discount

            if discount < 1.0 and correlated_agents:
                corr_info = ", ".join([f"{a}:{c:.2f}" for a, c in correlated_agents])
                if halved_due_to_high_corr:
                    logger.warning(
                        f"Signal {agent} weight HALVED due to very high correlation "
                        f"(>={self._high_correlation_threshold}) with [{corr_info}]"
                    )
                else:
                    logger.debug(
                        f"Signal {agent} weight discounted by {(1-discount)*100:.0f}% "
                        f"due to correlation with [{corr_info}]"
                    )

        # Normalize weights to sum to original total
        original_total = sum(weight_map.values())
        adjusted_total = sum(adjusted_weights.values())

        if adjusted_total > 0 and original_total > 0:
            scale = original_total / adjusted_total
            adjusted_weights = {k: v * scale for k, v in adjusted_weights.items()}

        return adjusted_weights

    def _maybe_update_monthly_correlations(self) -> None:
        """
        Update monthly correlations if enough time has passed (Phase D).

        Monthly correlations use a 30-day rolling window (720 hourly observations)
        and are updated hourly for computational efficiency.
        """
        now = datetime.now(timezone.utc)

        # Check if update is needed
        if self._last_monthly_correlation_update is not None:
            hours_since_update = (now - self._last_monthly_correlation_update).total_seconds() / 3600
            if hours_since_update < self._monthly_correlation_update_interval_hours:
                return

        # Update monthly correlations
        agents = list(self._signal_history.keys())

        for i, agent1 in enumerate(agents):
            for agent2 in agents[i + 1:]:
                corr = self._calculate_monthly_correlation(agent1, agent2)
                if corr is not None:
                    self._monthly_correlations[(agent1, agent2)] = corr
                    self._monthly_correlations[(agent2, agent1)] = corr

        self._last_monthly_correlation_update = now

        # Log highly correlated pairs
        high_corr_pairs = [
            (pair, corr) for pair, corr in self._monthly_correlations.items()
            if corr >= self._high_correlation_threshold and pair[0] < pair[1]  # Avoid duplicates
        ]
        if high_corr_pairs:
            for pair, corr in high_corr_pairs:
                logger.info(
                    f"High monthly correlation detected: {pair[0]}-{pair[1]} = {corr:.3f}"
                )

    def _calculate_monthly_correlation(self, agent1: str, agent2: str) -> float | None:
        """
        Calculate monthly rolling correlation between two agents (Phase D).

        Uses a 30-day window (720 hourly observations by default).
        Returns None if insufficient data.
        """
        history1 = self._signal_history.get(agent1, [])
        history2 = self._signal_history.get(agent2, [])

        # Need substantial history for monthly correlation
        if len(history1) < 30 or len(history2) < 30:
            return None

        # Time-align signals (same logic as short-term, but with larger window)
        MAX_TIME_DIFF_SECONDS = 3600  # 1 hour tolerance for hourly correlation

        dict1 = {ts: val for ts, val in history1}
        dict2 = {ts: val for ts, val in history2}

        values1 = []
        values2 = []

        for ts1, val1 in sorted(dict1.items(), reverse=True):
            best_match = None
            best_diff = float('inf')

            for ts2, val2 in dict2.items():
                diff = abs((ts1 - ts2).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_match = (ts2, val2)

            if best_match and best_diff <= MAX_TIME_DIFF_SECONDS:
                values1.append(val1)
                values2.append(best_match[1])

            # Use monthly lookback
            if len(values1) >= self._monthly_correlation_lookback:
                break

        if len(values1) < 30:
            return None

        # Calculate Pearson correlation
        try:
            mean1 = sum(values1) / len(values1)
            mean2 = sum(values2) / len(values2)

            numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2))
            denom1 = sum((v1 - mean1) ** 2 for v1 in values1) ** 0.5
            denom2 = sum((v2 - mean2) ** 2 for v2 in values2) ** 0.5

            if denom1 * denom2 == 0:
                return 0.0

            corr = numerator / (denom1 * denom2)
            return max(-1.0, min(1.0, corr))

        except Exception as e:
            logger.debug(f"Error calculating monthly correlation: {e}")
            return None

    def get_correlation_statistics(self) -> dict:
        """Get correlation adjustment statistics (Phase D)."""
        return {
            "short_term_correlations": len(self._signal_correlation_matrix),
            "monthly_correlations": len(self._monthly_correlations),
            "correlation_halved_count": self._correlation_halved_count,
            "high_correlation_threshold": self._high_correlation_threshold,
            "high_correlation_weight_factor": self._high_correlation_weight_factor,
            "last_monthly_update": (
                self._last_monthly_correlation_update.isoformat()
                if self._last_monthly_correlation_update else None
            ),
            "high_correlation_pairs": [
                {"agents": list(pair), "correlation": round(corr, 3)}
                for pair, corr in self._monthly_correlations.items()
                if corr >= self._high_correlation_threshold and pair[0] < pair[1]
            ],
        }

    def _calculate_effective_signal_count(self, signals: dict[str, SignalEvent]) -> float:
        """
        Calculate effective number of independent signals (#Q5).

        Similar to effective N in portfolio diversification.
        If all signals are independent: effective_n = n
        If all signals are perfectly correlated: effective_n = 1

        Formula: effective_n = (sum of weights)^2 / (sum of weights^2 adjusted for correlation)
        """
        agents = list(signals.keys())
        n = len(agents)

        if n <= 1:
            return float(n)

        # Build correlation matrix for these agents
        weights = [self._weights.get(agent, 0.1) for agent in agents]

        # Calculate weighted sum considering correlations
        # effective_n = 1 / sum(wi * wj * corr_ij)
        weighted_sum_sq = 0.0

        for i, agent_i in enumerate(agents):
            for j, agent_j in enumerate(agents):
                w_i = weights[i]
                w_j = weights[j]

                if i == j:
                    corr = 1.0
                else:
                    corr = self._signal_correlation_matrix.get((agent_i, agent_j), 0.0)

                weighted_sum_sq += w_i * w_j * corr

        total_weight = sum(weights)

        if weighted_sum_sq > 0 and total_weight > 0:
            # Normalize
            weighted_sum_sq /= (total_weight ** 2)
            effective_n = 1.0 / max(weighted_sum_sq, 0.01)
            # Cap at actual number of signals
            return min(effective_n, float(n))

        return float(n)

    def get_signal_correlations(self) -> dict[str, Any]:
        """
        Get signal correlation information for monitoring (#Q5).

        Returns:
            Dictionary with correlation matrix and statistics
        """
        return {
            "correlation_matrix": {
                f"{k[0]}-{k[1]}": v
                for k, v in self._signal_correlation_matrix.items()
                if k[0] < k[1]  # Only upper triangle
            },
            "signal_history_lengths": {
                agent: len(history)
                for agent, history in self._signal_history.items()
            },
            "highly_correlated_pairs": [
                (k[0], k[1], v)
                for k, v in self._signal_correlation_matrix.items()
                if k[0] < k[1] and abs(v) > 0.7
            ],
        }

    def _calculate_position_size(self, agg: SignalAggregation) -> int:
        """
        Calculate position size using Kelly criterion or conviction-based sizing.

        Methods:
        - Kelly criterion (f* = (bp - q) / b)
        - Conviction scaling
        - Correlation adjustment
        - Max position limits
        """
        if self._use_kelly_sizing and self._position_sizer:
            return self._calculate_kelly_size(agg)
        else:
            return self._calculate_conviction_size(agg)

    def _calculate_kelly_size(self, agg: SignalAggregation) -> int:
        """
        Calculate position size using Kelly criterion with IMPROVED money management.

        IMPROVEMENTS for better risk control:
        1. Quarter-Kelly (0.25x) instead of half-Kelly for safety
        2. Raw Kelly capped at 15% BEFORE applying fraction (was 25%)
        3. Maximum position capped at 2.5% of portfolio (was unlimited)
        4. Maximum total exposure limited to 50% of portfolio
        5. More aggressive drawdown reduction (starts at 3% drawdown)
        6. Additional cap: max 1% daily loss per position

        Kelly formula: f* = (bp - q) / b
        Where:
            b = avg_win / avg_loss (win/loss ratio)
            p = win probability
            q = 1 - p
            f* = optimal fraction of capital to risk

        Args:
            agg: Aggregated signal data with weighted confidence and direction

        Returns:
            Position size in shares (integer), or 0 if conditions not met
        """
        # Get strategy with highest contribution to this signal
        best_strategy = max(
            agg.signals.keys(),
            key=lambda s: self._weights.get(s, 0) * agg.signals[s].confidence
        )

        # Get strategy stats
        perf = self._strategy_performance.get(best_strategy)
        if not perf:
            return self._calculate_conviction_size(agg)

        # Require minimum trades for statistical significance
        MIN_TRADES_FOR_KELLY = 50
        WARN_TRADES_THRESHOLD = 100

        if perf.total_trades < MIN_TRADES_FOR_KELLY:
            logger.info(
                f"Kelly: Insufficient trades for {best_strategy} "
                f"({perf.total_trades}/{MIN_TRADES_FOR_KELLY}), using conviction sizing"
            )
            return self._calculate_conviction_size(agg)

        if perf.total_trades < WARN_TRADES_THRESHOLD:
            logger.warning(
                f"Kelly: Low sample size for {best_strategy} ({perf.total_trades} trades). "
                f"Position sizing may be unreliable."
            )

        # Kelly inputs from actual tracked data
        win_rate = perf.win_rate
        avg_win = perf.avg_win    # Dollar amounts from trade attribution
        avg_loss = perf.avg_loss  # Dollar amounts from trade attribution

        # Validate inputs
        if win_rate <= 0 or win_rate >= 1:
            return self._calculate_conviction_size(agg)
        if avg_win <= 0 or avg_loss <= 0:
            return self._calculate_conviction_size(agg)

        # Step 1: Calculate raw Kelly
        # BUGFIX: Kelly formula expects return RATIOS, not dollar amounts.
        # avg_win/avg_loss are in dollars from attribution - the ratio b = avg_win/avg_loss
        # is dimensionless and correct regardless of units (dollars cancel out).
        # However, the Kelly fraction f* = (bp - q)/b gives the fraction of bankroll to bet,
        # which is correct as long as b is the ratio of average win to average loss.
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        kelly_fraction = (b * p - q) / b if b > 0 else 0

        # Step 2: Ensure non-negative
        kelly_fraction = max(0, kelly_fraction)

        # IMPROVED Step 3: Cap raw Kelly at 15% (was 25%)
        # This prevents outsized positions even with seemingly good edge
        kelly_fraction = min(kelly_fraction, self._max_kelly_raw)
        logger.debug(f"Kelly: Raw fraction capped at {self._max_kelly_raw}: {kelly_fraction:.4f}")

        # IMPROVED Step 4: Apply quarter-Kelly (0.25x) instead of half-Kelly
        kelly_fraction *= self._kelly_fraction  # Now defaults to 0.25

        # Step 5: Sample size discount
        sample_discount = min(1.0, 0.5 + (perf.total_trades / 400))
        kelly_fraction *= sample_discount

        # IMPROVED Step 6: More aggressive drawdown adjustment
        # Start reducing at ANY drawdown, not just above threshold
        if self._portfolio_drawdown > 0.03:  # Start at 3% drawdown
            # Tiered reduction:
            # 3-5% drawdown: reduce to 80%
            # 5-8% drawdown: reduce to 50%
            # 8-10% drawdown: reduce to 20%
            # >10% drawdown: reduce to 10%
            if self._portfolio_drawdown >= 0.10:
                drawdown_multiplier = 0.10
            elif self._portfolio_drawdown >= 0.08:
                drawdown_multiplier = 0.20
            elif self._portfolio_drawdown >= 0.05:
                drawdown_multiplier = 0.50
            else:  # 3-5% drawdown
                drawdown_multiplier = 0.80

            kelly_fraction *= drawdown_multiplier
            logger.info(
                f"Kelly: Drawdown adjustment ({self._portfolio_drawdown:.1%}): "
                f"reduced to {drawdown_multiplier*100:.0f}%"
            )

        # Step 7: Calculate position value
        position_value = self._portfolio_value * kelly_fraction

        # Step 8: Apply conviction adjustment
        conviction_multiplier = agg.weighted_confidence * abs(agg.weighted_strength)
        position_value *= conviction_multiplier

        # Step 9: Apply correlation discount
        if self._correlation_manager:
            correlation_discount = self._get_correlation_discount(agg.symbol)
            position_value *= correlation_discount

        # IMPROVED Step 10: Apply max position % limit (2.5% of portfolio)
        max_position_value = self._portfolio_value * (self._max_position_pct / 100)
        if position_value > max_position_value:
            position_value = max_position_value
            logger.debug(
                f"Kelly: Position capped at {self._max_position_pct}% of portfolio "
                f"(${max_position_value:,.0f})"
            )

        # NEW Step 11: Check total exposure limit (50% of portfolio)
        # Get current gross exposure if broker is available
        current_exposure_pct = self._get_current_exposure_pct()
        remaining_exposure_pct = max(0, self._max_total_exposure_pct - current_exposure_pct)
        max_new_position_value = self._portfolio_value * (remaining_exposure_pct / 100)

        if position_value > max_new_position_value:
            if max_new_position_value <= 0:
                logger.warning(
                    f"Kelly: Total exposure limit reached ({current_exposure_pct:.1f}% >= "
                    f"{self._max_total_exposure_pct:.0f}%), no new positions allowed"
                )
                return 0
            position_value = max_new_position_value
            logger.info(
                f"Kelly: Position limited by total exposure ({current_exposure_pct:.1f}% used, "
                f"{remaining_exposure_pct:.1f}% remaining)"
            )

        # Step 12: Convert to contracts (FIX-01: include futures multiplier)
        estimated_price = self._price_cache.get(agg.symbol)
        if estimated_price is None or estimated_price <= 0:
            logger.warning(f"CIO: No price data for {agg.symbol}, using conviction sizing")
            return self._calculate_conviction_size(agg)

        # FIX-01: Futures multiplier - 1 MES contract = price * 5, 1 MNQ = price * 2
        spec = CONTRACT_SPECS.get(agg.symbol)
        multiplier = spec.multiplier if spec else 1.0
        notional_per_contract = estimated_price * multiplier
        size = int(position_value / notional_per_contract)

        # Apply max contracts limit
        size = min(size, self._max_position_size)

        # Apply decision mode size cap
        mode = self._get_effective_decision_mode()
        mode_cap = POSITION_SIZE_CAPS.get(mode, 1.0)
        if mode_cap < 1.0:
            original_size = size
            size = int(size * mode_cap)
            logger.info(
                f"Kelly: Decision mode {mode.value} cap applied: "
                f"{original_size} -> {size} ({mode_cap*100:.0f}%)"
            )

        # FIX-06: Minimum viable order = 1 for futures
        if size < 1:
            return 0

        logger.info(
            f"Kelly sizing: {agg.symbol} ${position_value:.0f} / "
            f"(${estimated_price:.2f} * {multiplier}) = {size} contracts"
        )
        return size

    def _get_current_exposure_pct(self) -> float:
        """
        Get current gross portfolio exposure as % of portfolio value.

        Returns 0.0 if broker is not available or no positions.
        """
        if not self._broker:
            return 0.0

        try:
            # This is synchronous call - in real implementation would be async
            # For now, use cached sector positions as proxy for exposure
            total_exposure = sum(abs(v) for v in self._sector_positions.values())
            return total_exposure * 100  # Convert to percentage
        except Exception as e:
            logger.debug(f"Could not get current exposure: {e}")
            return 0.0

    def _calculate_conviction_size(self, agg: SignalAggregation) -> int:
        """
        Calculate position size based on conviction (fallback method).

        Used when Kelly criterion cannot be applied (insufficient trade history
        or missing performance statistics). Sizes positions proportionally to
        signal conviction and strength.

        Formula:
            size = base_position_size * conviction * strength * regime_multiplier

        Where:
            conviction = weighted average confidence across contributing signals
            strength = weighted average signal strength (0 to 1)
            regime_multiplier = adjustment based on current market regime

        Args:
            agg: Aggregated signal data with weighted confidence and direction

        Returns:
            Position size in shares (integer), or 0 if below minimum threshold
        """
        conviction_factor = agg.weighted_confidence
        strength_factor = abs(agg.weighted_strength)

        # Get price and futures multiplier
        estimated_price = self._price_cache.get(agg.symbol)
        if estimated_price is None or estimated_price <= 0:
            logger.warning(f"CIO: No price for {agg.symbol} in conviction sizing, returning 0")
            return 0

        spec = CONTRACT_SPECS.get(agg.symbol)
        multiplier = spec.multiplier if spec else 1.0
        notional_per_contract = estimated_price * multiplier

        # Position sizing: use portfolio % allocation, not fixed dollar amount
        # For futures, $5K base is < 1 micro contract ($34K MES notional) -> always 0
        # Instead: max_position_pct of portfolio * conviction * strength / notional
        portfolio = self._portfolio_value if self._portfolio_value > 0 else 500_000.0
        max_dollar_allocation = portfolio * (self._max_position_pct / 100.0)
        dollar_amount = max_dollar_allocation * conviction_factor * strength_factor
        size = max(int(dollar_amount / notional_per_contract), 1 if conviction_factor >= 0.7 else 0)

        # Apply limits
        size = min(size, self._max_position_size)

        # P2: Apply regime-based allocation adjustment
        regime_multiplier = self._regime_allocation_multipliers.get(self._current_regime, 1.0)
        size = int(size * regime_multiplier)

        if regime_multiplier != 1.0:
            logger.debug(
                f"Conviction: Regime allocation adjustment for {self._current_regime.value}: "
                f"multiplier={regime_multiplier:.2f}"
            )

        # Apply decision mode size cap
        mode = self._get_effective_decision_mode()
        mode_cap = POSITION_SIZE_CAPS.get(mode, 1.0)
        if mode_cap < 1.0:
            original_size = size
            size = int(size * mode_cap)
            logger.info(
                f"Conviction: Decision mode {mode.value} cap applied: "
                f"{original_size} -> {size} ({mode_cap*100:.0f}%)"
            )

        # FIX-06: Minimum viable order = 1 for futures
        if size < 1:
            return 0

        logger.info(
            f"Conviction sizing: {agg.symbol} ${dollar_amount:.0f} / "
            f"(${estimated_price:.2f} * {multiplier}) = {size} contracts"
        )
        return size

    def _get_correlation_discount(self, symbol: str) -> float:
        """
        Get correlation-based discount for position sizing.

        Reduces size if highly correlated with existing positions.

        PM-12: Uses stress correlations when in stress mode.
        Stress correlations are typically higher (correlations tend to 1 in crisis),
        resulting in larger position discounts during stressed markets.
        """
        if not self._correlation_manager:
            return 1.0

        # PM-12: Check if we should use stress correlations
        if self._in_stress_mode and self._stress_correlation_cache:
            # Use stress correlations from cache
            max_correlation = 0.0
            for (sym1, sym2), corr in self._stress_correlation_cache.items():
                if sym1 == symbol or sym2 == symbol:
                    max_correlation = max(max_correlation, abs(corr))

            if max_correlation > 0.7:
                # Apply more aggressive discount in stress mode
                # Stress correlations are higher, so discount will be larger
                discount = 1.0 - (max_correlation - 0.7) / 0.6
                discount = max(0.3, discount)  # Floor at 0.3 in stress mode (vs 0.5 normal)
                logger.debug(
                    f"CIO: Stress correlation discount for {symbol}: {discount:.2f} "
                    f"(stress_corr={max_correlation:.2f})"
                )
                return discount
            return 1.0

        # Normal mode: use correlation manager
        highly_correlated = self._correlation_manager.get_highly_correlated_pairs(0.7)

        # Check if symbol is in any highly correlated pair
        max_correlation = 0.0
        for sym1, sym2, corr in highly_correlated:
            if sym1 == symbol or sym2 == symbol:
                max_correlation = max(max_correlation, abs(corr))

        # Apply discount: 1.0 at corr=0.7, 0.5 at corr=1.0
        if max_correlation > 0.7:
            discount = 1.0 - (max_correlation - 0.7) / 0.6
            return max(0.5, discount)

        return 1.0

    def _build_rationale(self, agg: SignalAggregation) -> str:
        """Build decision rationale from contributing signals."""
        parts = [f"CIO Decision for {agg.symbol}:"]
        parts.append(f"Direction: {agg.consensus_direction.value}")
        parts.append(f"Conviction: {agg.weighted_confidence:.2%}")
        parts.append(f"Strength: {agg.weighted_strength:.2f}")
        parts.append("Contributing signals:")

        for agent_name, signal in agg.signals.items():
            weight = self._weights.get(agent_name, 0.1)
            parts.append(
                f"  - {agent_name} ({weight:.0%}): {signal.direction.value}, "
                f"strength={signal.strength:.2f}, confidence={signal.confidence:.2f}"
            )
            if signal.rationale:
                parts.append(f"    Rationale: {signal.rationale[:100]}...")

        return " | ".join(parts)

    def _collect_data_sources(self, agg: SignalAggregation) -> tuple[str, ...]:
        """Collect all data sources from contributing signals."""
        sources = set()
        for signal in agg.signals.values():
            sources.update(signal.data_sources)
        return tuple(sorted(sources))

    # =========================================================================
    # EXTERNAL COMPONENT INTEGRATION
    # =========================================================================

    def set_position_sizer(self, position_sizer) -> None:
        """Set position sizer for Kelly criterion sizing."""
        self._position_sizer = position_sizer
        logger.info("CIO: Position sizer attached")

    def set_attribution(self, attribution) -> None:
        """Set performance attribution for tracking."""
        self._attribution = attribution
        logger.info("CIO: Performance attribution attached")

    def set_correlation_manager(self, correlation_manager) -> None:
        """Set correlation manager for correlation-adjusted sizing."""
        self._correlation_manager = correlation_manager
        logger.info("CIO: Correlation manager attached")

    def set_risk_budget_manager(self, risk_budget_manager) -> None:
        """
        Set cross-strategy risk budget manager (#P3).

        Enables:
        - Risk budget allocation across strategies
        - Position rejection if strategy is over budget
        - Rebalancing trigger monitoring
        """
        self._risk_budget_manager = risk_budget_manager
        logger.info("CIO: Risk budget manager attached")

    def set_broker(self, broker) -> None:
        """Set broker for leverage-aware deleveraging."""
        self._broker = broker
        logger.info("CIO: Broker attached for leverage monitoring")

    # =========================================================================
    # DECISION MODE MANAGEMENT
    # =========================================================================

    def _get_effective_decision_mode(self) -> DecisionMode:
        """
        Get the effective decision mode considering auto-escalation and override.

        Auto-escalation triggers based on portfolio drawdown:
        - >= 5% drawdown: EMERGENCY mode
        - >= 3% drawdown: DEFENSIVE mode
        - < 3% drawdown: NORMAL mode

        Manual override takes precedence over auto-escalation.

        Returns:
            The effective DecisionMode
        """
        # Manual override takes precedence
        if self._decision_mode_override is not None:
            return self._decision_mode_override

        # Auto-escalation based on drawdown
        if self._decision_mode_auto_escalation:
            if self._portfolio_drawdown >= self._emergency_mode_drawdown_threshold:
                return DecisionMode.EMERGENCY
            elif self._portfolio_drawdown >= self._defensive_mode_drawdown_threshold:
                return DecisionMode.DEFENSIVE

        return self._decision_mode

    def set_decision_mode(self, mode: DecisionMode, reason: str = "manual") -> None:
        """
        Set the decision mode with audit trail.

        Args:
            mode: The decision mode to set
            reason: Reason for the change
        """
        old_mode = self._decision_mode
        self._decision_mode = mode
        self._mode_change_history.append((datetime.now(timezone.utc), mode, reason))

        logger.warning(
            f"CIO: Decision mode changed from {old_mode.value} to {mode.value} "
            f"(reason: {reason})"
        )

        self._audit_logger.log_event_dict({
            "type": "decision_mode_change",
            "old_mode": old_mode.value,
            "new_mode": mode.value,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def set_decision_mode_override(self, mode: DecisionMode | None, reason: str = "manual") -> None:
        """
        Set a manual override for decision mode.

        Pass None to clear the override and return to auto-escalation.

        Args:
            mode: The mode to force, or None to clear override
            reason: Reason for the override
        """
        old_override = self._decision_mode_override
        self._decision_mode_override = mode

        if mode is not None:
            logger.warning(
                f"CIO: Decision mode OVERRIDE set to {mode.value} "
                f"(previous override: {old_override.value if old_override else 'none'}) "
                f"(reason: {reason})"
            )
        else:
            logger.info(
                f"CIO: Decision mode override CLEARED "
                f"(previous override: {old_override.value if old_override else 'none'}) "
                f"(reason: {reason})"
            )

        self._mode_change_history.append((datetime.now(timezone.utc), mode, f"override: {reason}"))

    def get_decision_mode_info(self) -> dict:
        """
        Get detailed information about current decision mode.

        Returns:
            Dict with mode info, thresholds, and history
        """
        effective_mode = self._get_effective_decision_mode()

        return {
            "current_mode": self._decision_mode.value,
            "effective_mode": effective_mode.value,
            "manual_override": self._decision_mode_override.value if self._decision_mode_override else None,
            "auto_escalation_enabled": self._decision_mode_auto_escalation,
            "portfolio_drawdown": self._portfolio_drawdown,
            "defensive_threshold": self._defensive_mode_drawdown_threshold,
            "emergency_threshold": self._emergency_mode_drawdown_threshold,
            "position_size_cap": POSITION_SIZE_CAPS.get(effective_mode, 1.0),
            "allowed_strategies": (
                list(EMERGENCY_MODE_CORE_STRATEGIES) if effective_mode == DecisionMode.EMERGENCY
                else "all"
            ),
            "recent_changes": [
                {"time": t.isoformat(), "mode": m.value if m else "cleared", "reason": r}
                for t, m, r in list(self._mode_change_history)[-5:]
            ],
        }

    def is_strategy_allowed_in_current_mode(self, strategy_name: str) -> bool:
        """
        Check if a strategy is allowed to contribute signals in current mode.

        In EMERGENCY mode, only core strategies are allowed.

        Args:
            strategy_name: Name of the strategy agent

        Returns:
            True if strategy is allowed
        """
        mode = self._get_effective_decision_mode()

        if mode == DecisionMode.EMERGENCY:
            return strategy_name in EMERGENCY_MODE_CORE_STRATEGIES

        return True

    # =========================================================================
    # HUMAN-IN-THE-LOOP MODE MANAGEMENT
    # =========================================================================

    def enable_human_cio_mode(self, reason: str = "manual") -> None:
        """
        Enable Human-in-the-Loop mode.

        When enabled, all trading decisions require explicit human approval.
        Decisions are queued and expire after timeout if not acted upon.

        Args:
            reason: Why human-in-the-loop mode is being activated
        """
        self._human_cio_enabled = True
        self._human_cio_reason = reason
        self.set_decision_mode_override(DecisionMode.HUMAN_CIO, f"human_cio: {reason}")

        logger.warning(
            f"HUMAN-IN-THE-LOOP MODE ENABLED. Reason: {reason}. "
            f"All decisions now require human approval."
        )

        self._audit_logger.log_event_dict({
            "type": "human_cio_mode_enabled",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def disable_human_cio_mode(self, authorized_by: str) -> None:
        """
        Disable Human-in-the-Loop mode and return to normal operation.

        Args:
            authorized_by: Username of person disabling
        """
        self._human_cio_enabled = False
        old_reason = self._human_cio_reason
        self._human_cio_reason = ""
        self.set_decision_mode_override(None, f"human_cio_disabled by {authorized_by}")

        # Cancel all pending decisions
        pending_count = len(self._pending_human_decisions)
        for decision_id, pending in list(self._pending_human_decisions.items()):
            self._record_human_decision(pending, "expired", authorized_by, "mode_disabled")
        self._pending_human_decisions.clear()

        logger.warning(
            f"HUMAN-IN-THE-LOOP MODE DISABLED by {authorized_by}. "
            f"Previous reason: {old_reason}. Cancelled {pending_count} pending decisions. "
            f"Returning to auto-trading."
        )

        self._audit_logger.log_event_dict({
            "type": "human_cio_mode_disabled",
            "authorized_by": authorized_by,
            "previous_reason": old_reason,
            "cancelled_decisions": pending_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def is_human_cio_mode(self) -> bool:
        """Check if Human-in-the-Loop mode is active."""
        return self._human_cio_enabled or self._get_effective_decision_mode() == DecisionMode.HUMAN_CIO

    def queue_human_decision(
        self,
        symbol: str,
        direction: SignalDirection,
        quantity: int,
        conviction: float,
        rationale: str,
        signals: dict,
        aggregation: Any,
    ) -> str:
        """
        Queue a decision for human approval (thread-safe with bounded queue).

        Args:
            symbol: Trading symbol
            direction: Proposed direction
            quantity: Proposed size
            conviction: Aggregated conviction
            rationale: Why this trade is proposed
            signals: Original signals
            aggregation: SignalAggregation object

        Returns:
            Decision ID for tracking
        """
        import uuid
        decision_id = f"HUMAN-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now(timezone.utc)

        pending = PendingHumanDecision(
            decision_id=decision_id,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            conviction=conviction,
            rationale=rationale,
            signals={k: v.to_audit_dict() if hasattr(v, 'to_audit_dict') else str(v) for k, v in signals.items()},
            aggregation=aggregation,
            created_at=now,
            expires_at=now + timedelta(seconds=self._human_decision_timeout_seconds),
            status="pending",
        )

        with self._human_decision_lock:
            # Issue 4.2: Bounded queue - evict oldest if at max capacity
            while len(self._pending_human_decisions) >= self._max_pending_human_decisions:
                # Find oldest decision
                oldest_id = min(
                    self._pending_human_decisions.keys(),
                    key=lambda k: self._pending_human_decisions[k].created_at
                )
                evicted = self._pending_human_decisions.pop(oldest_id)
                logger.warning(
                    f"Human decision queue full ({self._max_pending_human_decisions}), "
                    f"evicted oldest: {oldest_id} for {evicted.symbol}"
                )
                self._record_human_decision(evicted, "evicted_queue_full", "system", "Queue overflow")

            self._pending_human_decisions[decision_id] = pending

        logger.info(
            f"HUMAN APPROVAL REQUIRED: {decision_id} | {symbol} {direction.value} "
            f"qty={quantity} conviction={conviction:.2f} | Expires in {self._human_decision_timeout_seconds}s"
        )

        # Notify via callback if registered
        if self._human_decision_callback:
            try:
                self._human_decision_callback(pending)
            except Exception as e:
                logger.error(f"Human decision callback failed: {e}")

        return decision_id

    def approve_human_decision(
        self,
        decision_id: str,
        approved_by: str,
        modified_quantity: int | None = None,
        notes: str = "",
    ) -> tuple[bool, str]:
        """
        Approve a pending human decision (thread-safe).

        Args:
            decision_id: The pending decision to approve
            approved_by: Username of person approving
            modified_quantity: Optionally override the proposed quantity
            notes: Additional notes about the decision

        Returns:
            Tuple of (success, message)
        """
        with self._human_decision_lock:
            if decision_id not in self._pending_human_decisions:
                return False, f"Decision {decision_id} not found or already processed"

            pending = self._pending_human_decisions.pop(decision_id)

        # Check if expired
        if datetime.now(timezone.utc) > pending.expires_at:
            self._record_human_decision(pending, "expired", approved_by, "approved_after_expiry")
            return False, f"Decision {decision_id} has expired"

        # Record approval
        final_quantity = modified_quantity if modified_quantity is not None else pending.quantity
        pending.status = "approved"

        self._record_human_decision(
            pending, "approved", approved_by, notes,
            modified_quantity=final_quantity
        )

        logger.warning(
            f"HUMAN DECISION APPROVED: {decision_id} | {pending.symbol} {pending.direction.value} "
            f"qty={final_quantity} (original: {pending.quantity}) | By: {approved_by}"
        )

        return True, f"Decision {decision_id} approved for {pending.symbol} {pending.direction.value} qty={final_quantity}"

    def reject_human_decision(
        self,
        decision_id: str,
        rejected_by: str,
        reason: str = "",
    ) -> tuple[bool, str]:
        """
        Reject a pending human decision (thread-safe).

        Args:
            decision_id: The pending decision to reject
            rejected_by: Username of person rejecting
            reason: Reason for rejection

        Returns:
            Tuple of (success, message)
        """
        with self._human_decision_lock:
            if decision_id not in self._pending_human_decisions:
                return False, f"Decision {decision_id} not found or already processed"

            pending = self._pending_human_decisions.pop(decision_id)
        pending.status = "rejected"

        self._record_human_decision(pending, "rejected", rejected_by, reason)

        logger.warning(
            f"HUMAN DECISION REJECTED: {decision_id} | {pending.symbol} {pending.direction.value} "
            f"qty={pending.quantity} | By: {rejected_by} | Reason: {reason}"
        )

        return True, f"Decision {decision_id} rejected"

    def _record_human_decision(
        self,
        pending: PendingHumanDecision,
        outcome: str,
        by: str,
        notes: str,
        modified_quantity: int | None = None,
    ) -> None:
        """Record human decision to audit history."""
        record = {
            "decision_id": pending.decision_id,
            "symbol": pending.symbol,
            "direction": pending.direction.value,
            "proposed_quantity": pending.quantity,
            "final_quantity": modified_quantity or pending.quantity,
            "conviction": pending.conviction,
            "outcome": outcome,  # approved, rejected, expired
            "by": by,
            "notes": notes,
            "created_at": pending.created_at.isoformat(),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "signals": pending.signals,
        }

        self._human_decision_history.append(record)

        # Mandatory audit logging
        self._audit_logger.log_event_dict({
            "type": "human_decision",
            **record,
        })

    def get_pending_human_decisions(self) -> list[dict]:
        """Get all pending human decisions awaiting approval (thread-safe)."""
        now = datetime.now(timezone.utc)
        result = []

        with self._human_decision_lock:
            for decision_id, pending in self._pending_human_decisions.items():
                remaining_seconds = (pending.expires_at - now).total_seconds()
                result.append({
                    "decision_id": decision_id,
                    "symbol": pending.symbol,
                    "direction": pending.direction.value,
                    "quantity": pending.quantity,
                    "conviction": pending.conviction,
                    "rationale": pending.rationale,
                    "created_at": pending.created_at.isoformat(),
                    "expires_at": pending.expires_at.isoformat(),
                    "remaining_seconds": max(0, remaining_seconds),
                    "expired": remaining_seconds <= 0,
                })

        return result

    def get_human_decision_history(self, limit: int = 50) -> list[dict]:
        """Get recent human decision history."""
        return list(self._human_decision_history)[-limit:]

    def expire_stale_human_decisions(self) -> int:
        """
        Expire any pending decisions that have passed their timeout (thread-safe).

        Called periodically by orchestrator or event loop.

        Returns:
            Number of decisions expired
        """
        now = datetime.now(timezone.utc)
        expired_count = 0

        with self._human_decision_lock:
            for decision_id, pending in list(self._pending_human_decisions.items()):
                if now > pending.expires_at:
                    self._pending_human_decisions.pop(decision_id)
                    self._record_human_decision(pending, "expired", "system", "timeout")
                    expired_count += 1

                    logger.warning(
                        f"HUMAN DECISION EXPIRED: {decision_id} | {pending.symbol} {pending.direction.value} "
                        f"(no response within {self._human_decision_timeout_seconds}s)"
                    )

        return expired_count

    def set_human_decision_callback(self, callback: Any) -> None:
        """
        Set callback to notify external systems of pending decisions.

        The callback receives a PendingHumanDecision when a new decision is queued.
        """
        self._human_decision_callback = callback
        logger.info("Human decision callback registered")

    def set_portfolio_value(self, value: float) -> None:
        """Update portfolio value for position sizing."""
        self._portfolio_value = value

    def update_price(self, symbol: str, price: float) -> None:
        """
        Update price cache for a symbol.

        Called by orchestrator when market data is received.
        Required for accurate Kelly position sizing.
        """
        if price > 0:
            self._price_cache[symbol] = price

    def update_prices(self, prices: dict[str, float]) -> None:
        """
        Bulk update price cache.

        Called by orchestrator with latest market prices.
        """
        for symbol, price in prices.items():
            if price > 0:
                self._price_cache[symbol] = price

    def set_market_regime(self, regime: MarketRegime) -> None:
        """
        Set current market regime.

        This triggers recalculation of dynamic weights.
        """
        if regime != self._current_regime:
            old_regime = self._current_regime
            self._current_regime = regime
            logger.info(f"CIO: Market regime changed from {old_regime.value} to {regime.value}")

            if self._use_dynamic_weights:
                self._update_dynamic_weights()

    # =========================================================================
    # Phase 2: VIX-BASED REGIME SIGNAL WEIGHTS
    # =========================================================================

    def update_regime_from_vix(
        self,
        vix_current: float,
        vix_ma: float | None = None,
    ) -> MarketRegime:
        """
        Update market regime based on VIX levels (Phase 2).

        Automatically detects market regime from VIX data and adjusts
        signal weights accordingly.

        VIX thresholds:
        - VIX < 15: RISK_ON (complacency, but momentum works)
        - VIX 15-20: NEUTRAL (normal conditions)
        - VIX 20-25: TRENDING (moderate uncertainty)
        - VIX 25-35: VOLATILE (high uncertainty)
        - VIX > 35: RISK_OFF (crisis conditions)

        Args:
            vix_current: Current VIX level
            vix_ma: Optional 20-day VIX MA for spike detection

        Returns:
            Detected MarketRegime
        """
        # Determine base regime from VIX level
        if vix_current < 15:
            regime = MarketRegime.RISK_ON
        elif vix_current < 20:
            regime = MarketRegime.NEUTRAL
        elif vix_current < 25:
            regime = MarketRegime.TRENDING
        elif vix_current < 35:
            regime = MarketRegime.VOLATILE
        else:
            regime = MarketRegime.RISK_OFF

        # Check for VIX spike (current >> MA)
        if vix_ma is not None and vix_ma > 0:
            vix_ratio = vix_current / vix_ma
            if vix_ratio >= 1.5 and regime != MarketRegime.RISK_OFF:
                # VIX spike often indicates regime transition
                regime = MarketRegime.VOLATILE
                logger.info(f"CIO: VIX spike detected ({vix_ratio:.2f}x MA), regime -> VOLATILE")

        # Update regime
        self.set_market_regime(regime)

        # Store VIX data for reference
        self._vix_current = vix_current
        self._vix_ma = vix_ma

        return regime

    def get_vix_adjusted_weights(self) -> dict[str, float]:
        """
        Get signal weights adjusted for current VIX level (Phase 2).

        Additional VIX-based adjustments beyond standard regime weights:
        - High VIX (>25): Increase macro/sentiment weights (flight-to-quality signals)
        - Low VIX (<15): Increase momentum weights (trend continuation)
        - VIX spike: Increase options vol weights (volatility opportunities)

        Returns:
            Dictionary of VIX-adjusted weights
        """
        if not hasattr(self, '_vix_current') or self._vix_current is None:
            return dict(self._weights)

        vix = self._vix_current
        weights = dict(self._weights)

        # Additional VIX-specific adjustments
        if vix > 30:
            # Crisis: boost macro, reduce momentum
            if "MacroAgent" in weights:
                weights["MacroAgent"] *= 1.3
            if "MomentumAgent" in weights:
                weights["MomentumAgent"] *= 0.6
            if "SentimentAgent" in weights:
                weights["SentimentAgent"] *= 1.2  # Contrarian signals valuable
            if "MACDvAgent" in weights:
                weights["MACDvAgent"] *= 1.4  # Vol opportunities

        elif vix > 25:
            # Elevated: balanced caution
            if "MACDvAgent" in weights:
                weights["MACDvAgent"] *= 1.3
            if "MacroAgent" in weights:
                weights["MacroAgent"] *= 1.1
            if "MomentumAgent" in weights:
                weights["MomentumAgent"] *= 0.8

        elif vix < 12:
            # Complacency: favor momentum but add caution
            if "MomentumAgent" in weights:
                weights["MomentumAgent"] *= 1.2
            if "MacroAgent" in weights:
                weights["MacroAgent"] *= 1.1  # Watch for regime change

        # VIX spike adjustment
        if hasattr(self, '_vix_ma') and self._vix_ma is not None and self._vix_ma > 0:
            vix_ratio = vix / self._vix_ma
            if vix_ratio >= 1.3:
                # Spike: increase contrarian/vol signals
                if "SentimentAgent" in weights:
                    weights["SentimentAgent"] *= (1.0 + (vix_ratio - 1.3) * 0.5)
                if "MACDvAgent" in weights:
                    weights["MACDvAgent"] *= (1.0 + (vix_ratio - 1.3) * 0.3)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def get_regime_allocation_multiplier(self) -> float:
        """
        Get position size multiplier based on current regime (Phase 2).

        During high volatility/risk-off regimes, reduce overall position sizing.

        Returns:
            Multiplier for position sizes (0.5 to 1.2)
        """
        base_mult = self._regime_allocation_multipliers.get(
            self._current_regime,
            1.0
        )

        # Additional VIX-based adjustment
        if hasattr(self, '_vix_current') and self._vix_current is not None:
            vix = self._vix_current

            if vix > 40:
                # Extreme fear: reduce to minimum
                return min(base_mult, 0.3)
            elif vix > 30:
                # High fear: significant reduction
                return min(base_mult, 0.5)
            elif vix > 25:
                # Elevated: moderate reduction
                return min(base_mult, 0.7)

        return base_mult

    def get_vix_status(self) -> dict[str, Any]:
        """Get VIX tracking status for monitoring."""
        return {
            "current": getattr(self, '_vix_current', None),
            "ma": getattr(self, '_vix_ma', None),
            "regime": self._current_regime.value,
            "allocation_multiplier": self.get_regime_allocation_multiplier(),
            "vix_adjusted_weights": self.get_vix_adjusted_weights(),
        }

    def update_strategy_performance(
        self,
        strategy: str,
        rolling_sharpe: float,
        win_rate: float,
        recent_pnl: float,
        signal_accuracy: float = 0.5,
        avg_win: float = 0.0,
        avg_loss: float = 0.0,
        total_trades: int = 0,
    ) -> None:
        """
        Update performance metrics for a strategy.

        Called by the orchestrator or attribution system.

        Args:
            strategy: Strategy name
            rolling_sharpe: Rolling Sharpe ratio
            win_rate: Probability of winning trade (0-1)
            recent_pnl: Recent P&L in dollars
            signal_accuracy: Signal accuracy rate (0-1)
            avg_win: Average profit on winning trades (dollars)
            avg_loss: Average loss on losing trades (positive dollars)
            total_trades: Total number of trades for statistical significance
        """
        self._strategy_performance[strategy] = StrategyPerformance(
            strategy=strategy,
            rolling_sharpe=rolling_sharpe,
            win_rate=win_rate,
            recent_pnl=recent_pnl,
            signal_accuracy=signal_accuracy,
            last_update=datetime.now(timezone.utc),
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
        )

        # Log Kelly-relevant metrics
        if avg_win > 0 and avg_loss > 0 and total_trades >= 30:
            b = avg_win / avg_loss
            kelly_raw = (b * win_rate - (1 - win_rate)) / b if b > 0 else 0
            logger.debug(
                f"CIO: Updated {strategy} performance: "
                f"sharpe={rolling_sharpe:.2f}, win_rate={win_rate:.1%}, "
                f"avg_win=${avg_win:.2f}, avg_loss=${avg_loss:.2f}, "
                f"Kelly_raw={kelly_raw:.1%}"
            )
        else:
            logger.debug(
                f"CIO: Updated {strategy} performance: "
                f"sharpe={rolling_sharpe:.2f}, win_rate={win_rate:.1%}"
            )

    def get_current_weights(self) -> dict[str, float]:
        """Get current effective signal weights."""
        return dict(self._weights)

    def get_base_weights(self) -> dict[str, float]:
        """Get base (unadjusted) signal weights."""
        return dict(self._base_weights)

    # =========================================================================
    # P2: HISTORICAL DECISION ACCURACY TRACKING
    # =========================================================================

    def _record_decision(
        self,
        decision_id: str,
        symbol: str,
        direction: SignalDirection,
        quantity: int,
        conviction_score: float,
        contributing_strategies: list[str],
    ) -> None:
        """
        Record a decision for accuracy tracking (P2).

        Stores decision details for later outcome recording and accuracy analysis.
        """
        record = DecisionRecord(
            decision_id=decision_id,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            conviction_score=conviction_score,
            timestamp=datetime.now(timezone.utc),
            contributing_strategies=contributing_strategies,
            regime_at_decision=self._current_regime.value,
        )

        self._decision_history.append(record)

        # Trim old history
        if len(self._decision_history) > self._max_decision_history:
            self._decision_history = self._decision_history[-self._max_decision_history:]

        logger.debug(f"CIO: Recorded decision {decision_id[:8]} for accuracy tracking")

    def record_decision_outcome(
        self,
        decision_id: str,
        pnl: float,
        direction_correct: bool,
    ) -> None:
        """
        Record the outcome of a decision (P2).

        Called by the orchestrator or attribution system when a position is closed.

        Args:
            decision_id: The decision ID
            pnl: Realized P&L from the position
            direction_correct: Whether the direction call was correct
        """
        for record in self._decision_history:
            if record.decision_id == decision_id and not record.outcome_recorded:
                record.outcome_pnl = pnl
                record.outcome_direction_correct = direction_correct
                record.outcome_recorded = True

                # Update accuracy metrics
                self._update_decision_accuracy(record)

                logger.debug(
                    f"CIO: Recorded outcome for decision {decision_id[:8]}: "
                    f"pnl=${pnl:.2f}, correct={direction_correct}"
                )
                return

        logger.warning(f"CIO: Decision {decision_id} not found for outcome recording")

    def _update_decision_accuracy(self, record: DecisionRecord) -> None:
        """
        Update accuracy metrics after recording an outcome (P2).
        """
        # Update overall accuracy
        recorded_decisions = [d for d in self._decision_history if d.outcome_recorded]
        self._total_decisions_tracked = len(recorded_decisions)

        if self._total_decisions_tracked > 0:
            correct_count = sum(1 for d in recorded_decisions if d.outcome_direction_correct)
            self._overall_decision_accuracy = correct_count / self._total_decisions_tracked

        # Update accuracy by strategy
        for strategy in record.contributing_strategies:
            strategy_decisions = [
                d for d in recorded_decisions
                if strategy in d.contributing_strategies
            ]
            if strategy_decisions:
                correct = sum(1 for d in strategy_decisions if d.outcome_direction_correct)
                self._decision_accuracy_by_strategy[strategy] = {
                    "accuracy": correct / len(strategy_decisions),
                    "count": len(strategy_decisions),
                }

        # Update accuracy by regime
        regime = record.regime_at_decision
        regime_decisions = [d for d in recorded_decisions if d.regime_at_decision == regime]
        if regime_decisions:
            correct = sum(1 for d in regime_decisions if d.outcome_direction_correct)
            self._decision_accuracy_by_regime[regime] = {
                "accuracy": correct / len(regime_decisions),
                "count": len(regime_decisions),
            }

    def get_decision_accuracy(self) -> dict[str, Any]:
        """
        Get decision accuracy metrics (P2).

        Returns:
            Dictionary with accuracy statistics by overall, strategy, and regime
        """
        return {
            "overall_accuracy": self._overall_decision_accuracy,
            "total_decisions_tracked": self._total_decisions_tracked,
            "by_strategy": dict(self._decision_accuracy_by_strategy),
            "by_regime": dict(self._decision_accuracy_by_regime),
            "recent_decisions": [
                {
                    "decision_id": d.decision_id[:8],
                    "symbol": d.symbol,
                    "direction": d.direction.value,
                    "conviction": d.conviction_score,
                    "outcome_recorded": d.outcome_recorded,
                    "correct": d.outcome_direction_correct,
                    "pnl": d.outcome_pnl,
                }
                for d in self._decision_history[-10:]
            ],
        }

    def get_strategy_accuracy_weight_adjustment(self, strategy: str) -> float:
        """
        Get weight adjustment based on historical accuracy (P2).

        Strategies with higher accuracy get a boost, lower accuracy get penalized.

        Args:
            strategy: Strategy name

        Returns:
            Multiplier for strategy weight (0.5 to 1.5)
        """
        if strategy not in self._decision_accuracy_by_strategy:
            return 1.0

        metrics = self._decision_accuracy_by_strategy[strategy]
        accuracy = metrics.get("accuracy", 0.5)
        count = metrics.get("count", 0)

        # Need minimum sample size for reliable adjustment
        if count < 20:
            return 1.0

        # Baseline is 50% accuracy (random)
        # Scale: 30% accuracy -> 0.6x, 50% -> 1.0x, 70% -> 1.4x
        adjustment = 0.5 + accuracy
        return max(0.5, min(1.5, adjustment))

    def get_status(self) -> dict[str, Any]:
        """Get agent status for monitoring."""
        base_status = super().get_status()

        base_status.update({
            "current_regime": self._current_regime.value,
            "use_dynamic_weights": self._use_dynamic_weights,
            "use_kelly_sizing": self._use_kelly_sizing,
            "current_weights": self._weights,
            "base_weights": self._base_weights,
            "min_conviction": self._min_conviction,
            "active_decisions": len(self._active_decisions),
            "pending_aggregations": len(self._pending_aggregations),
            "portfolio_value": self._portfolio_value,
            # PM-01: Sector concentration tracking
            "sector_concentration": {
                "max_allowed": self._max_sector_concentration,
                "current_exposures": dict(self._sector_positions),
                "symbols_mapped": len(self._symbol_to_sector),
            },
            # PM-06: Portfolio drawdown tracking
            "portfolio_drawdown": {
                "current_drawdown": self._portfolio_drawdown,
                "peak_value": self._portfolio_peak,
                "drawdown_threshold": self._drawdown_kelly_threshold,
                "kelly_floor": self._drawdown_kelly_floor,
                "kelly_adjusted": self._portfolio_drawdown > self._drawdown_kelly_threshold,
            },
            # PM-12: Stress mode tracking
            "stress_mode": {
                "active": self._in_stress_mode,
                "stress_correlations_cached": len(self._stress_correlation_cache),
            },
            "strategy_performance": {
                k: {
                    "sharpe": v.rolling_sharpe,
                    "win_rate": v.win_rate,
                    "avg_win": v.avg_win,
                    "avg_loss": v.avg_loss,
                    "total_trades": v.total_trades,
                    "kelly_eligible": v.total_trades >= 30 and v.avg_win > 0 and v.avg_loss > 0,
                }
                for k, v in self._strategy_performance.items()
            },
            # Signal correlation tracking (#Q5)
            "signal_correlation": {
                "enabled": self._use_correlation_adjustment,
                "history_size": sum(len(h) for h in self._signal_history.values()),
                "correlation_pairs": len(self._signal_correlation_matrix) // 2,  # Each pair counted twice
                "highly_correlated": len([
                    1 for v in self._signal_correlation_matrix.values() if abs(v) > 0.7
                ]) // 2,
            },
            # P2: Historical decision accuracy tracking
            "decision_accuracy": {
                "overall_accuracy": self._overall_decision_accuracy,
                "total_tracked": self._total_decisions_tracked,
                "by_strategy": dict(self._decision_accuracy_by_strategy),
                "by_regime": dict(self._decision_accuracy_by_regime),
            },
            # P2: Signal confidence weighting
            "confidence_weighting": {
                "min_confidence": self._min_signal_confidence,
                "weight_power": self._confidence_weight_power,
            },
            # P2: Regime-based allocation
            "regime_allocation": {
                "current_regime": self._current_regime.value,
                "current_multiplier": self._regime_allocation_multipliers.get(self._current_regime, 1.0),
                "multipliers": {k.value: v for k, v in self._regime_allocation_multipliers.items()},
            },
            # AUTONOMOUS POSITION MANAGEMENT
            "position_management": {
                "enabled": self._position_management_enabled,
                "tracked_positions": len(self._tracked_positions),
                "last_review": self._last_position_review.isoformat() if self._last_position_review else None,
                "review_interval_seconds": self._position_review_interval_seconds,
                "config": self._position_management_config.to_dict(),
                "stats": dict(self._position_management_stats),
                "positions": {
                    symbol: {
                        "pnl_pct": pos.pnl_pct,
                        "holding_hours": pos.holding_hours,
                        "conviction": pos.current_conviction,
                        "is_long": pos.is_long,
                    }
                    for symbol, pos in self._tracked_positions.items()
                },
            },
        })

        return base_status

"""
Execution Optimizer (Phase 7)
=============================

Advanced execution optimization components for algorithmic trading.

Features:
- 7.1 Adaptive TWAP with Volatility Adjustment
- 7.2 Dynamic Slippage Caps
- 7.3 Session-Aware Execution Rules
- 7.4 Smart Algo Selection
- 7.5 Fill Quality Monitoring

MATURITY: ALPHA
---------------
Status: New implementation
- [x] Adaptive TWAP timing
- [x] Dynamic slippage caps
- [x] Session-aware rules
- [x] Smart algo selection
- [x] Fill quality metrics
- [ ] Integration with execution agent
- [ ] Production validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


# =============================================================================
# 7.1 ADAPTIVE TWAP WITH VOLATILITY ADJUSTMENT
# =============================================================================

class VolatilityRegime(Enum):
    """Volatility regime for execution adjustment."""
    LOW = "low"           # Vol < 0.5 * avg -> slower execution
    NORMAL = "normal"     # 0.5 * avg < vol < 1.5 * avg
    HIGH = "high"         # Vol > 1.5 * avg -> faster execution
    EXTREME = "extreme"   # Vol > 2.5 * avg -> immediate/pause


@dataclass
class AdaptiveTWAPConfig:
    """Configuration for adaptive TWAP."""
    base_slices: int = 10
    base_interval_seconds: int = 60
    vol_adjustment_factor: float = 0.3  # How much vol affects timing
    min_slices: int = 3
    max_slices: int = 50
    min_interval_seconds: int = 15
    max_interval_seconds: int = 300


@dataclass
class AdaptiveTWAPPlan:
    """Execution plan for adaptive TWAP."""
    symbol: str
    total_quantity: int
    num_slices: int
    slice_sizes: list[int]
    intervals_seconds: list[float]
    volatility_regime: VolatilityRegime
    estimated_duration_minutes: float
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AdaptiveTWAP:
    """
    Adaptive TWAP with volatility adjustment (Phase 7.1).

    Adjusts slice timing based on current market volatility:
    - Low vol: Slower execution (larger intervals)
    - High vol: Faster execution (smaller intervals)
    - Extreme vol: Pause or rush to complete
    """

    def __init__(self, config: AdaptiveTWAPConfig | None = None):
        """Initialize adaptive TWAP."""
        self._config = config or AdaptiveTWAPConfig()
        self._volatility_history: dict[str, list[float]] = {}

        logger.info(
            f"AdaptiveTWAP initialized: "
            f"base_slices={self._config.base_slices}, "
            f"base_interval={self._config.base_interval_seconds}s"
        )

    def update_volatility(self, symbol: str, volatility: float) -> None:
        """Update volatility observation for symbol."""
        if symbol not in self._volatility_history:
            self._volatility_history[symbol] = []

        self._volatility_history[symbol].append(volatility)
        if len(self._volatility_history[symbol]) > 100:
            self._volatility_history[symbol].pop(0)

    def get_volatility_regime(self, symbol: str, current_vol: float) -> VolatilityRegime:
        """Determine volatility regime relative to historical."""
        history = self._volatility_history.get(symbol, [])

        if len(history) < 10:
            return VolatilityRegime.NORMAL

        avg_vol = np.mean(history)

        if avg_vol == 0:
            return VolatilityRegime.NORMAL

        ratio = current_vol / avg_vol

        if ratio > 2.5:
            return VolatilityRegime.EXTREME
        elif ratio > 1.5:
            return VolatilityRegime.HIGH
        elif ratio < 0.5:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.NORMAL

    def generate_plan(
        self,
        symbol: str,
        quantity: int,
        side: str,
        current_vol: float,
        urgency: float = 0.5,  # 0 = passive, 1 = aggressive
    ) -> AdaptiveTWAPPlan:
        """
        Generate adaptive TWAP execution plan.

        Args:
            symbol: Instrument symbol
            quantity: Total quantity to execute
            side: "BUY" or "SELL"
            current_vol: Current volatility
            urgency: Execution urgency (0-1)

        Returns:
            AdaptiveTWAPPlan with slice schedule
        """
        self.update_volatility(symbol, current_vol)
        regime = self.get_volatility_regime(symbol, current_vol)

        # Base parameters
        base_slices = self._config.base_slices
        base_interval = self._config.base_interval_seconds

        # Adjust for volatility
        if regime == VolatilityRegime.EXTREME:
            # Rush execution or pause based on urgency
            if urgency > 0.7:
                num_slices = self._config.min_slices
                interval = self._config.min_interval_seconds
            else:
                num_slices = self._config.max_slices
                interval = self._config.max_interval_seconds
        elif regime == VolatilityRegime.HIGH:
            # Faster execution to avoid adverse moves
            vol_adj = 1.0 - self._config.vol_adjustment_factor
            num_slices = max(
                self._config.min_slices,
                int(base_slices * vol_adj)
            )
            interval = max(
                self._config.min_interval_seconds,
                int(base_interval * vol_adj)
            )
        elif regime == VolatilityRegime.LOW:
            # Slower execution for better fills
            vol_adj = 1.0 + self._config.vol_adjustment_factor
            num_slices = min(
                self._config.max_slices,
                int(base_slices * vol_adj)
            )
            interval = min(
                self._config.max_interval_seconds,
                int(base_interval * vol_adj)
            )
        else:
            num_slices = base_slices
            interval = base_interval

        # Adjust for urgency
        urgency_adj = 1.0 - urgency * 0.3  # Max 30% reduction
        num_slices = max(
            self._config.min_slices,
            int(num_slices * urgency_adj)
        )
        interval = max(
            self._config.min_interval_seconds,
            int(interval * urgency_adj)
        )

        # Calculate slice sizes (slightly front-loaded for high vol)
        if regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
            # Front-load execution
            weights = np.array([1.2 ** (num_slices - i - 1) for i in range(num_slices)])
        else:
            # Uniform distribution
            weights = np.ones(num_slices)

        weights = weights / weights.sum()
        slice_sizes = [max(1, int(quantity * w)) for w in weights]

        # Adjust for rounding
        total_allocated = sum(slice_sizes)
        if total_allocated < quantity:
            slice_sizes[-1] += quantity - total_allocated
        elif total_allocated > quantity:
            diff = total_allocated - quantity
            for i in range(min(diff, len(slice_sizes))):
                if slice_sizes[i] > 1:
                    slice_sizes[i] -= 1

        # Create interval schedule
        intervals = [float(interval)] * num_slices

        duration = sum(intervals) / 60

        return AdaptiveTWAPPlan(
            symbol=symbol,
            total_quantity=quantity,
            num_slices=num_slices,
            slice_sizes=slice_sizes,
            intervals_seconds=intervals,
            volatility_regime=regime,
            estimated_duration_minutes=duration,
            rationale=(
                f"Adaptive TWAP: {regime.value} vol regime, "
                f"{num_slices} slices, {interval}s intervals, "
                f"urgency={urgency:.1f}"
            ),
        )


# =============================================================================
# 7.2 DYNAMIC SLIPPAGE CAPS
# =============================================================================

@dataclass
class SlippageConfig:
    """Configuration for dynamic slippage caps."""
    base_slippage_bps: float = 10.0  # Base slippage in basis points
    vol_mult: float = 2.0  # Multiplier for volatility
    size_mult: float = 0.5  # Multiplier for size relative to volume
    min_cap_bps: float = 5.0
    max_cap_bps: float = 50.0
    urgency_mult: float = 1.5  # Higher urgency = higher allowed slippage


@dataclass
class SlippageCap:
    """Dynamic slippage cap calculation."""
    symbol: str
    cap_bps: float
    cap_price: float
    components: dict[str, float]  # Breakdown of cap components
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DynamicSlippageCaps:
    """
    Dynamic slippage caps based on market conditions (Phase 7.2).

    Adjusts maximum acceptable slippage based on:
    - Current volatility
    - Order size relative to volume
    - Execution urgency
    - Historical slippage observations
    """

    def __init__(self, config: SlippageConfig | None = None):
        """Initialize dynamic slippage caps."""
        self._config = config or SlippageConfig()
        self._slippage_history: dict[str, list[float]] = {}

        logger.info(
            f"DynamicSlippageCaps initialized: "
            f"base={self._config.base_slippage_bps}bps"
        )

    def record_slippage(self, symbol: str, slippage_bps: float) -> None:
        """Record observed slippage for adaptive capping."""
        if symbol not in self._slippage_history:
            self._slippage_history[symbol] = []

        self._slippage_history[symbol].append(slippage_bps)
        if len(self._slippage_history[symbol]) > 100:
            self._slippage_history[symbol].pop(0)

    def calculate_cap(
        self,
        symbol: str,
        price: float,
        order_size: int,
        avg_volume: float,
        volatility: float,
        urgency: float = 0.5,
    ) -> SlippageCap:
        """
        Calculate dynamic slippage cap.

        Args:
            symbol: Instrument symbol
            price: Current price
            order_size: Order size in shares
            avg_volume: Average daily volume
            volatility: Current volatility (annualized)
            urgency: Execution urgency (0-1)

        Returns:
            SlippageCap with calculated limits
        """
        # Base slippage
        base = self._config.base_slippage_bps

        # Volatility component
        # Convert annualized vol to daily vol (approx)
        daily_vol = volatility / np.sqrt(252)
        vol_component = daily_vol * 10000 * self._config.vol_mult

        # Size component (market impact)
        if avg_volume > 0:
            size_pct = order_size / avg_volume
            size_component = size_pct * 10000 * self._config.size_mult
        else:
            size_component = 10.0  # Default for unknown volume

        # Urgency adjustment
        urgency_adj = 1.0 + (urgency - 0.5) * self._config.urgency_mult

        # Historical adjustment
        history = self._slippage_history.get(symbol, [])
        if len(history) >= 5:
            hist_avg = np.mean(history)
            hist_std = np.std(history)
            hist_component = hist_avg + hist_std
        else:
            hist_component = 0.0

        # Total cap
        total_bps = (base + vol_component + size_component + hist_component) * urgency_adj
        total_bps = max(self._config.min_cap_bps, min(self._config.max_cap_bps, total_bps))

        # Convert to price
        cap_price = price * total_bps / 10000

        return SlippageCap(
            symbol=symbol,
            cap_bps=total_bps,
            cap_price=cap_price,
            components={
                "base": base,
                "volatility": vol_component,
                "size": size_component,
                "historical": hist_component,
                "urgency_adj": urgency_adj,
            },
        )


# =============================================================================
# 7.3 SESSION-AWARE EXECUTION RULES
# =============================================================================

class SessionPhase(Enum):
    """Trading session phases."""
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"  # First 30 min
    MID_DAY = "mid_day"
    MARKET_CLOSE = "market_close"  # Last 30 min
    AFTER_HOURS = "after_hours"
    OVERNIGHT = "overnight"


@dataclass
class SessionRule:
    """Execution rule for session phase."""
    phase: SessionPhase
    allowed: bool
    max_participation_pct: float  # Max % of volume
    preferred_algo: str
    urgency_penalty: float  # Reduces effective urgency
    size_limit_pct: float  # Max position size as % of ADV
    rationale: str


# Default session rules
DEFAULT_SESSION_RULES = {
    SessionPhase.PRE_MARKET: SessionRule(
        phase=SessionPhase.PRE_MARKET,
        allowed=True,
        max_participation_pct=5.0,
        preferred_algo="LIMIT",
        urgency_penalty=0.3,
        size_limit_pct=1.0,
        rationale="Low liquidity, wide spreads",
    ),
    SessionPhase.MARKET_OPEN: SessionRule(
        phase=SessionPhase.MARKET_OPEN,
        allowed=True,
        max_participation_pct=3.0,  # Lower due to volatility
        preferred_algo="TWAP",
        urgency_penalty=0.2,
        size_limit_pct=5.0,
        rationale="High volatility at open, use TWAP",
    ),
    SessionPhase.MID_DAY: SessionRule(
        phase=SessionPhase.MID_DAY,
        allowed=True,
        max_participation_pct=10.0,
        preferred_algo="VWAP",
        urgency_penalty=0.0,
        size_limit_pct=10.0,
        rationale="Optimal liquidity, use VWAP",
    ),
    SessionPhase.MARKET_CLOSE: SessionRule(
        phase=SessionPhase.MARKET_CLOSE,
        allowed=True,
        max_participation_pct=5.0,
        preferred_algo="TWAP",
        urgency_penalty=0.1,
        size_limit_pct=5.0,
        rationale="Closing volatility, limit participation",
    ),
    SessionPhase.AFTER_HOURS: SessionRule(
        phase=SessionPhase.AFTER_HOURS,
        allowed=True,
        max_participation_pct=2.0,
        preferred_algo="LIMIT",
        urgency_penalty=0.4,
        size_limit_pct=0.5,
        rationale="Very low liquidity",
    ),
    SessionPhase.OVERNIGHT: SessionRule(
        phase=SessionPhase.OVERNIGHT,
        allowed=False,
        max_participation_pct=0.0,
        preferred_algo="NONE",
        urgency_penalty=1.0,
        size_limit_pct=0.0,
        rationale="Market closed",
    ),
}


class SessionAwareExecution:
    """
    Session-aware execution rules (Phase 7.3).

    Adjusts execution parameters based on session phase:
    - Open: Lower participation, TWAP preferred
    - Mid-day: Higher participation, VWAP preferred
    - Close: Rush completion, TWAP preferred
    """

    def __init__(self, rules: dict[SessionPhase, SessionRule] | None = None):
        """Initialize session-aware execution."""
        self._rules = rules or DEFAULT_SESSION_RULES

        logger.info("SessionAwareExecution initialized")

    def get_session_phase(
        self,
        current_time: datetime,
        market_open: time = time(9, 30),
        market_close: time = time(16, 0),
    ) -> SessionPhase:
        """
        Determine current session phase.

        Args:
            current_time: Current time (should be in market timezone)
            market_open: Market open time
            market_close: Market close time

        Returns:
            SessionPhase
        """
        current = current_time.time()

        # Pre-market: 4:00 AM - 9:30 AM
        if time(4, 0) <= current < market_open:
            return SessionPhase.PRE_MARKET

        # Market open: First 30 minutes
        open_end = (
            datetime.combine(datetime.today(), market_open) + timedelta(minutes=30)
        ).time()
        if market_open <= current < open_end:
            return SessionPhase.MARKET_OPEN

        # Market close: Last 30 minutes
        close_start = (
            datetime.combine(datetime.today(), market_close) - timedelta(minutes=30)
        ).time()
        if close_start <= current < market_close:
            return SessionPhase.MARKET_CLOSE

        # Mid-day
        if open_end <= current < close_start:
            return SessionPhase.MID_DAY

        # After hours: 4:00 PM - 8:00 PM
        if market_close <= current < time(20, 0):
            return SessionPhase.AFTER_HOURS

        # Overnight
        return SessionPhase.OVERNIGHT

    def get_rule(self, phase: SessionPhase) -> SessionRule:
        """Get rule for session phase."""
        return self._rules.get(phase, DEFAULT_SESSION_RULES[SessionPhase.MID_DAY])

    def apply_rules(
        self,
        current_time: datetime,
        order_size: int,
        avg_daily_volume: float,
        base_urgency: float,
    ) -> dict[str, Any]:
        """
        Apply session rules to execution parameters.

        Args:
            current_time: Current time
            order_size: Planned order size
            avg_daily_volume: Average daily volume
            base_urgency: Base urgency (0-1)

        Returns:
            Adjusted execution parameters
        """
        phase = self.get_session_phase(current_time)
        rule = self.get_rule(phase)

        # Check if execution allowed
        if not rule.allowed:
            return {
                "allowed": False,
                "phase": phase,
                "reason": rule.rationale,
            }

        # Adjust participation
        max_size = avg_daily_volume * (rule.max_participation_pct / 100)
        adjusted_size = min(order_size, int(max_size))

        # Adjust urgency
        adjusted_urgency = max(0.0, base_urgency - rule.urgency_penalty)

        # Size limit check
        size_limit = avg_daily_volume * (rule.size_limit_pct / 100)
        if order_size > size_limit:
            adjusted_size = int(size_limit)
            oversized = True
        else:
            oversized = False

        return {
            "allowed": True,
            "phase": phase,
            "preferred_algo": rule.preferred_algo,
            "adjusted_size": adjusted_size,
            "adjusted_urgency": adjusted_urgency,
            "max_participation_pct": rule.max_participation_pct,
            "oversized": oversized,
            "rationale": rule.rationale,
        }


# =============================================================================
# 7.4 SMART ALGO SELECTION
# =============================================================================

class AlgoType(Enum):
    """Available execution algorithms."""
    MARKET = "MARKET"        # Immediate execution
    LIMIT = "LIMIT"          # Passive limit order
    TWAP = "TWAP"            # Time-weighted average
    VWAP = "VWAP"            # Volume-weighted average
    ICEBERG = "ICEBERG"      # Hidden size
    MIDPOINT_PEG = "MIDPEG"  # Midpoint peg


@dataclass
class AlgoRecommendation:
    """Algorithm recommendation."""
    algo: AlgoType
    confidence: float  # 0.0 to 1.0
    parameters: dict[str, Any]
    rationale: str
    alternatives: list[tuple[AlgoType, float]]  # [(algo, confidence), ...]


class SmartAlgoSelector:
    """
    Smart algorithm selection (Phase 7.4).

    Selects optimal execution algorithm based on:
    - Order characteristics (size, side, urgency)
    - Market conditions (volatility, spread, liquidity)
    - Session timing
    - Historical performance
    """

    def __init__(self):
        """Initialize smart algo selector."""
        self._algo_performance: dict[str, list[float]] = {}

        logger.info("SmartAlgoSelector initialized")

    def record_performance(self, algo: str, slippage_bps: float) -> None:
        """Record algorithm performance."""
        if algo not in self._algo_performance:
            self._algo_performance[algo] = []

        self._algo_performance[algo].append(slippage_bps)
        if len(self._algo_performance[algo]) > 50:
            self._algo_performance[algo].pop(0)

    def select_algo(
        self,
        order_size: int,
        side: str,
        price: float,
        spread_bps: float,
        volatility: float,
        avg_volume: float,
        urgency: float,
        session_phase: SessionPhase,
    ) -> AlgoRecommendation:
        """
        Select optimal execution algorithm.

        Args:
            order_size: Order size
            side: "BUY" or "SELL"
            price: Current price
            spread_bps: Current spread in basis points
            volatility: Current volatility
            avg_volume: Average daily volume
            urgency: Execution urgency (0-1)
            session_phase: Current session phase

        Returns:
            AlgoRecommendation with selected algorithm
        """
        scores: dict[AlgoType, float] = {}

        # Calculate order size relative to volume
        size_pct = (order_size / avg_volume * 100) if avg_volume > 0 else 10.0

        # MARKET: Good for small, urgent orders
        market_score = 0.3
        if urgency > 0.8:
            market_score += 0.4
        if size_pct < 0.1:
            market_score += 0.2
        if spread_bps > 20:
            market_score -= 0.3
        scores[AlgoType.MARKET] = max(0, market_score)

        # LIMIT: Good for low urgency, tight spreads
        limit_score = 0.3
        if urgency < 0.3:
            limit_score += 0.3
        if spread_bps < 5:
            limit_score += 0.2
        if size_pct < 1.0:
            limit_score += 0.1
        scores[AlgoType.LIMIT] = limit_score

        # TWAP: Good for medium-large orders, all conditions
        twap_score = 0.5
        if 0.5 <= size_pct <= 5.0:
            twap_score += 0.3
        if session_phase in [SessionPhase.MARKET_OPEN, SessionPhase.MARKET_CLOSE]:
            twap_score += 0.2
        if volatility > 0.3:  # High vol
            twap_score += 0.1
        scores[AlgoType.TWAP] = twap_score

        # VWAP: Good for tracking volume profile
        vwap_score = 0.5
        if 1.0 <= size_pct <= 10.0:
            vwap_score += 0.3
        if session_phase == SessionPhase.MID_DAY:
            vwap_score += 0.2
        if urgency < 0.7:
            vwap_score += 0.1
        scores[AlgoType.VWAP] = vwap_score

        # ICEBERG: Good for large orders
        iceberg_score = 0.2
        if size_pct > 5.0:
            iceberg_score += 0.4
        if spread_bps > 10:
            iceberg_score += 0.1
        scores[AlgoType.ICEBERG] = iceberg_score

        # MIDPOINT_PEG: Good for tight spreads, patient execution
        midpeg_score = 0.2
        if spread_bps > 3 and spread_bps < 20:
            midpeg_score += 0.3
        if urgency < 0.4:
            midpeg_score += 0.2
        scores[AlgoType.MIDPOINT_PEG] = midpeg_score

        # Historical performance adjustment
        for algo in scores:
            perf = self._algo_performance.get(algo.value, [])
            if len(perf) >= 10:
                avg_slip = np.mean(perf)
                # Lower slippage = higher score
                scores[algo] -= avg_slip / 100  # Normalize

        # Select best algo
        sorted_algos = sorted(scores.items(), key=lambda x: -x[1])
        best_algo, best_score = sorted_algos[0]
        alternatives = [(algo, score) for algo, score in sorted_algos[1:3]]

        # Determine parameters
        if best_algo == AlgoType.TWAP:
            params = {
                "slices": max(5, min(20, int(size_pct * 3))),
                "interval_seconds": 60 if urgency > 0.5 else 120,
            }
        elif best_algo == AlgoType.VWAP:
            params = {
                "participation_rate": min(20, max(5, int(10 - urgency * 5))),
            }
        elif best_algo == AlgoType.ICEBERG:
            params = {
                "display_size": max(100, order_size // 10),
            }
        else:
            params = {}

        return AlgoRecommendation(
            algo=best_algo,
            confidence=min(1.0, best_score),
            parameters=params,
            rationale=(
                f"Selected {best_algo.value}: "
                f"size={size_pct:.1f}% ADV, spread={spread_bps:.1f}bps, "
                f"urgency={urgency:.1f}, session={session_phase.value}"
            ),
            alternatives=alternatives,
        )


# =============================================================================
# 7.5 FILL QUALITY MONITORING
# =============================================================================

@dataclass
class FillQualityMetrics:
    """Fill quality metrics for an order."""
    symbol: str
    algo: str
    side: str
    quantity: int
    avg_fill_price: float
    arrival_price: float
    vwap: float
    slippage_vs_arrival_bps: float
    slippage_vs_vwap_bps: float
    implementation_shortfall_bps: float
    fill_rate: float  # 0.0 to 1.0
    execution_time_seconds: float
    num_fills: int
    spread_capture_pct: float  # % of spread captured (positive = good)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FillQualityMonitor:
    """
    Fill quality monitoring (Phase 7.5).

    Tracks and analyzes execution quality metrics:
    - Slippage vs arrival price
    - Slippage vs VWAP
    - Implementation shortfall
    - Fill rate
    - Execution time
    """

    def __init__(self):
        """Initialize fill quality monitor."""
        self._metrics_history: list[FillQualityMetrics] = []
        self._symbol_metrics: dict[str, list[FillQualityMetrics]] = {}
        self._algo_metrics: dict[str, list[FillQualityMetrics]] = {}

        logger.info("FillQualityMonitor initialized")

    def calculate_metrics(
        self,
        symbol: str,
        algo: str,
        side: str,
        quantity: int,
        fills: list[tuple[float, int]],  # [(price, qty), ...]
        arrival_price: float,
        vwap: float,
        execution_start: datetime,
        execution_end: datetime,
        bid_at_arrival: float | None = None,
        ask_at_arrival: float | None = None,
    ) -> FillQualityMetrics:
        """
        Calculate fill quality metrics.

        Args:
            symbol: Instrument symbol
            algo: Algorithm used
            side: "BUY" or "SELL"
            quantity: Target quantity
            fills: List of (price, quantity) tuples
            arrival_price: Price at order arrival
            vwap: Market VWAP during execution
            execution_start: Start time
            execution_end: End time
            bid_at_arrival: Bid price at arrival (optional)
            ask_at_arrival: Ask price at arrival (optional)

        Returns:
            FillQualityMetrics
        """
        if not fills:
            return FillQualityMetrics(
                symbol=symbol,
                algo=algo,
                side=side,
                quantity=quantity,
                avg_fill_price=0.0,
                arrival_price=arrival_price,
                vwap=vwap,
                slippage_vs_arrival_bps=0.0,
                slippage_vs_vwap_bps=0.0,
                implementation_shortfall_bps=0.0,
                fill_rate=0.0,
                execution_time_seconds=0.0,
                num_fills=0,
                spread_capture_pct=0.0,
            )

        # Calculate average fill price
        total_cost = sum(price * qty for price, qty in fills)
        filled_qty = sum(qty for _, qty in fills)
        avg_fill_price = total_cost / filled_qty if filled_qty > 0 else 0.0

        # Slippage calculations (positive = bad for buyer)
        if side == "BUY":
            slippage_arrival = (avg_fill_price - arrival_price) / arrival_price * 10000
            slippage_vwap = (avg_fill_price - vwap) / vwap * 10000
        else:  # SELL
            slippage_arrival = (arrival_price - avg_fill_price) / arrival_price * 10000
            slippage_vwap = (vwap - avg_fill_price) / vwap * 10000

        # Implementation shortfall (vs decision price = arrival)
        impl_shortfall = slippage_arrival

        # Fill rate
        fill_rate = filled_qty / quantity if quantity > 0 else 0.0

        # Execution time
        exec_time = (execution_end - execution_start).total_seconds()

        # Spread capture
        spread_capture = 0.0
        if bid_at_arrival is not None and ask_at_arrival is not None:
            spread = ask_at_arrival - bid_at_arrival
            midpoint = (bid_at_arrival + ask_at_arrival) / 2
            if spread > 0:
                if side == "BUY":
                    # Good if we paid less than ask
                    spread_capture = (ask_at_arrival - avg_fill_price) / spread * 100
                else:  # SELL
                    # Good if we sold higher than bid
                    spread_capture = (avg_fill_price - bid_at_arrival) / spread * 100

        metrics = FillQualityMetrics(
            symbol=symbol,
            algo=algo,
            side=side,
            quantity=quantity,
            avg_fill_price=avg_fill_price,
            arrival_price=arrival_price,
            vwap=vwap,
            slippage_vs_arrival_bps=slippage_arrival,
            slippage_vs_vwap_bps=slippage_vwap,
            implementation_shortfall_bps=impl_shortfall,
            fill_rate=fill_rate,
            execution_time_seconds=exec_time,
            num_fills=len(fills),
            spread_capture_pct=spread_capture,
        )

        # Store metrics
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 1000:
            self._metrics_history.pop(0)

        if symbol not in self._symbol_metrics:
            self._symbol_metrics[symbol] = []
        self._symbol_metrics[symbol].append(metrics)

        if algo not in self._algo_metrics:
            self._algo_metrics[algo] = []
        self._algo_metrics[algo].append(metrics)

        return metrics

    def get_summary_stats(
        self,
        symbol: str | None = None,
        algo: str | None = None,
        lookback: int = 50,
    ) -> dict[str, Any]:
        """
        Get summary statistics for execution quality.

        Args:
            symbol: Filter by symbol (optional)
            algo: Filter by algorithm (optional)
            lookback: Number of recent executions to analyze

        Returns:
            Summary statistics dictionary
        """
        if symbol:
            metrics = self._symbol_metrics.get(symbol, [])[-lookback:]
        elif algo:
            metrics = self._algo_metrics.get(algo, [])[-lookback:]
        else:
            metrics = self._metrics_history[-lookback:]

        if not metrics:
            return {
                "count": 0,
                "avg_slippage_arrival_bps": None,
                "avg_slippage_vwap_bps": None,
                "avg_fill_rate": None,
            }

        slippage_arrivals = [m.slippage_vs_arrival_bps for m in metrics]
        slippage_vwaps = [m.slippage_vs_vwap_bps for m in metrics]
        fill_rates = [m.fill_rate for m in metrics]
        spread_captures = [m.spread_capture_pct for m in metrics if m.spread_capture_pct != 0]

        return {
            "count": len(metrics),
            "avg_slippage_arrival_bps": np.mean(slippage_arrivals),
            "std_slippage_arrival_bps": np.std(slippage_arrivals),
            "avg_slippage_vwap_bps": np.mean(slippage_vwaps),
            "std_slippage_vwap_bps": np.std(slippage_vwaps),
            "avg_fill_rate": np.mean(fill_rates),
            "avg_spread_capture_pct": np.mean(spread_captures) if spread_captures else None,
            "worst_slippage_bps": max(slippage_arrivals),
            "best_slippage_bps": min(slippage_arrivals),
        }

    def get_status(self) -> dict[str, Any]:
        """Get monitor status."""
        return {
            "total_executions": len(self._metrics_history),
            "symbols_tracked": len(self._symbol_metrics),
            "algos_tracked": len(self._algo_metrics),
            "recent_avg_slippage_bps": (
                np.mean([m.slippage_vs_arrival_bps for m in self._metrics_history[-20:]])
                if self._metrics_history else None
            ),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_adaptive_twap(config: dict[str, Any] | None = None) -> AdaptiveTWAP:
    """Create AdaptiveTWAP instance."""
    if config:
        cfg = AdaptiveTWAPConfig(**config)
    else:
        cfg = AdaptiveTWAPConfig()
    return AdaptiveTWAP(cfg)


def create_dynamic_slippage_caps(config: dict[str, Any] | None = None) -> DynamicSlippageCaps:
    """Create DynamicSlippageCaps instance."""
    if config:
        cfg = SlippageConfig(**config)
    else:
        cfg = SlippageConfig()
    return DynamicSlippageCaps(cfg)


def create_session_aware_execution() -> SessionAwareExecution:
    """Create SessionAwareExecution instance."""
    return SessionAwareExecution()


def create_smart_algo_selector() -> SmartAlgoSelector:
    """Create SmartAlgoSelector instance."""
    return SmartAlgoSelector()


def create_fill_quality_monitor() -> FillQualityMonitor:
    """Create FillQualityMonitor instance."""
    return FillQualityMonitor()

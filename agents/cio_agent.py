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
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from core.agent_base import DecisionAgent, AgentConfig
from core.events import (
    Event,
    EventType,
    SignalEvent,
    DecisionEvent,
    ValidatedDecisionEvent,
    SignalDirection,
    OrderSide,
    OrderType,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger
    from core.position_sizing import PositionSizer, StrategyStats
    from core.attribution import PerformanceAttribution
    from core.correlation_manager import CorrelationManager, CorrelationRegime
    from core.risk_budget import RiskBudgetManager


logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification for weight adjustment."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    VOLATILE = "volatile"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    NEUTRAL = "neutral"


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

        # Base signal weights by strategy
        self._base_weights = {
            "MacroAgent": config.parameters.get("signal_weight_macro", 0.15),
            "StatArbAgent": config.parameters.get("signal_weight_stat_arb", 0.25),
            "MomentumAgent": config.parameters.get("signal_weight_momentum", 0.25),
            "MarketMakingAgent": config.parameters.get("signal_weight_market_making", 0.15),
            "OptionsVolAgent": config.parameters.get("signal_weight_options_vol", 0.20),
        }

        # Current effective weights (may be adjusted dynamically)
        self._weights = dict(self._base_weights)

        # Decision thresholds
        self._min_conviction = config.parameters.get("min_conviction_threshold", 0.6)
        self._max_concurrent = config.parameters.get("max_concurrent_decisions", 5)

        # Dynamic weight settings
        self._use_dynamic_weights = config.parameters.get("use_dynamic_weights", True)
        self._performance_weight_factor = config.parameters.get("performance_weight_factor", 0.3)
        self._regime_weight_factor = config.parameters.get("regime_weight_factor", 0.2)

        # Position sizing settings
        self._use_kelly_sizing = config.parameters.get("use_kelly_sizing", True)
        self._kelly_fraction = config.parameters.get("kelly_fraction", 0.5)  # Half-Kelly
        self._base_position_size = config.parameters.get("base_position_size", 100)
        self._max_position_size = config.parameters.get("max_position_size", 1000)
        self._portfolio_value = config.parameters.get("portfolio_value", 1_000_000.0)

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

        # Signal history for correlation tracking (#Q5)
        self._signal_history: dict[str, list[tuple[datetime, float]]] = {}  # agent -> [(time, direction_val)]
        self._signal_correlation_matrix: dict[tuple[str, str], float] = {}  # (agent1, agent2) -> correlation
        self._max_signal_history = config.parameters.get("max_signal_history", 100)
        self._correlation_lookback = config.parameters.get("signal_correlation_lookback", 50)
        self._use_correlation_adjustment = config.parameters.get("use_signal_correlation_adjustment", True)

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
                "OptionsVolAgent": 1.4,
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

    async def initialize(self) -> None:
        """Initialize CIO agent."""
        logger.info(f"CIOAgent initializing with weights: {self._weights}")
        logger.info(f"Min conviction threshold: {self._min_conviction}")

    def get_subscribed_events(self) -> list[EventType]:
        """CIO subscribes to validated decisions only - signals come via barrier."""
        return [EventType.VALIDATED_DECISION]

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
        - Process signals together after synchronization
        - Avoid making decisions on partial signals
        """
        while self._running:
            try:
                # Wait for barrier to complete (blocking call with timeout)
                signals = await self._event_bus.wait_for_signals()

                if signals:
                    logger.info(f"CIO: Barrier complete with {len(signals)} signals")
                    await self._process_barrier_signals(signals)
                else:
                    # No signals, wait a bit before checking again
                    await asyncio.sleep(0.1)

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
        """
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

    async def _make_decision_from_aggregation(self, agg: SignalAggregation) -> None:
        """
        Make a trading decision from aggregated signals (post-barrier).

        This is THE decision point - all decisions go through here.
        Called only after barrier synchronization completes (fan-in).
        """
        symbol = agg.symbol

        # Aggregate signals
        self._aggregate_signals(agg)

        # Check conviction threshold
        if agg.weighted_confidence < self._min_conviction:
            logger.debug(
                f"CIO: Insufficient conviction for {symbol} "
                f"({agg.weighted_confidence:.2f} < {self._min_conviction})"
            )
            return

        # Clean up stale decisions first
        self._cleanup_stale_decisions()

        # Check concurrent decisions limit
        if len(self._active_decisions) >= self._max_concurrent:
            logger.warning(f"CIO: Max concurrent decisions ({len(self._active_decisions)}) reached, skipping {symbol}")
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
                logger.warning(
                    f"CIO: Strategy {best_strategy} is FROZEN ({budget.freeze_reason}), "
                    f"skipping decision for {symbol}"
                )
                return

            available_budget = self._risk_budget_manager.get_available_budget(best_strategy)
            if available_budget <= 0:
                logger.warning(
                    f"CIO: Strategy {best_strategy} has no available risk budget, "
                    f"skipping decision for {symbol}"
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
            return

        # Calculate position size using Kelly criterion and market data
        quantity = self._calculate_position_size(agg)

        if quantity == 0:
            return

        # Create decision event
        decision = DecisionEvent(
            source_agent=self.name,
            symbol=symbol,
            action=OrderSide.BUY if agg.consensus_direction == SignalDirection.LONG else OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.LIMIT,  # Default to limit orders
            limit_price=None,  # Will be set by execution
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
        """
        # Update dynamic weights if enabled
        if self._use_dynamic_weights:
            self._update_dynamic_weights()

        # Record signals to history for correlation tracking (#Q5)
        self._record_signals_to_history(agg.signals)

        # Get correlation-adjusted weights (#Q5)
        if self._use_correlation_adjustment and len(agg.signals) > 1:
            adjusted_weights = self._get_correlation_adjusted_weights(agg.signals)
            agg.correlation_adjusted = True
        else:
            adjusted_weights = {agent: self._weights.get(agent, 0.1) for agent in agg.signals}
            agg.correlation_adjusted = False

        total_weight = 0.0
        weighted_strength = 0.0
        weighted_confidence = 0.0

        long_votes = 0.0
        short_votes = 0.0

        for agent_name, signal in agg.signals.items():
            weight = adjusted_weights.get(agent_name, 0.1)
            total_weight += weight

            # Aggregate strength and confidence
            weighted_strength += signal.strength * weight
            weighted_confidence += signal.confidence * weight

            # Count directional votes
            if signal.direction == SignalDirection.LONG:
                long_votes += weight
            elif signal.direction == SignalDirection.SHORT:
                short_votes += weight

        if total_weight > 0:
            agg.weighted_strength = weighted_strength / total_weight
            agg.weighted_confidence = weighted_confidence / total_weight

        # Calculate effective signal count (#Q5)
        agg.effective_signal_count = self._calculate_effective_signal_count(agg.signals)

        # Determine consensus direction
        if long_votes > short_votes and long_votes > total_weight * 0.4:
            agg.consensus_direction = SignalDirection.LONG
        elif short_votes > long_votes and short_votes > total_weight * 0.4:
            agg.consensus_direction = SignalDirection.SHORT
        else:
            agg.consensus_direction = SignalDirection.FLAT

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
        signals: dict[str, SignalEvent]
    ) -> dict[str, float]:
        """
        Adjust signal weights to account for correlation (#Q5).

        Highly correlated signals should not be double-counted.
        Uses a discount factor based on pairwise correlations.

        The adjustment reduces the effective weight of signals that are
        highly correlated with others, preventing overconfidence from
        redundant information.
        """
        agents = list(signals.keys())
        base_weights = {agent: self._weights.get(agent, 0.1) for agent in agents}

        if len(agents) < 2:
            return base_weights

        # Calculate correlation discount for each agent
        adjusted_weights = {}

        for agent in agents:
            # Find maximum correlation with other signals in this set
            max_corr = 0.0
            correlated_agents = []

            for other_agent in agents:
                if other_agent == agent:
                    continue

                corr = self._signal_correlation_matrix.get((agent, other_agent))
                if corr is not None:
                    if abs(corr) > 0.5:  # Significant correlation
                        correlated_agents.append((other_agent, corr))
                    max_corr = max(max_corr, abs(corr))

            # Apply discount based on correlation
            # High correlation (>0.8) reduces weight significantly
            # Moderate correlation (0.5-0.8) reduces weight moderately
            if max_corr > 0.8:
                discount = 0.5  # 50% weight reduction
            elif max_corr > 0.6:
                discount = 0.75  # 25% weight reduction
            elif max_corr > 0.5:
                discount = 0.9  # 10% weight reduction
            else:
                discount = 1.0  # No discount

            adjusted_weights[agent] = base_weights[agent] * discount

            if discount < 1.0 and correlated_agents:
                corr_info = ", ".join([f"{a}:{c:.2f}" for a, c in correlated_agents])
                logger.debug(
                    f"Signal {agent} weight discounted by {(1-discount)*100:.0f}% "
                    f"due to correlation with [{corr_info}]"
                )

        # Normalize weights to sum to original total
        original_total = sum(base_weights.values())
        adjusted_total = sum(adjusted_weights.values())

        if adjusted_total > 0 and original_total > 0:
            scale = original_total / adjusted_total
            adjusted_weights = {k: v * scale for k, v in adjusted_weights.items()}

        return adjusted_weights

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
        Calculate position size using Kelly criterion.

        Kelly formula: f* = (bp - q) / b
        Where:
        - b = avg_win / avg_loss (win/loss ratio)
        - p = win probability
        - q = 1 - p

        Uses actual tracked statistics rather than estimation from Sharpe.
        """
        # Get strategy with highest contribution
        best_strategy = max(
            agg.signals.keys(),
            key=lambda s: self._weights.get(s, 0) * agg.signals[s].confidence
        )

        # Get strategy stats
        perf = self._strategy_performance.get(best_strategy)
        if not perf:
            # Fall back to conviction-based sizing
            return self._calculate_conviction_size(agg)

        # P1-9: Need minimum trades for statistical significance
        # 30 trades gives ~18% standard error on win rate estimation
        # 50 trades gives ~14% standard error - more reliable for sizing
        MIN_TRADES_FOR_KELLY = 50  # Increased from 30 for better reliability
        WARN_TRADES_THRESHOLD = 100  # Warn if below this - still learning

        if perf.total_trades < MIN_TRADES_FOR_KELLY:
            logger.info(
                f"Kelly: Insufficient trades for {best_strategy} "
                f"({perf.total_trades}/{MIN_TRADES_FOR_KELLY}), using conviction sizing"
            )
            return self._calculate_conviction_size(agg)

        if perf.total_trades < WARN_TRADES_THRESHOLD:
            logger.warning(
                f"Kelly: Low sample size for {best_strategy} ({perf.total_trades} trades). "
                f"Position sizing may be unreliable. Consider reducing kelly_fraction."
            )

        # Kelly inputs from actual tracked data
        win_rate = perf.win_rate
        avg_win = perf.avg_win
        avg_loss = perf.avg_loss

        # Validate inputs
        if win_rate <= 0 or win_rate >= 1:
            return self._calculate_conviction_size(agg)

        if avg_win <= 0 or avg_loss <= 0:
            return self._calculate_conviction_size(agg)

        # Calculate b = win/loss ratio (odds)
        b = avg_win / avg_loss

        p = win_rate
        q = 1 - p

        # Kelly formula: f* = (bp - q) / b
        kelly_fraction = (b * p - q) / b if b > 0 else 0

        # Ensure non-negative (negative means don't bet)
        kelly_fraction = max(0, kelly_fraction)

        # Cap at reasonable maximum (full Kelly is too aggressive)
        MAX_KELLY = 0.25  # Never risk more than 25% even with perfect Kelly
        kelly_fraction = min(kelly_fraction, MAX_KELLY)

        # Apply half-Kelly for safety
        kelly_fraction *= self._kelly_fraction

        # P1-9: Apply sample size discount - be more conservative with fewer trades
        # At 50 trades: 0.7x, at 100 trades: 0.85x, at 200+ trades: 1.0x
        sample_discount = min(1.0, 0.5 + (perf.total_trades / 400))
        kelly_fraction *= sample_discount

        # Calculate position value
        position_value = self._portfolio_value * kelly_fraction

        # Apply conviction adjustment
        conviction_multiplier = agg.weighted_confidence * abs(agg.weighted_strength)
        position_value *= conviction_multiplier

        # Apply correlation discount if available
        if self._correlation_manager:
            correlation_discount = self._get_correlation_discount(agg.symbol)
            position_value *= correlation_discount

        # Convert to shares using actual market price
        estimated_price = self._price_cache.get(agg.symbol)
        if estimated_price is None or estimated_price <= 0:
            logger.warning(f"CIO: No price data for {agg.symbol}, using conviction sizing")
            return self._calculate_conviction_size(agg)

        size = int(position_value / estimated_price)

        # Apply limits
        size = min(size, self._max_position_size)

        if size < 10:
            return 0

        return size

    def _calculate_conviction_size(self, agg: SignalAggregation) -> int:
        """Calculate position size based on conviction (fallback method)."""
        conviction_factor = agg.weighted_confidence
        strength_factor = abs(agg.weighted_strength)

        size = int(self._base_position_size * conviction_factor * strength_factor)

        # Apply limits
        size = min(size, self._max_position_size)

        if size < 10:
            return 0

        return size

    def _get_correlation_discount(self, symbol: str) -> float:
        """
        Get correlation-based discount for position sizing.

        Reduces size if highly correlated with existing positions.
        """
        if not self._correlation_manager:
            return 1.0

        # Get highly correlated pairs
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
        })

        return base_status

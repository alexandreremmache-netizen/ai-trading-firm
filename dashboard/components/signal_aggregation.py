"""
Signal Aggregation
==================

Multi-agent signal tracking and visualization for the trading system.

Aggregates signals from all signal agents:
- MacroAgent: Macro/sentiment signals
- StatArbAgent: Statistical arbitrage signals
- MomentumAgent: Momentum/trend signals
- MarketMakingAgent: Market making signals
- OptionsVolAgent: Options volatility signals
- SentimentAgent: Market sentiment signals
- ForecastingAgent: Price forecasting signals

Features:
- Signal recording with full metadata
- Per-symbol signal aggregation
- Consensus direction calculation
- Agreement/disagreement metrics
- Signal heatmap data generation
- WebSocket-ready export to dict
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from statistics import mean, stdev
from typing import Any

from core.events import SignalEvent, SignalDirection


logger = logging.getLogger(__name__)


# Known signal agents in the system
SIGNAL_AGENTS = [
    "MacroAgent",
    "StatArbAgent",
    "MomentumAgent",
    "MarketMakingAgent",
    "OptionsVolAgent",
    "SentimentAgent",
    "ForecastingAgent",
]


class ConsensusDirection(Enum):
    """Consensus direction across multiple agents."""
    STRONG_LONG = "strong_long"      # High agreement on LONG
    LONG = "long"                    # Majority LONG
    WEAK_LONG = "weak_long"          # Slight LONG bias
    NEUTRAL = "neutral"              # No clear consensus
    WEAK_SHORT = "weak_short"        # Slight SHORT bias
    SHORT = "short"                  # Majority SHORT
    STRONG_SHORT = "strong_short"    # High agreement on SHORT
    CONFLICTED = "conflicted"        # Strong disagreement


@dataclass
class SignalRecord:
    """
    Record of a single signal from an agent.

    Contains full signal metadata for tracking and compliance.
    """
    signal_id: str
    timestamp: datetime
    agent_name: str
    symbol: str
    direction: SignalDirection
    strength: float  # 0.0 to 1.0 (absolute value)
    confidence: float  # 0.0 to 1.0
    rationale: str
    data_sources: tuple[str, ...]
    strategy_name: str = ""
    target_price: float | None = None
    stop_loss: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strength": round(self.strength, 4),
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "data_sources": list(self.data_sources),
            "strategy_name": self.strategy_name,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
        }

    @classmethod
    def from_signal_event(cls, event: SignalEvent) -> SignalRecord:
        """Create a SignalRecord from a SignalEvent."""
        return cls(
            signal_id=event.event_id,
            timestamp=event.timestamp,
            agent_name=event.source_agent,
            symbol=event.symbol,
            direction=event.direction,
            strength=abs(event.strength),  # Normalize to 0-1
            confidence=event.confidence,
            rationale=event.rationale,
            data_sources=event.data_sources,
            strategy_name=event.strategy_name,
            target_price=event.target_price,
            stop_loss=event.stop_loss,
        )


@dataclass
class AggregatedSignal:
    """
    Aggregated signal view for a symbol across all agents.

    Provides consensus metrics and individual agent signals.
    """
    symbol: str
    signals_by_agent: dict[str, SignalRecord]
    consensus_direction: ConsensusDirection
    avg_strength: float  # 0.0 to 1.0
    avg_confidence: float  # 0.0 to 1.0
    agreement_score: float  # 0.0 to 1.0 (1.0 = full agreement)
    disagreement_score: float  # 0.0 to 1.0 (1.0 = full disagreement)
    num_long_signals: int
    num_short_signals: int
    num_flat_signals: int
    last_update: datetime
    weighted_direction_score: float  # -1.0 (SHORT) to 1.0 (LONG)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "symbol": self.symbol,
            "signals_by_agent": {
                agent: signal.to_dict()
                for agent, signal in self.signals_by_agent.items()
            },
            "consensus_direction": self.consensus_direction.value,
            "avg_strength": round(self.avg_strength, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "agreement_score": round(self.agreement_score, 4),
            "disagreement_score": round(self.disagreement_score, 4),
            "num_long_signals": self.num_long_signals,
            "num_short_signals": self.num_short_signals,
            "num_flat_signals": self.num_flat_signals,
            "last_update": self.last_update.isoformat(),
            "weighted_direction_score": round(self.weighted_direction_score, 4),
            "total_signals": len(self.signals_by_agent),
        }


@dataclass
class AgentSignalStats:
    """Statistics for a single agent's signals."""
    agent_name: str
    total_signals: int = 0
    long_signals: int = 0
    short_signals: int = 0
    flat_signals: int = 0
    avg_strength: float = 0.0
    avg_confidence: float = 0.0
    symbols_covered: set[str] = field(default_factory=set)
    last_signal_time: datetime | None = None
    is_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "agent_name": self.agent_name,
            "total_signals": self.total_signals,
            "long_signals": self.long_signals,
            "short_signals": self.short_signals,
            "flat_signals": self.flat_signals,
            "avg_strength": round(self.avg_strength, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "symbols_covered": list(self.symbols_covered),
            "num_symbols": len(self.symbols_covered),
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
            "is_active": self.is_active,
        }


@dataclass
class HeatmapCell:
    """Single cell in the signal heatmap (symbol x agent)."""
    symbol: str
    agent_name: str
    direction: SignalDirection | None
    strength: float
    confidence: float
    timestamp: datetime | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "symbol": self.symbol,
            "agent_name": self.agent_name,
            "direction": self.direction.value if self.direction else None,
            "strength": round(self.strength, 4),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            # Computed value for heatmap coloring: -1 (short) to 1 (long)
            "value": self._compute_value(),
        }

    def _compute_value(self) -> float:
        """Compute heatmap value: -1 (strong short) to 1 (strong long)."""
        if self.direction is None:
            return 0.0

        if self.direction == SignalDirection.FLAT:
            return 0.0
        elif self.direction == SignalDirection.LONG:
            return self.strength * self.confidence
        else:  # SHORT
            return -self.strength * self.confidence


class SignalAggregator:
    """
    Aggregates and tracks signals from multiple agents.

    Maintains signal history and computes consensus metrics
    for visualization and decision support.

    Usage:
        aggregator = SignalAggregator()

        # Record signals from events
        aggregator.record_signal(signal_event)

        # Get aggregated view for a symbol
        agg = aggregator.get_aggregated_view("AAPL")

        # Get signals by agent
        signals = aggregator.get_signals_by_agent("MacroAgent")

        # Get heatmap data
        heatmap = aggregator.get_signal_heatmap()

        # Export for WebSocket
        data = aggregator.to_dict()
    """

    # Maximum signals to keep in history
    MAX_SIGNALS = 5000

    # Time window for "active" signal (seconds)
    ACTIVE_WINDOW_SECONDS = 300  # 5 minutes

    # Time window for "recent" signals (seconds)
    RECENT_WINDOW_SECONDS = 3600  # 1 hour

    def __init__(
        self,
        max_signals: int = MAX_SIGNALS,
        signal_agents: list[str] | None = None,
    ):
        """
        Initialize the signal aggregator.

        Args:
            max_signals: Maximum signals to keep in history (default 5000)
            signal_agents: List of signal agent names to track
        """
        self._max_signals = max_signals
        self._signal_agents = signal_agents or SIGNAL_AGENTS.copy()

        # Signal storage
        self._all_signals: deque[SignalRecord] = deque(maxlen=max_signals)

        # Current signals by symbol (latest per agent)
        self._current_signals: dict[str, dict[str, SignalRecord]] = {}

        # Agent statistics
        self._agent_stats: dict[str, AgentSignalStats] = {
            agent: AgentSignalStats(agent_name=agent)
            for agent in self._signal_agents
        }

        # Running statistics for EMA calculations
        self._strength_ema: dict[str, float] = {}  # Per agent
        self._confidence_ema: dict[str, float] = {}  # Per agent

        # Metrics
        self._total_signals_recorded = 0

        logger.info(
            f"SignalAggregator initialized with max_signals={max_signals}, "
            f"tracking agents: {self._signal_agents}"
        )

    def record_signal(self, signal: SignalEvent | SignalRecord) -> SignalRecord:
        """
        Record a signal from an agent.

        Args:
            signal: SignalEvent or SignalRecord to record

        Returns:
            The recorded SignalRecord
        """
        # Convert SignalEvent to SignalRecord if needed
        if isinstance(signal, SignalEvent):
            record = SignalRecord.from_signal_event(signal)
        else:
            record = signal

        # Add to history
        self._all_signals.append(record)
        self._total_signals_recorded += 1

        # Update current signals for symbol
        symbol = record.symbol
        if symbol not in self._current_signals:
            self._current_signals[symbol] = {}

        self._current_signals[symbol][record.agent_name] = record

        # Update agent statistics
        self._update_agent_stats(record)

        logger.debug(
            f"Recorded signal: {record.agent_name} -> {record.symbol} "
            f"{record.direction.value} (strength={record.strength:.2f})"
        )

        return record

    def _update_agent_stats(self, record: SignalRecord) -> None:
        """Update statistics for the agent that generated the signal."""
        agent = record.agent_name

        # Create stats if agent is new
        if agent not in self._agent_stats:
            self._agent_stats[agent] = AgentSignalStats(agent_name=agent)
            # Add to known agents if not already tracked
            if agent not in self._signal_agents:
                self._signal_agents.append(agent)

        stats = self._agent_stats[agent]

        # Update counts
        stats.total_signals += 1

        if record.direction == SignalDirection.LONG:
            stats.long_signals += 1
        elif record.direction == SignalDirection.SHORT:
            stats.short_signals += 1
        else:
            stats.flat_signals += 1

        stats.symbols_covered.add(record.symbol)
        stats.last_signal_time = record.timestamp

        # Update EMA for strength and confidence
        alpha = 0.1  # Smoothing factor

        if agent not in self._strength_ema:
            self._strength_ema[agent] = record.strength
        else:
            self._strength_ema[agent] = (
                alpha * record.strength + (1 - alpha) * self._strength_ema[agent]
            )

        if agent not in self._confidence_ema:
            self._confidence_ema[agent] = record.confidence
        else:
            self._confidence_ema[agent] = (
                alpha * record.confidence + (1 - alpha) * self._confidence_ema[agent]
            )

        stats.avg_strength = self._strength_ema[agent]
        stats.avg_confidence = self._confidence_ema[agent]

    def get_signals_by_symbol(
        self,
        symbol: str,
        limit: int = 100,
        include_history: bool = False,
    ) -> list[SignalRecord]:
        """
        Get signals for a specific symbol.

        Args:
            symbol: Symbol to get signals for
            limit: Maximum number of signals to return
            include_history: If True, include historical signals; otherwise only current

        Returns:
            List of SignalRecord for the symbol
        """
        if include_history:
            # Get from history (most recent first)
            signals = [s for s in self._all_signals if s.symbol == symbol]
            signals.reverse()
            return signals[:limit]
        else:
            # Get current signals only
            current = self._current_signals.get(symbol, {})
            return list(current.values())[:limit]

    def get_signals_by_agent(
        self,
        agent_name: str,
        limit: int = 100,
    ) -> list[SignalRecord]:
        """
        Get signals from a specific agent.

        Args:
            agent_name: Agent name to filter by
            limit: Maximum number of signals to return

        Returns:
            List of SignalRecord from the agent (most recent first)
        """
        signals = [s for s in self._all_signals if s.agent_name == agent_name]
        signals.reverse()
        return signals[:limit]

    def get_aggregated_view(self, symbol: str) -> AggregatedSignal | None:
        """
        Get aggregated signal view for a symbol.

        Computes consensus direction, agreement scores, and other metrics
        across all agents that have signaled on this symbol.

        Args:
            symbol: Symbol to aggregate signals for

        Returns:
            AggregatedSignal or None if no signals exist
        """
        current = self._current_signals.get(symbol)
        if not current:
            return None

        signals_by_agent = current.copy()

        # Compute metrics
        strengths = []
        confidences = []
        direction_scores = []  # -1 for short, 0 for flat, 1 for long

        num_long = 0
        num_short = 0
        num_flat = 0
        last_update = datetime.min.replace(tzinfo=timezone.utc)

        for signal in signals_by_agent.values():
            strengths.append(signal.strength)
            confidences.append(signal.confidence)

            if signal.direction == SignalDirection.LONG:
                direction_scores.append(signal.strength * signal.confidence)
                num_long += 1
            elif signal.direction == SignalDirection.SHORT:
                direction_scores.append(-signal.strength * signal.confidence)
                num_short += 1
            else:
                direction_scores.append(0.0)
                num_flat += 1

            if signal.timestamp > last_update:
                last_update = signal.timestamp

        # Calculate averages
        avg_strength = mean(strengths) if strengths else 0.0
        avg_confidence = mean(confidences) if confidences else 0.0
        weighted_direction = mean(direction_scores) if direction_scores else 0.0

        # Calculate agreement/disagreement
        agreement_score, disagreement_score = self._calculate_agreement(
            direction_scores
        )

        # Determine consensus direction
        consensus = self._determine_consensus(
            weighted_direction,
            agreement_score,
            num_long,
            num_short,
            num_flat,
            len(signals_by_agent),
        )

        return AggregatedSignal(
            symbol=symbol,
            signals_by_agent=signals_by_agent,
            consensus_direction=consensus,
            avg_strength=avg_strength,
            avg_confidence=avg_confidence,
            agreement_score=agreement_score,
            disagreement_score=disagreement_score,
            num_long_signals=num_long,
            num_short_signals=num_short,
            num_flat_signals=num_flat,
            last_update=last_update,
            weighted_direction_score=weighted_direction,
        )

    def _calculate_agreement(
        self,
        direction_scores: list[float],
    ) -> tuple[float, float]:
        """
        Calculate agreement and disagreement scores.

        Agreement = 1 - normalized standard deviation of direction scores
        Disagreement = 1 - agreement

        Returns:
            Tuple of (agreement_score, disagreement_score)
        """
        if len(direction_scores) < 2:
            return 1.0, 0.0  # Single signal = full agreement

        try:
            std = stdev(direction_scores)
            # Normalize: max possible stdev for [-1, 1] range is 1
            normalized_std = min(std, 1.0)
            agreement = 1.0 - normalized_std
            disagreement = normalized_std
            return agreement, disagreement
        except Exception:
            return 1.0, 0.0

    def _determine_consensus(
        self,
        weighted_direction: float,
        agreement: float,
        num_long: int,
        num_short: int,
        num_flat: int,
        total: int,
    ) -> ConsensusDirection:
        """
        Determine the consensus direction based on signals.

        Args:
            weighted_direction: Weighted average direction (-1 to 1)
            agreement: Agreement score (0 to 1)
            num_long: Number of LONG signals
            num_short: Number of SHORT signals
            num_flat: Number of FLAT signals
            total: Total number of signals

        Returns:
            ConsensusDirection
        """
        if total == 0:
            return ConsensusDirection.NEUTRAL

        # Check for conflicted state (low agreement with mixed signals)
        if agreement < 0.3 and num_long > 0 and num_short > 0:
            return ConsensusDirection.CONFLICTED

        # Determine direction based on weighted score and agreement
        if weighted_direction > 0.6 and agreement > 0.6:
            return ConsensusDirection.STRONG_LONG
        elif weighted_direction > 0.4:
            return ConsensusDirection.LONG
        elif weighted_direction > 0.1:
            return ConsensusDirection.WEAK_LONG
        elif weighted_direction < -0.6 and agreement > 0.6:
            return ConsensusDirection.STRONG_SHORT
        elif weighted_direction < -0.4:
            return ConsensusDirection.SHORT
        elif weighted_direction < -0.1:
            return ConsensusDirection.WEAK_SHORT
        else:
            return ConsensusDirection.NEUTRAL

    def get_signal_history(
        self,
        symbol: str | None = None,
        agent_name: str | None = None,
        direction: SignalDirection | None = None,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[SignalRecord]:
        """
        Get historical signals with optional filters.

        Args:
            symbol: Filter by symbol (optional)
            agent_name: Filter by agent name (optional)
            direction: Filter by direction (optional)
            since: Only include signals after this time (optional)
            limit: Maximum number of signals to return

        Returns:
            List of SignalRecord matching filters (most recent first)
        """
        signals = list(self._all_signals)
        signals.reverse()  # Most recent first

        # Apply filters
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]

        if agent_name:
            signals = [s for s in signals if s.agent_name == agent_name]

        if direction:
            signals = [s for s in signals if s.direction == direction]

        if since:
            signals = [s for s in signals if s.timestamp >= since]

        return signals[:limit]

    def get_all_aggregated_views(self) -> dict[str, AggregatedSignal]:
        """
        Get aggregated views for all symbols with current signals.

        Returns:
            Dict mapping symbol to AggregatedSignal
        """
        result = {}
        for symbol in self._current_signals:
            agg = self.get_aggregated_view(symbol)
            if agg:
                result[symbol] = agg
        return result

    def get_agent_statistics(self) -> dict[str, AgentSignalStats]:
        """
        Get statistics for all tracked agents.

        Returns:
            Dict mapping agent name to AgentSignalStats
        """
        now = datetime.now(timezone.utc)
        active_threshold = now - timedelta(seconds=self.ACTIVE_WINDOW_SECONDS)

        # Update active status for each agent
        for stats in self._agent_stats.values():
            if stats.last_signal_time:
                stats.is_active = stats.last_signal_time >= active_threshold
            else:
                stats.is_active = False

        return self._agent_stats.copy()

    def get_signal_heatmap(
        self,
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Get signal heatmap data (symbol x agent matrix).

        Args:
            symbols: List of symbols to include (None = all with signals)

        Returns:
            Dict with heatmap data for frontend visualization
        """
        # Determine symbols to include
        if symbols:
            target_symbols = symbols
        else:
            target_symbols = list(self._current_signals.keys())

        target_symbols.sort()

        # Build heatmap data
        rows = []  # One row per symbol
        cells: list[HeatmapCell] = []

        for symbol in target_symbols:
            row = {"symbol": symbol, "agents": {}}
            current = self._current_signals.get(symbol, {})

            for agent in self._signal_agents:
                signal = current.get(agent)

                if signal:
                    cell = HeatmapCell(
                        symbol=symbol,
                        agent_name=agent,
                        direction=signal.direction,
                        strength=signal.strength,
                        confidence=signal.confidence,
                        timestamp=signal.timestamp,
                    )
                else:
                    cell = HeatmapCell(
                        symbol=symbol,
                        agent_name=agent,
                        direction=None,
                        strength=0.0,
                        confidence=0.0,
                        timestamp=None,
                    )

                cells.append(cell)
                row["agents"][agent] = cell.to_dict()

            rows.append(row)

        return {
            "symbols": target_symbols,
            "agents": self._signal_agents,
            "rows": rows,
            "cells": [c.to_dict() for c in cells],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_consensus_summary(self) -> dict[str, Any]:
        """
        Get summary of consensus across all symbols.

        Returns:
            Dict with consensus counts and breakdowns
        """
        consensus_counts: dict[str, int] = {c.value: 0 for c in ConsensusDirection}
        symbol_consensus: dict[str, str] = {}

        for symbol in self._current_signals:
            agg = self.get_aggregated_view(symbol)
            if agg:
                consensus_counts[agg.consensus_direction.value] += 1
                symbol_consensus[symbol] = agg.consensus_direction.value

        # Calculate overall market bias
        bullish = sum([
            consensus_counts[ConsensusDirection.STRONG_LONG.value],
            consensus_counts[ConsensusDirection.LONG.value],
            consensus_counts[ConsensusDirection.WEAK_LONG.value],
        ])
        bearish = sum([
            consensus_counts[ConsensusDirection.STRONG_SHORT.value],
            consensus_counts[ConsensusDirection.SHORT.value],
            consensus_counts[ConsensusDirection.WEAK_SHORT.value],
        ])
        neutral = consensus_counts[ConsensusDirection.NEUTRAL.value]
        conflicted = consensus_counts[ConsensusDirection.CONFLICTED.value]

        total = len(symbol_consensus)
        if total > 0:
            market_bias = (bullish - bearish) / total
        else:
            market_bias = 0.0

        return {
            "consensus_counts": consensus_counts,
            "symbol_consensus": symbol_consensus,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "conflicted_count": conflicted,
            "total_symbols": total,
            "market_bias": round(market_bias, 4),  # -1 to 1
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_disagreement_alerts(
        self,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Get symbols where agents strongly disagree.

        Args:
            threshold: Disagreement score threshold (0-1)

        Returns:
            List of alerts for high-disagreement symbols
        """
        alerts = []

        for symbol in self._current_signals:
            agg = self.get_aggregated_view(symbol)
            if agg and agg.disagreement_score >= threshold:
                # Identify conflicting agents
                long_agents = [
                    agent for agent, sig in agg.signals_by_agent.items()
                    if sig.direction == SignalDirection.LONG
                ]
                short_agents = [
                    agent for agent, sig in agg.signals_by_agent.items()
                    if sig.direction == SignalDirection.SHORT
                ]

                alerts.append({
                    "symbol": symbol,
                    "disagreement_score": round(agg.disagreement_score, 4),
                    "long_agents": long_agents,
                    "short_agents": short_agents,
                    "num_long": agg.num_long_signals,
                    "num_short": agg.num_short_signals,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        # Sort by disagreement score descending
        alerts.sort(key=lambda x: x["disagreement_score"], reverse=True)
        return alerts

    def clear(self) -> None:
        """Clear all signals and reset statistics."""
        self._all_signals.clear()
        self._current_signals.clear()
        self._strength_ema.clear()
        self._confidence_ema.clear()
        self._total_signals_recorded = 0

        # Reset agent stats
        self._agent_stats = {
            agent: AgentSignalStats(agent_name=agent)
            for agent in self._signal_agents
        }

        logger.info("SignalAggregator cleared")

    def to_dict(self) -> dict[str, Any]:
        """
        Export aggregator state to dictionary for WebSocket streaming.

        Returns:
            Complete aggregator state as dict
        """
        # Get recent signals (last 50)
        recent_signals = list(self._all_signals)
        recent_signals.reverse()
        recent_signals = recent_signals[:50]

        # Get all aggregated views
        aggregated = self.get_all_aggregated_views()

        # Get agent stats
        agent_stats = self.get_agent_statistics()

        # Get consensus summary
        consensus = self.get_consensus_summary()

        # Get disagreement alerts
        alerts = self.get_disagreement_alerts(threshold=0.4)

        return {
            "recent_signals": [s.to_dict() for s in recent_signals],
            "aggregated_views": {
                symbol: agg.to_dict()
                for symbol, agg in aggregated.items()
            },
            "agent_statistics": {
                agent: stats.to_dict()
                for agent, stats in agent_stats.items()
            },
            "consensus_summary": consensus,
            "disagreement_alerts": alerts,
            "heatmap": self.get_signal_heatmap(),
            "total_signals_recorded": self._total_signals_recorded,
            "buffer_size": len(self._all_signals),
            "buffer_max_size": self._max_signals,
            "symbols_tracked": len(self._current_signals),
            "agents_tracked": self._signal_agents,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @property
    def total_signals(self) -> int:
        """Total number of signals recorded."""
        return self._total_signals_recorded

    @property
    def buffer_size(self) -> int:
        """Current number of signals in the buffer."""
        return len(self._all_signals)

    @property
    def symbols_tracked(self) -> list[str]:
        """List of symbols currently tracked."""
        return list(self._current_signals.keys())

    @property
    def agents_tracked(self) -> list[str]:
        """List of agents being tracked."""
        return self._signal_agents.copy()

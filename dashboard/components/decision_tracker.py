"""
Decision Tracker
================

Track CIO decisions and their outcomes throughout the decision pipeline.

Monitors the decision lifecycle:
Proposed -> Risk Validated -> Compliance Validated -> Executed

Features:
- Track decision pipeline stages
- Link decisions to fills for P&L tracking
- Calculate decision statistics (win rate, conviction analysis)
- WebSocket-ready export to dict
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

from core.events import (
    DecisionEvent,
    ValidatedDecisionEvent,
    FillEvent,
    OrderSide,
)


logger = logging.getLogger(__name__)


class DecisionOutcome(Enum):
    """Outcome status of a decision."""
    PENDING = "pending"              # Decision proposed, awaiting validation
    RISK_VALIDATED = "risk_validated"  # Passed risk validation
    COMPLIANCE_VALIDATED = "compliance_validated"  # Passed compliance validation
    EXECUTED = "executed"            # Order executed (fill received)
    REJECTED = "rejected"            # Rejected by risk or compliance
    PARTIAL = "partial"              # Partially filled
    CANCELLED = "cancelled"          # Cancelled before execution
    EXPIRED = "expired"              # Timed out without execution


class DecisionAction(Enum):
    """Trading action for a decision."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class DecisionRecord:
    """
    Record of a single CIO decision with full lifecycle tracking.

    Captures decision metadata, pipeline state, and P&L outcomes.
    """
    decision_id: str
    timestamp: datetime
    symbol: str
    action: DecisionAction
    quantity: int
    conviction_score: float
    contributing_signals: tuple[str, ...]
    rationale: str
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    pnl_impact: float | None = None

    # Pipeline tracking
    risk_validated_at: datetime | None = None
    compliance_validated_at: datetime | None = None
    executed_at: datetime | None = None

    # Validation details
    risk_adjusted_quantity: int | None = None
    rejection_reason: str | None = None
    rejection_source: str | None = None  # "risk" or "compliance"

    # Execution details
    fill_price: float | None = None
    fill_quantity: int = 0
    commission: float = 0.0
    order_ids: list[str] = field(default_factory=list)

    # P&L tracking
    entry_price: float | None = None
    exit_price: float | None = None
    realized_pnl: float | None = None
    unrealized_pnl: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "action": self.action.value,
            "quantity": self.quantity,
            "conviction_score": round(self.conviction_score, 4),
            "contributing_signals": list(self.contributing_signals),
            "rationale": self.rationale[:200] if len(self.rationale) > 200 else self.rationale,
            "outcome": self.outcome.value,
            "pnl_impact": round(self.pnl_impact, 2) if self.pnl_impact is not None else None,
            # Pipeline tracking
            "risk_validated_at": self.risk_validated_at.isoformat() if self.risk_validated_at else None,
            "compliance_validated_at": self.compliance_validated_at.isoformat() if self.compliance_validated_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            # Validation details
            "risk_adjusted_quantity": self.risk_adjusted_quantity,
            "rejection_reason": self.rejection_reason,
            "rejection_source": self.rejection_source,
            # Execution details
            "fill_price": self.fill_price,
            "fill_quantity": self.fill_quantity,
            "commission": round(self.commission, 4) if self.commission else 0.0,
            # P&L tracking
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "realized_pnl": round(self.realized_pnl, 2) if self.realized_pnl is not None else None,
            "unrealized_pnl": round(self.unrealized_pnl, 2) if self.unrealized_pnl is not None else None,
        }

    @property
    def is_terminal(self) -> bool:
        """Check if decision has reached a terminal state."""
        return self.outcome in (
            DecisionOutcome.EXECUTED,
            DecisionOutcome.REJECTED,
            DecisionOutcome.CANCELLED,
            DecisionOutcome.EXPIRED,
        )

    @property
    def pipeline_latency_ms(self) -> float | None:
        """Calculate total pipeline latency from proposal to execution."""
        if self.executed_at:
            return (self.executed_at - self.timestamp).total_seconds() * 1000
        return None

    @property
    def is_profitable(self) -> bool | None:
        """Check if the decision was profitable (requires realized P&L)."""
        if self.realized_pnl is not None:
            return self.realized_pnl > 0
        if self.pnl_impact is not None:
            return self.pnl_impact > 0
        return None


@dataclass
class DecisionStatistics:
    """
    Aggregated statistics about CIO decisions.

    Provides metrics for monitoring decision quality and performance.
    """
    total_decisions: int = 0
    win_rate: float = 0.0
    avg_conviction: float = 0.0
    decisions_by_action: dict[str, int] = field(default_factory=dict)
    decisions_by_symbol: dict[str, int] = field(default_factory=dict)
    decisions_by_outcome: dict[str, int] = field(default_factory=dict)

    # P&L metrics
    total_pnl: float = 0.0
    avg_pnl_per_decision: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    profit_factor: float = 0.0  # gross_profit / gross_loss

    # Pipeline metrics
    avg_pipeline_latency_ms: float = 0.0
    rejection_rate: float = 0.0
    execution_rate: float = 0.0

    # Time-based metrics
    decisions_last_hour: int = 0
    decisions_today: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Conviction analysis
    avg_conviction_winners: float = 0.0
    avg_conviction_losers: float = 0.0
    high_conviction_win_rate: float = 0.0  # Win rate for conviction > 0.7

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "total_decisions": self.total_decisions,
            "win_rate": round(self.win_rate, 4),
            "avg_conviction": round(self.avg_conviction, 4),
            "decisions_by_action": self.decisions_by_action,
            "decisions_by_symbol": self.decisions_by_symbol,
            "decisions_by_outcome": self.decisions_by_outcome,
            # P&L metrics
            "total_pnl": round(self.total_pnl, 2),
            "avg_pnl_per_decision": round(self.avg_pnl_per_decision, 2),
            "max_win": round(self.max_win, 2),
            "max_loss": round(self.max_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            # Pipeline metrics
            "avg_pipeline_latency_ms": round(self.avg_pipeline_latency_ms, 2),
            "rejection_rate": round(self.rejection_rate, 4),
            "execution_rate": round(self.execution_rate, 4),
            # Time-based metrics
            "decisions_last_hour": self.decisions_last_hour,
            "decisions_today": self.decisions_today,
            "last_updated": self.last_updated.isoformat(),
            # Conviction analysis
            "avg_conviction_winners": round(self.avg_conviction_winners, 4),
            "avg_conviction_losers": round(self.avg_conviction_losers, 4),
            "high_conviction_win_rate": round(self.high_conviction_win_rate, 4),
        }


class DecisionTracker:
    """
    Tracks and analyzes CIO decisions throughout their lifecycle.

    Maintains a circular buffer of decisions and computes statistics
    for monitoring decision quality and P&L performance.

    Usage:
        tracker = DecisionTracker()

        # Record a new decision
        tracker.record_decision(decision_event)

        # Update decision through pipeline
        tracker.update_outcome(decision_id, outcome, ...)

        # Link fills for P&L tracking
        tracker.link_fill(decision_id, fill_event)

        # Get recent decisions
        recent = tracker.get_recent_decisions(limit=50)

        # Get statistics
        stats = tracker.get_decision_stats()

        # Export for WebSocket streaming
        data = tracker.to_dict()
    """

    # Maximum number of decisions to keep in the circular buffer
    MAX_DECISIONS = 1000

    # Decision expiry timeout (decisions pending for longer are marked expired)
    DECISION_EXPIRY_SECONDS = 300  # 5 minutes

    def __init__(self, max_decisions: int = MAX_DECISIONS):
        """
        Initialize the decision tracker.

        Args:
            max_decisions: Maximum decisions to keep in circular buffer (default 1000)
        """
        self._max_decisions = max_decisions
        self._decisions: deque[DecisionRecord] = deque(maxlen=max_decisions)
        self._decision_index: dict[str, DecisionRecord] = {}  # decision_id -> record

        # Mapping from order_id to decision_id for fill linking
        self._order_to_decision: dict[str, str] = {}

        # Statistics tracking
        self._total_decisions = 0
        self._total_executed = 0
        self._total_rejected = 0
        self._total_profitable = 0
        self._total_unprofitable = 0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self._pipeline_latencies: deque[float] = deque(maxlen=1000)

        logger.info(f"DecisionTracker initialized with max_decisions={max_decisions}")

    def record_decision(
        self,
        decision: DecisionEvent,
    ) -> DecisionRecord:
        """
        Record a new CIO decision.

        Args:
            decision: The DecisionEvent from CIO agent

        Returns:
            DecisionRecord for the recorded decision
        """
        # Determine action
        if decision.action == OrderSide.BUY:
            action = DecisionAction.BUY
        elif decision.action == OrderSide.SELL:
            action = DecisionAction.SELL
        else:
            action = DecisionAction.HOLD

        # Create the record
        record = DecisionRecord(
            decision_id=decision.event_id,
            timestamp=decision.timestamp,
            symbol=decision.symbol,
            action=action,
            quantity=decision.quantity,
            conviction_score=decision.conviction_score,
            contributing_signals=decision.contributing_signals,
            rationale=decision.rationale,
            outcome=DecisionOutcome.PENDING,
        )

        # Add to circular buffer
        self._decisions.append(record)
        self._decision_index[decision.event_id] = record
        self._total_decisions += 1

        # Clean up old index entries if buffer wrapped
        self._cleanup_index()

        logger.debug(
            f"Recorded decision {decision.event_id[:8]} for {decision.symbol}: "
            f"{action.value} {decision.quantity} (conviction={decision.conviction_score:.2f})"
        )

        return record

    def update_outcome(
        self,
        decision_id: str,
        outcome: DecisionOutcome,
        adjusted_quantity: int | None = None,
        rejection_reason: str | None = None,
        rejection_source: str | None = None,
    ) -> DecisionRecord | None:
        """
        Update the outcome of a decision as it moves through the pipeline.

        Args:
            decision_id: ID of the decision to update
            outcome: New outcome status
            adjusted_quantity: Risk-adjusted quantity (if applicable)
            rejection_reason: Reason for rejection (if rejected)
            rejection_source: Source of rejection ("risk" or "compliance")

        Returns:
            Updated DecisionRecord or None if not found
        """
        record = self._decision_index.get(decision_id)
        if not record:
            logger.warning(f"Decision {decision_id[:8]} not found for outcome update")
            return None

        now = datetime.now(timezone.utc)

        # Update based on outcome
        if outcome == DecisionOutcome.RISK_VALIDATED:
            record.outcome = outcome
            record.risk_validated_at = now
            if adjusted_quantity is not None:
                record.risk_adjusted_quantity = adjusted_quantity

        elif outcome == DecisionOutcome.COMPLIANCE_VALIDATED:
            record.outcome = outcome
            record.compliance_validated_at = now

        elif outcome == DecisionOutcome.REJECTED:
            record.outcome = outcome
            record.rejection_reason = rejection_reason
            record.rejection_source = rejection_source
            self._total_rejected += 1

        elif outcome == DecisionOutcome.EXECUTED:
            record.outcome = outcome
            record.executed_at = now
            self._total_executed += 1
            # Track pipeline latency
            if record.timestamp:
                latency = (now - record.timestamp).total_seconds() * 1000
                self._pipeline_latencies.append(latency)

        elif outcome in (DecisionOutcome.CANCELLED, DecisionOutcome.EXPIRED):
            record.outcome = outcome

        else:
            record.outcome = outcome

        logger.debug(
            f"Updated decision {decision_id[:8]} outcome to {outcome.value}"
        )

        return record

    def update_from_validated_decision(
        self,
        validated: ValidatedDecisionEvent,
        validation_stage: str = "risk",
    ) -> DecisionRecord | None:
        """
        Update a decision from a ValidatedDecisionEvent.

        Args:
            validated: The ValidatedDecisionEvent
            validation_stage: "risk" or "compliance"

        Returns:
            Updated DecisionRecord or None if not found
        """
        decision_id = validated.original_decision_id
        record = self._decision_index.get(decision_id)

        if not record:
            logger.warning(
                f"Decision {decision_id[:8]} not found for validation update"
            )
            return None

        if validated.approved:
            if validation_stage == "risk":
                outcome = DecisionOutcome.RISK_VALIDATED
            else:
                outcome = DecisionOutcome.COMPLIANCE_VALIDATED

            return self.update_outcome(
                decision_id=decision_id,
                outcome=outcome,
                adjusted_quantity=validated.adjusted_quantity,
            )
        else:
            return self.update_outcome(
                decision_id=decision_id,
                outcome=DecisionOutcome.REJECTED,
                rejection_reason=validated.rejection_reason,
                rejection_source=validation_stage,
            )

    def link_order(self, decision_id: str, order_id: str) -> None:
        """
        Link an order ID to a decision for fill tracking.

        Args:
            decision_id: ID of the decision
            order_id: ID of the order created from this decision
        """
        record = self._decision_index.get(decision_id)
        if record:
            record.order_ids.append(order_id)
            self._order_to_decision[order_id] = decision_id
            logger.debug(f"Linked order {order_id[:8]} to decision {decision_id[:8]}")

    def link_fill(
        self,
        fill: FillEvent,
    ) -> DecisionRecord | None:
        """
        Link a fill event to its decision for P&L tracking.

        Args:
            fill: The FillEvent from broker

        Returns:
            Updated DecisionRecord or None if not linked
        """
        # Find decision by order_id
        decision_id = self._order_to_decision.get(fill.order_id)
        if not decision_id:
            logger.debug(f"No decision linked to order {fill.order_id[:8]}")
            return None

        record = self._decision_index.get(decision_id)
        if not record:
            logger.warning(f"Decision {decision_id[:8]} not found for fill linking")
            return None

        # Update fill details
        if record.fill_price is None:
            record.fill_price = fill.fill_price
            record.entry_price = fill.fill_price
        else:
            # Average fill price for multiple fills
            total_value = record.fill_price * record.fill_quantity + fill.fill_price * fill.filled_quantity
            record.fill_quantity += fill.filled_quantity
            record.fill_price = total_value / record.fill_quantity if record.fill_quantity > 0 else 0

        record.fill_quantity = record.fill_quantity or 0
        record.fill_quantity += fill.filled_quantity
        record.commission += fill.commission

        # Update outcome based on fill status
        if record.fill_quantity >= record.quantity:
            record.outcome = DecisionOutcome.EXECUTED
            record.executed_at = fill.timestamp
            self._total_executed += 1
            # Track pipeline latency
            latency = (fill.timestamp - record.timestamp).total_seconds() * 1000
            self._pipeline_latencies.append(latency)
        elif record.fill_quantity > 0:
            record.outcome = DecisionOutcome.PARTIAL

        logger.debug(
            f"Linked fill to decision {decision_id[:8]}: "
            f"{fill.filled_quantity} @ {fill.fill_price} "
            f"(total filled: {record.fill_quantity}/{record.quantity})"
        )

        return record

    def update_pnl(
        self,
        decision_id: str,
        realized_pnl: float | None = None,
        unrealized_pnl: float | None = None,
        exit_price: float | None = None,
    ) -> DecisionRecord | None:
        """
        Update P&L for a decision.

        Args:
            decision_id: ID of the decision
            realized_pnl: Realized P&L from closed position
            unrealized_pnl: Unrealized P&L for open position
            exit_price: Exit price if position was closed

        Returns:
            Updated DecisionRecord or None if not found
        """
        record = self._decision_index.get(decision_id)
        if not record:
            logger.warning(f"Decision {decision_id[:8]} not found for P&L update")
            return None

        if realized_pnl is not None:
            # Track P&L for win/loss calculations
            old_pnl = record.realized_pnl
            record.realized_pnl = realized_pnl
            record.pnl_impact = realized_pnl

            # Update profit/loss tracking
            if old_pnl is None:  # First time setting realized P&L
                if realized_pnl > 0:
                    self._total_profitable += 1
                    self._gross_profit += realized_pnl
                elif realized_pnl < 0:
                    self._total_unprofitable += 1
                    self._gross_loss += abs(realized_pnl)

        if unrealized_pnl is not None:
            record.unrealized_pnl = unrealized_pnl

        if exit_price is not None:
            record.exit_price = exit_price

        logger.debug(
            f"Updated P&L for decision {decision_id[:8]}: "
            f"realized={realized_pnl}, unrealized={unrealized_pnl}"
        )

        return record

    def get_decision(self, decision_id: str) -> DecisionRecord | None:
        """
        Get a decision by ID.

        Args:
            decision_id: ID of the decision

        Returns:
            DecisionRecord or None if not found
        """
        return self._decision_index.get(decision_id)

    def get_recent_decisions(self, limit: int = 100) -> list[DecisionRecord]:
        """
        Get the most recent decisions.

        Args:
            limit: Maximum number of decisions to return

        Returns:
            List of DecisionRecord, most recent first
        """
        # Convert deque to list and reverse to get most recent first
        decisions = list(self._decisions)
        decisions.reverse()
        return decisions[:limit]

    def get_decisions_by_symbol(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[DecisionRecord]:
        """
        Get recent decisions for a specific symbol.

        Args:
            symbol: Symbol to filter by
            limit: Maximum number of decisions to return

        Returns:
            List of matching DecisionRecord
        """
        matching = [d for d in self._decisions if d.symbol == symbol]
        matching.reverse()
        return matching[:limit]

    def get_decisions_by_outcome(
        self,
        outcome: DecisionOutcome,
        limit: int = 100,
    ) -> list[DecisionRecord]:
        """
        Get recent decisions with a specific outcome.

        Args:
            outcome: Outcome to filter by
            limit: Maximum number of decisions to return

        Returns:
            List of matching DecisionRecord
        """
        matching = [d for d in self._decisions if d.outcome == outcome]
        matching.reverse()
        return matching[:limit]

    def get_pending_decisions(self) -> list[DecisionRecord]:
        """
        Get all decisions that are still pending (not in terminal state).

        Returns:
            List of pending DecisionRecord
        """
        return [d for d in self._decisions if not d.is_terminal]

    def get_decision_stats(self) -> DecisionStatistics:
        """
        Calculate and return decision statistics.

        Returns:
            DecisionStatistics with current metrics
        """
        now = datetime.now(timezone.utc)
        stats = DecisionStatistics(last_updated=now)

        # Basic counts
        stats.total_decisions = self._total_decisions

        if not self._decisions:
            return stats

        # Decisions by action
        for d in self._decisions:
            action_key = d.action.value
            stats.decisions_by_action[action_key] = stats.decisions_by_action.get(action_key, 0) + 1

        # Decisions by symbol
        for d in self._decisions:
            symbol_key = d.symbol
            stats.decisions_by_symbol[symbol_key] = stats.decisions_by_symbol.get(symbol_key, 0) + 1

        # Decisions by outcome
        for d in self._decisions:
            outcome_key = d.outcome.value
            stats.decisions_by_outcome[outcome_key] = stats.decisions_by_outcome.get(outcome_key, 0) + 1

        # Win rate calculation
        total_with_pnl = self._total_profitable + self._total_unprofitable
        if total_with_pnl > 0:
            stats.win_rate = self._total_profitable / total_with_pnl

        # Average conviction
        convictions = [d.conviction_score for d in self._decisions if d.conviction_score > 0]
        if convictions:
            stats.avg_conviction = sum(convictions) / len(convictions)

        # P&L metrics
        stats.total_pnl = self._gross_profit - self._gross_loss

        decisions_with_pnl = [d for d in self._decisions if d.realized_pnl is not None]
        if decisions_with_pnl:
            stats.avg_pnl_per_decision = stats.total_pnl / len(decisions_with_pnl)
            pnls = [d.realized_pnl for d in decisions_with_pnl]
            stats.max_win = max(pnls) if pnls else 0.0
            stats.max_loss = min(pnls) if pnls else 0.0

        # Profit factor
        if self._gross_loss > 0:
            stats.profit_factor = self._gross_profit / self._gross_loss

        # Pipeline metrics
        latencies = list(self._pipeline_latencies)
        if latencies:
            stats.avg_pipeline_latency_ms = sum(latencies) / len(latencies)

        if self._total_decisions > 0:
            stats.rejection_rate = self._total_rejected / self._total_decisions
            stats.execution_rate = self._total_executed / self._total_decisions

        # Time-based metrics
        one_hour_ago = now - timedelta(hours=1)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        stats.decisions_last_hour = sum(
            1 for d in self._decisions if d.timestamp >= one_hour_ago
        )
        stats.decisions_today = sum(
            1 for d in self._decisions if d.timestamp >= today_start
        )

        # Conviction analysis
        winners = [d for d in decisions_with_pnl if d.realized_pnl is not None and d.realized_pnl > 0]
        losers = [d for d in decisions_with_pnl if d.realized_pnl is not None and d.realized_pnl < 0]

        if winners:
            stats.avg_conviction_winners = sum(d.conviction_score for d in winners) / len(winners)

        if losers:
            stats.avg_conviction_losers = sum(d.conviction_score for d in losers) / len(losers)

        # High conviction win rate
        high_conviction_threshold = 0.7
        high_conviction_decisions = [
            d for d in decisions_with_pnl
            if d.conviction_score >= high_conviction_threshold
        ]
        if high_conviction_decisions:
            high_conviction_wins = sum(
                1 for d in high_conviction_decisions
                if d.realized_pnl is not None and d.realized_pnl > 0
            )
            stats.high_conviction_win_rate = high_conviction_wins / len(high_conviction_decisions)

        return stats

    def expire_stale_decisions(self) -> int:
        """
        Mark stale pending decisions as expired.

        Returns:
            Number of decisions marked as expired
        """
        now = datetime.now(timezone.utc)
        expired_count = 0
        expiry_threshold = now - timedelta(seconds=self.DECISION_EXPIRY_SECONDS)

        for record in self._decisions:
            if record.outcome == DecisionOutcome.PENDING and record.timestamp < expiry_threshold:
                record.outcome = DecisionOutcome.EXPIRED
                expired_count += 1
                logger.debug(f"Decision {record.decision_id[:8]} marked as expired")

        if expired_count > 0:
            logger.info(f"Expired {expired_count} stale decisions")

        return expired_count

    def _cleanup_index(self) -> None:
        """Clean up index entries for decisions that have been evicted from buffer."""
        # Get IDs of decisions still in buffer
        active_ids = {d.decision_id for d in self._decisions}

        # Remove stale index entries
        stale_ids = [did for did in self._decision_index if did not in active_ids]
        for did in stale_ids:
            del self._decision_index[did]

        # Clean up order mapping
        stale_orders = [
            oid for oid, did in self._order_to_decision.items()
            if did not in active_ids
        ]
        for oid in stale_orders:
            del self._order_to_decision[oid]

    def clear(self) -> None:
        """Clear all tracked decisions and reset statistics."""
        self._decisions.clear()
        self._decision_index.clear()
        self._order_to_decision.clear()
        self._total_decisions = 0
        self._total_executed = 0
        self._total_rejected = 0
        self._total_profitable = 0
        self._total_unprofitable = 0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self._pipeline_latencies.clear()
        logger.info("DecisionTracker cleared")

    def to_dict(self) -> dict[str, Any]:
        """
        Export tracker state to dictionary for WebSocket streaming.

        Returns:
            Complete tracker state as dict
        """
        stats = self.get_decision_stats()
        recent = self.get_recent_decisions(limit=50)
        pending = self.get_pending_decisions()

        return {
            "statistics": stats.to_dict(),
            "recent_decisions": [d.to_dict() for d in recent],
            "pending_decisions": [d.to_dict() for d in pending],
            "total_decisions_tracked": self._total_decisions,
            "buffer_size": len(self._decisions),
            "buffer_max_size": self._max_decisions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_decision_pipeline_summary(self) -> dict[str, Any]:
        """
        Get a summary of decisions at each pipeline stage.

        Returns:
            Dict with counts and details by pipeline stage
        """
        stages = {
            "pending": [],
            "risk_validated": [],
            "compliance_validated": [],
            "executed": [],
            "rejected": [],
            "partial": [],
            "expired": [],
            "cancelled": [],
        }

        for d in self._decisions:
            stage_key = d.outcome.value
            if stage_key in stages:
                stages[stage_key].append({
                    "decision_id": d.decision_id[:8],
                    "symbol": d.symbol,
                    "action": d.action.value,
                    "quantity": d.quantity,
                    "timestamp": d.timestamp.isoformat(),
                })

        summary = {}
        for stage, decisions in stages.items():
            summary[stage] = {
                "count": len(decisions),
                "decisions": decisions[-10:],  # Last 10 per stage
            }

        return summary

    @property
    def total_decisions(self) -> int:
        """Total number of decisions recorded."""
        return self._total_decisions

    @property
    def buffer_size(self) -> int:
        """Current number of decisions in the buffer."""
        return len(self._decisions)

    @property
    def is_buffer_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self._decisions) >= self._max_decisions

"""
Event Persistence Layer
=======================

Persists unprocessed events to disk to prevent data loss on restart.
Addresses issue #S4: Event bus doesn't persist unprocessed events.

Features:
- SQLite-based persistence for durability
- Write-ahead logging for crash safety
- Automatic cleanup of processed events
- Recovery of unprocessed events on startup
- Support for event priority ordering
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Generator

from core.events import (
    Event,
    EventType,
    SignalDirection,
    OrderSide,
    OrderType,
    RiskAlertSeverity,
    OrderState,
    MarketDataEvent,
    SignalEvent,
    DecisionEvent,
    ValidatedDecisionEvent,
    OrderEvent,
    FillEvent,
    RiskAlertEvent,
    RollSignalEvent,
    RollCompleteEvent,
    SurveillanceAlertEvent,
    TransactionReportEvent,
    StressTestResultEvent,
    CorrelationAlertEvent,
    GreeksUpdateEvent,
    KillSwitchEvent,
    OrderStateChangeEvent,
)


logger = logging.getLogger(__name__)


class EventStatus(Enum):
    """Status of persisted event."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PersistedEvent:
    """Wrapper for persisted event with metadata."""
    id: int
    event_id: str
    event_type: str
    event_data: dict[str, Any]
    status: EventStatus
    created_at: datetime
    processed_at: datetime | None = None
    retry_count: int = 0
    priority: bool = False
    error_message: str | None = None


@dataclass
class PersistenceConfig:
    """Configuration for event persistence."""
    db_path: str = "data/events.db"
    enable_wal: bool = True  # Write-ahead logging for better performance
    checkpoint_interval_seconds: int = 300  # 5 minutes
    cleanup_completed_after_hours: int = 24
    max_retry_count: int = 3
    persist_market_data: bool = False  # Market data is high volume, usually skip
    batch_insert_size: int = 100


class EventPersistence:
    """
    Event persistence layer using SQLite.

    Provides durability for events to prevent loss on system restart.
    Uses WAL mode for better concurrent performance.
    """

    def __init__(self, config: PersistenceConfig | None = None):
        self._config = config or PersistenceConfig()
        self._db_path = Path(self._config.db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._initialized = False
        self._pending_batch: list[tuple[str, str, str, bool]] = []
        self._batch_lock = asyncio.Lock()

    def initialize(self) -> None:
        """Initialize the database and create tables if needed."""
        if self._initialized:
            return

        with self._get_connection() as conn:
            # Enable WAL mode for better concurrent access
            if self._config.enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")

            # Create events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persisted_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    processed_at TEXT,
                    retry_count INTEGER DEFAULT 0,
                    priority INTEGER DEFAULT 0,
                    error_message TEXT,
                    UNIQUE(event_id)
                )
            """)

            # Create indices for efficient queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_status
                ON persisted_events(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_created
                ON persisted_events(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_priority
                ON persisted_events(priority, created_at)
            """)

            conn.commit()

        self._initialized = True
        logger.info(f"Event persistence initialized at {self._db_path}")

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection (thread-safe)."""
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(
                    str(self._db_path),
                    check_same_thread=False,
                    timeout=30.0
                )
                self._conn.row_factory = sqlite3.Row
            yield self._conn

    def persist_event(self, event: Event, priority: bool = False) -> bool:
        """
        Persist an event to the database.

        Args:
            event: Event to persist
            priority: If True, mark as high priority

        Returns:
            True if persisted successfully, False otherwise
        """
        # Skip market data if configured (high volume)
        if (event.event_type == EventType.MARKET_DATA
            and not self._config.persist_market_data):
            return True

        try:
            event_data = event.to_audit_dict()
            event_json = json.dumps(event_data, default=str)

            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO persisted_events
                    (event_id, event_type, event_data, status, created_at, priority)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.event_type.value,
                    event_json,
                    EventStatus.PENDING.value,
                    datetime.now(timezone.utc).isoformat(),
                    1 if priority else 0
                ))
                conn.commit()

            logger.debug(f"Persisted event {event.event_id} ({event.event_type.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to persist event {event.event_id}: {e}")
            return False

    async def persist_event_async(self, event: Event, priority: bool = False) -> bool:
        """Async version of persist_event using run_in_executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.persist_event, event, priority)

    def mark_processing(self, event_id: str) -> bool:
        """Mark an event as being processed."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    UPDATE persisted_events
                    SET status = ?
                    WHERE event_id = ? AND status = ?
                """, (
                    EventStatus.PROCESSING.value,
                    event_id,
                    EventStatus.PENDING.value
                ))
                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to mark event {event_id} as processing: {e}")
            return False

    def mark_completed(self, event_id: str) -> bool:
        """Mark an event as completed (processed successfully)."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE persisted_events
                    SET status = ?, processed_at = ?
                    WHERE event_id = ?
                """, (
                    EventStatus.COMPLETED.value,
                    datetime.now(timezone.utc).isoformat(),
                    event_id
                ))
                conn.commit()

            logger.debug(f"Marked event {event_id} as completed")
            return True

        except Exception as e:
            logger.error(f"Failed to mark event {event_id} as completed: {e}")
            return False

    async def mark_completed_async(self, event_id: str) -> bool:
        """Async version of mark_completed."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.mark_completed, event_id)

    def mark_failed(self, event_id: str, error_message: str) -> bool:
        """Mark an event as failed with error message."""
        try:
            with self._get_connection() as conn:
                # Increment retry count
                conn.execute("""
                    UPDATE persisted_events
                    SET status = ?,
                        error_message = ?,
                        retry_count = retry_count + 1,
                        processed_at = ?
                    WHERE event_id = ?
                """, (
                    EventStatus.FAILED.value,
                    error_message,
                    datetime.now(timezone.utc).isoformat(),
                    event_id
                ))
                conn.commit()

            logger.warning(f"Marked event {event_id} as failed: {error_message}")
            return True

        except Exception as e:
            logger.error(f"Failed to mark event {event_id} as failed: {e}")
            return False

    def get_pending_events(
        self,
        limit: int = 100,
        include_failed: bool = False
    ) -> list[PersistedEvent]:
        """
        Get pending events for processing.

        Args:
            limit: Maximum number of events to return
            include_failed: Include failed events that can be retried

        Returns:
            List of persisted events ordered by priority and creation time
        """
        try:
            statuses = [EventStatus.PENDING.value]
            if include_failed:
                statuses.append(EventStatus.FAILED.value)

            placeholders = ','.join('?' * len(statuses))

            with self._get_connection() as conn:
                query = f"""
                    SELECT * FROM persisted_events
                    WHERE status IN ({placeholders})
                    AND (retry_count < ? OR status = ?)
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?
                """
                cursor = conn.execute(
                    query,
                    (*statuses, self._config.max_retry_count, EventStatus.PENDING.value, limit)
                )
                rows = cursor.fetchall()

            events = []
            for row in rows:
                events.append(PersistedEvent(
                    id=row['id'],
                    event_id=row['event_id'],
                    event_type=row['event_type'],
                    event_data=json.loads(row['event_data']),
                    status=EventStatus(row['status']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    processed_at=(
                        datetime.fromisoformat(row['processed_at'])
                        if row['processed_at'] else None
                    ),
                    retry_count=row['retry_count'],
                    priority=bool(row['priority']),
                    error_message=row['error_message']
                ))

            return events

        except Exception as e:
            logger.error(f"Failed to get pending events: {e}")
            return []

    def reconstruct_event(self, persisted: PersistedEvent) -> Event | None:
        """
        Reconstruct an Event object from persisted data.

        Args:
            persisted: Persisted event data

        Returns:
            Reconstructed Event or None if reconstruction fails
        """
        try:
            event_type = EventType(persisted.event_type)
            data = persisted.event_data

            # Common fields for all events
            base_fields = {
                'event_id': data.get('event_id'),
                'timestamp': datetime.fromisoformat(data.get('timestamp', '')),
                'source_agent': data.get('source_agent', 'unknown'),
            }

            # Reconstruct based on event type
            if event_type == EventType.MARKET_DATA:
                return MarketDataEvent(
                    **base_fields,
                    symbol=data.get('symbol', ''),
                    exchange=data.get('exchange', ''),
                    bid=data.get('bid', 0.0),
                    ask=data.get('ask', 0.0),
                    last=data.get('last', 0.0),
                    volume=data.get('volume', 0),
                    bid_size=data.get('bid_size', 0),
                    ask_size=data.get('ask_size', 0),
                    high=data.get('high', 0.0),
                    low=data.get('low', 0.0),
                    open_price=data.get('open', 0.0),
                    close=data.get('close', 0.0),
                )

            elif event_type == EventType.SIGNAL:
                return SignalEvent(
                    **base_fields,
                    strategy_name=data.get('strategy_name', ''),
                    symbol=data.get('symbol', ''),
                    direction=SignalDirection(data.get('direction', 'flat')),
                    strength=data.get('strength', 0.0),
                    confidence=data.get('confidence', 0.0),
                    target_price=data.get('target_price'),
                    stop_loss=data.get('stop_loss'),
                    rationale=data.get('rationale', ''),
                    data_sources=tuple(data.get('data_sources', [])),
                )

            elif event_type == EventType.DECISION:
                action = data.get('action')
                return DecisionEvent(
                    **base_fields,
                    symbol=data.get('symbol', ''),
                    action=OrderSide(action) if action else None,
                    quantity=data.get('quantity', 0),
                    order_type=OrderType(data.get('order_type', 'limit')),
                    limit_price=data.get('limit_price'),
                    stop_price=data.get('stop_price'),
                    rationale=data.get('rationale', ''),
                    contributing_signals=tuple(data.get('contributing_signals', [])),
                    data_sources=tuple(data.get('data_sources', [])),
                    conviction_score=data.get('conviction_score', 0.0),
                )

            elif event_type == EventType.VALIDATED_DECISION:
                return ValidatedDecisionEvent(
                    **base_fields,
                    original_decision_id=data.get('original_decision_id', ''),
                    approved=data.get('approved', False),
                    adjusted_quantity=data.get('adjusted_quantity'),
                    rejection_reason=data.get('rejection_reason'),
                    risk_metrics=data.get('risk_metrics', {}),
                    compliance_checks=tuple(data.get('compliance_checks', [])),
                )

            elif event_type == EventType.ORDER:
                return OrderEvent(
                    **base_fields,
                    decision_id=data.get('decision_id', ''),
                    validation_id=data.get('validation_id', ''),
                    symbol=data.get('symbol', ''),
                    side=OrderSide(data.get('side', 'buy')),
                    quantity=data.get('quantity', 0),
                    order_type=OrderType(data.get('order_type', 'limit')),
                    limit_price=data.get('limit_price'),
                    stop_price=data.get('stop_price'),
                    broker_order_id=data.get('broker_order_id'),
                    algo=data.get('algo', 'TWAP'),
                )

            elif event_type == EventType.FILL:
                return FillEvent(
                    **base_fields,
                    order_id=data.get('order_id', ''),
                    broker_order_id=data.get('broker_order_id', 0),
                    symbol=data.get('symbol', ''),
                    side=OrderSide(data.get('side', 'buy')),
                    filled_quantity=data.get('filled_quantity', 0),
                    fill_price=data.get('fill_price', 0.0),
                    commission=data.get('commission', 0.0),
                    exchange=data.get('exchange', ''),
                )

            elif event_type == EventType.RISK_ALERT:
                return RiskAlertEvent(
                    **base_fields,
                    severity=RiskAlertSeverity(data.get('severity', 'info')),
                    alert_type=data.get('alert_type', ''),
                    message=data.get('message', ''),
                    current_value=data.get('current_value', 0.0),
                    threshold_value=data.get('threshold_value', 0.0),
                    affected_symbols=tuple(data.get('affected_symbols', [])),
                    halt_trading=data.get('halt_trading', False),
                )

            elif event_type == EventType.ROLL_SIGNAL:
                return RollSignalEvent(
                    **base_fields,
                    symbol=data.get('symbol', ''),
                    from_contract=data.get('from_contract', ''),
                    to_contract=data.get('to_contract', ''),
                    days_to_expiry=data.get('days_to_expiry', 0),
                    roll_date=data.get('roll_date', ''),
                    urgency=data.get('urgency', 'normal'),
                )

            elif event_type == EventType.ROLL_COMPLETE:
                return RollCompleteEvent(
                    **base_fields,
                    symbol=data.get('symbol', ''),
                    from_contract=data.get('from_contract', ''),
                    to_contract=data.get('to_contract', ''),
                    price_adjustment=data.get('price_adjustment', 0.0),
                    rolled_quantity=data.get('rolled_quantity', 0),
                )

            elif event_type == EventType.SURVEILLANCE_ALERT:
                return SurveillanceAlertEvent(
                    **base_fields,
                    alert_type=data.get('alert_type', ''),
                    severity=data.get('severity', 'medium'),
                    symbol=data.get('symbol', ''),
                    description=data.get('description', ''),
                    evidence=tuple(data.get('evidence', [])),
                    requires_review=data.get('requires_review', True),
                )

            elif event_type == EventType.TRANSACTION_REPORT:
                return TransactionReportEvent(
                    **base_fields,
                    report_id=data.get('report_id', ''),
                    transaction_reference=data.get('transaction_reference', ''),
                    symbol=data.get('symbol', ''),
                    quantity=data.get('quantity', 0.0),
                    price=data.get('price', 0.0),
                    status=data.get('status', 'submitted'),
                    regulator=data.get('regulator', 'AMF'),
                )

            elif event_type == EventType.STRESS_TEST_RESULT:
                return StressTestResultEvent(
                    **base_fields,
                    scenario_id=data.get('scenario_id', ''),
                    scenario_name=data.get('scenario_name', ''),
                    pnl_impact=data.get('pnl_impact', 0.0),
                    pnl_impact_pct=data.get('pnl_impact_pct', 0.0),
                    passes_limit=data.get('passes_limit', True),
                    limit_breaches=tuple(data.get('limit_breaches', [])),
                )

            elif event_type == EventType.CORRELATION_ALERT:
                return CorrelationAlertEvent(
                    **base_fields,
                    alert_type=data.get('alert_type', ''),
                    previous_regime=data.get('previous_regime', ''),
                    new_regime=data.get('new_regime', ''),
                    average_correlation=data.get('average_correlation', 0.0),
                    max_pairwise_correlation=data.get('max_pairwise_correlation', 0.0),
                    affected_pairs=tuple(data.get('affected_pairs', [])),
                )

            elif event_type == EventType.GREEKS_UPDATE:
                return GreeksUpdateEvent(
                    **base_fields,
                    portfolio_delta=data.get('portfolio_delta', 0.0),
                    portfolio_gamma=data.get('portfolio_gamma', 0.0),
                    portfolio_vega=data.get('portfolio_vega', 0.0),
                    portfolio_theta=data.get('portfolio_theta', 0.0),
                    delta_limit_pct=data.get('delta_limit_pct', 0.0),
                    gamma_limit_pct=data.get('gamma_limit_pct', 0.0),
                    vega_limit_pct=data.get('vega_limit_pct', 0.0),
                    any_breach=data.get('any_breach', False),
                )

            elif event_type == EventType.KILL_SWITCH:
                return KillSwitchEvent(
                    **base_fields,
                    activated=data.get('activated', True),
                    reason=data.get('reason', ''),
                    trigger_type=data.get('trigger_type', 'manual'),
                    affected_symbols=tuple(data.get('affected_symbols', [])),
                    cancel_pending_orders=data.get('cancel_pending_orders', True),
                    close_positions=data.get('close_positions', False),
                )

            elif event_type == EventType.ORDER_STATE_CHANGE:
                return OrderStateChangeEvent(
                    **base_fields,
                    order_id=data.get('order_id', ''),
                    broker_order_id=data.get('broker_order_id'),
                    symbol=data.get('symbol', ''),
                    previous_state=OrderState(data.get('previous_state', 'created')),
                    new_state=OrderState(data.get('new_state', 'pending')),
                    reason=data.get('reason', ''),
                    filled_quantity=data.get('filled_quantity', 0),
                    remaining_quantity=data.get('remaining_quantity', 0),
                    avg_fill_price=data.get('avg_fill_price', 0.0),
                )

            else:
                # Generic event reconstruction
                return Event(
                    **base_fields,
                    event_type=event_type,
                )

        except Exception as e:
            logger.error(f"Failed to reconstruct event {persisted.event_id}: {e}")
            return None

    def cleanup_completed_events(self, hours: int | None = None) -> int:
        """
        Clean up completed events older than the specified hours.

        Args:
            hours: Hours to keep completed events (uses config default if None)

        Returns:
            Number of events deleted
        """
        hours = hours or self._config.cleanup_completed_after_hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM persisted_events
                    WHERE status = ? AND processed_at < ?
                """, (EventStatus.COMPLETED.value, cutoff.isoformat()))
                deleted = cursor.rowcount
                conn.commit()

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} completed events older than {hours}h")

            return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup completed events: {e}")
            return 0

    def get_statistics(self) -> dict[str, Any]:
        """Get persistence statistics."""
        try:
            with self._get_connection() as conn:
                # Count by status
                cursor = conn.execute("""
                    SELECT status, COUNT(*) as count
                    FROM persisted_events
                    GROUP BY status
                """)
                status_counts = {row['status']: row['count'] for row in cursor.fetchall()}

                # Count by event type
                cursor = conn.execute("""
                    SELECT event_type, COUNT(*) as count
                    FROM persisted_events
                    WHERE status = ?
                    GROUP BY event_type
                """, (EventStatus.PENDING.value,))
                pending_by_type = {row['event_type']: row['count'] for row in cursor.fetchall()}

                # Get oldest pending event
                cursor = conn.execute("""
                    SELECT MIN(created_at) as oldest
                    FROM persisted_events
                    WHERE status = ?
                """, (EventStatus.PENDING.value,))
                row = cursor.fetchone()
                oldest_pending = row['oldest'] if row else None

            return {
                'status_counts': status_counts,
                'pending_by_type': pending_by_type,
                'oldest_pending': oldest_pending,
                'db_path': str(self._db_path),
            }

        except Exception as e:
            logger.error(f"Failed to get persistence statistics: {e}")
            return {}

    def reset_stale_processing(self, timeout_minutes: int = 5) -> int:
        """
        Reset events stuck in 'processing' status back to 'pending'.

        This handles cases where a process crashed while processing an event.

        Args:
            timeout_minutes: Minutes after which processing is considered stale

        Returns:
            Number of events reset
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)

        try:
            with self._get_connection() as conn:
                # Find events stuck in processing for too long
                cursor = conn.execute("""
                    UPDATE persisted_events
                    SET status = ?, retry_count = retry_count + 1
                    WHERE status = ? AND created_at < ?
                """, (
                    EventStatus.PENDING.value,
                    EventStatus.PROCESSING.value,
                    cutoff.isoformat()
                ))
                reset_count = cursor.rowcount
                conn.commit()

            if reset_count > 0:
                logger.warning(f"Reset {reset_count} stale processing events to pending")

            return reset_count

        except Exception as e:
            logger.error(f"Failed to reset stale processing events: {e}")
            return 0

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
                logger.info("Event persistence closed")


# Global instance for convenience
_persistence_instance: EventPersistence | None = None


def get_event_persistence(config: PersistenceConfig | None = None) -> EventPersistence:
    """Get or create the global event persistence instance."""
    global _persistence_instance
    if _persistence_instance is None:
        _persistence_instance = EventPersistence(config)
        _persistence_instance.initialize()
    return _persistence_instance


def reset_event_persistence() -> None:
    """Reset the global event persistence instance (for testing)."""
    global _persistence_instance
    if _persistence_instance:
        _persistence_instance.close()
        _persistence_instance = None

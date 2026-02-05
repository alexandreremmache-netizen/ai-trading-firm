"""
Reconciliation Agent
====================

Position reconciliation for live trading safety.

Compares theoretical positions (tracked by CIO) against actual positions
from Interactive Brokers. Detects and alerts on discrepancies.

Features:
- Periodic reconciliation (default: every 60 seconds)
- Position quantity mismatch detection
- Unexpected position detection (in broker but not tracked)
- Missing position detection (tracked but not in broker)
- Cost basis drift monitoring
- Optional auto-correction (disabled by default)
- CRITICAL alert generation for significant discrepancies
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.agent_base import BaseAgent, AgentConfig
from core.events import Event, EventType, RiskAlertEvent

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger
    from core.broker import IBBroker

logger = logging.getLogger(__name__)


class DiscrepancySeverity(Enum):
    """Severity levels for position discrepancies."""
    INFO = "info"           # Minor drift, expected during fills
    WARNING = "warning"     # Moderate discrepancy, needs monitoring
    CRITICAL = "critical"   # Major discrepancy, requires immediate attention


class DiscrepancyType(Enum):
    """Types of position discrepancies."""
    QUANTITY_MISMATCH = "quantity_mismatch"    # Different quantities
    UNEXPECTED_POSITION = "unexpected"          # In broker but not tracked
    MISSING_POSITION = "missing"                # Tracked but not in broker
    COST_BASIS_DRIFT = "cost_basis_drift"       # Entry price difference
    SIDE_MISMATCH = "side_mismatch"             # Long vs short mismatch


@dataclass
class TheoreticalPosition:
    """A position as tracked by the CIO."""
    symbol: str
    quantity: float
    side: str  # "long" or "short"
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class BrokerPosition:
    """A position as reported by the broker."""
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class Discrepancy:
    """A detected position discrepancy."""
    discrepancy_id: str
    discrepancy_type: DiscrepancyType
    severity: DiscrepancySeverity
    symbol: str
    detected_at: datetime
    message: str

    # Theoretical vs actual values
    theoretical_quantity: Optional[float] = None
    actual_quantity: Optional[float] = None
    theoretical_side: Optional[str] = None
    actual_side: Optional[str] = None
    theoretical_cost: Optional[float] = None
    actual_cost: Optional[float] = None

    # Resolution
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "discrepancy_id": self.discrepancy_id,
            "type": self.discrepancy_type.value,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "detected_at": self.detected_at.isoformat(),
            "message": self.message,
            "theoretical_quantity": self.theoretical_quantity,
            "actual_quantity": self.actual_quantity,
            "theoretical_side": self.theoretical_side,
            "actual_side": self.actual_side,
            "resolved": self.resolved,
        }


@dataclass
class ReconciliationResult:
    """Result of a reconciliation run."""
    run_id: str
    run_time: datetime
    duration_ms: float
    positions_checked: int
    discrepancies_found: int
    critical_count: int
    warning_count: int
    info_count: int
    discrepancies: List[Discrepancy]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "run_time": self.run_time.isoformat(),
            "duration_ms": round(self.duration_ms, 2),
            "positions_checked": self.positions_checked,
            "discrepancies_found": self.discrepancies_found,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "discrepancies": [d.to_dict() for d in self.discrepancies],
        }


class ReconciliationAgent(BaseAgent):
    """
    Reconciliation Agent for position safety.

    Periodically compares theoretical positions from CIO against
    actual positions from Interactive Brokers.

    Safety features:
    - Detects partial fills that weren't tracked
    - Detects cancelled orders that CIO thinks went through
    - Detects positions in broker that we don't know about
    - Generates CRITICAL alerts for immediate attention

    Auto-correction (when enabled):
    - Can generate orders to correct position mismatches
    - Requires explicit opt-in (disabled by default)
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: "EventBus",
        audit_logger: "AuditLogger",
        broker: "IBBroker",
        get_theoretical_positions_fn=None,
    ):
        super().__init__(config, event_bus, audit_logger)
        self._broker = broker
        self._get_theoretical_positions = get_theoretical_positions_fn

        # Configuration
        self._reconcile_interval_seconds = config.parameters.get(
            "reconcile_interval_seconds", 60.0
        )
        self._auto_correct_enabled = config.parameters.get(
            "auto_correct_enabled", False  # Manual by default for safety
        )
        self._discrepancy_threshold_pct = config.parameters.get(
            "discrepancy_threshold_pct", 1.0  # 1% tolerance
        )
        self._cost_basis_threshold_pct = config.parameters.get(
            "cost_basis_threshold_pct", 5.0  # 5% cost basis tolerance
        )

        # State
        self._reconciliation_task: asyncio.Task | None = None
        self._running = False
        self._run_counter = 0

        # Discrepancy tracking (bounded)
        self._active_discrepancies: Dict[str, Discrepancy] = {}  # symbol -> active discrepancy
        self._discrepancy_history: deque[Discrepancy] = deque(maxlen=500)
        self._reconciliation_history: deque[ReconciliationResult] = deque(maxlen=100)

        # Statistics
        self._total_reconciliations = 0
        self._total_discrepancies_found = 0
        self._total_critical_alerts = 0
        self._total_auto_corrections = 0
        self._last_reconciliation: datetime | None = None

        logger.info(
            f"ReconciliationAgent initialized: "
            f"interval={self._reconcile_interval_seconds}s, "
            f"auto_correct={self._auto_correct_enabled}"
        )

    async def initialize(self) -> None:
        """Initialize reconciliation agent."""
        logger.info("ReconciliationAgent initializing")

    def get_subscribed_events(self) -> List[EventType]:
        """Reconciliation agent doesn't subscribe to events - runs on timer."""
        return []

    async def process_event(self, event: Event) -> None:
        """Reconciliation agent doesn't process events."""
        pass

    async def start_reconciliation_loop(self) -> None:
        """Start the background reconciliation loop."""
        if self._reconciliation_task is not None:
            logger.warning("Reconciliation loop already running")
            return

        self._running = True
        self._reconciliation_task = asyncio.create_task(
            self._reconciliation_loop()
        )
        logger.info("Reconciliation loop started")

    async def stop_reconciliation_loop(self) -> None:
        """Stop the reconciliation loop."""
        self._running = False
        if self._reconciliation_task is not None:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass
            self._reconciliation_task = None
            logger.info("Reconciliation loop stopped")

    async def _reconciliation_loop(self) -> None:
        """Background loop that runs periodic reconciliation."""
        while self._running:
            try:
                await self.reconcile_positions()
                await asyncio.sleep(self._reconcile_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in reconciliation loop: {e}")
                await asyncio.sleep(10.0)  # Back off on error

    async def reconcile_positions(self) -> ReconciliationResult:
        """
        Run a single reconciliation check.

        Compares theoretical positions from CIO against actual broker positions.

        Returns:
            ReconciliationResult with findings
        """
        start_time = datetime.now(timezone.utc)
        self._run_counter += 1
        run_id = f"recon_{start_time.strftime('%Y%m%d_%H%M%S')}_{self._run_counter}"

        discrepancies: List[Discrepancy] = []

        try:
            # Get theoretical positions from CIO
            theoretical = await self._get_theoretical_positions_safe()

            # Get actual positions from broker
            actual = await self._get_broker_positions_safe()

            # Build lookup maps
            theoretical_map = {p.symbol: p for p in theoretical}
            actual_map = {p.symbol: p for p in actual}

            # Check all theoretical positions
            for symbol, theo_pos in theoretical_map.items():
                if symbol in actual_map:
                    # Position exists in both - check for mismatches
                    actual_pos = actual_map[symbol]
                    disc = self._compare_positions(symbol, theo_pos, actual_pos)
                    if disc:
                        discrepancies.append(disc)
                else:
                    # Position tracked but not in broker
                    disc = self._create_missing_position_discrepancy(symbol, theo_pos)
                    discrepancies.append(disc)

            # Check for unexpected positions in broker
            for symbol, actual_pos in actual_map.items():
                if symbol not in theoretical_map:
                    disc = self._create_unexpected_position_discrepancy(symbol, actual_pos)
                    discrepancies.append(disc)

            # Process discrepancies
            for disc in discrepancies:
                await self._handle_discrepancy(disc)

        except Exception as e:
            logger.error(f"Reconciliation error: {e}")
            discrepancies.append(Discrepancy(
                discrepancy_id=f"{run_id}_error",
                discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
                severity=DiscrepancySeverity.CRITICAL,
                symbol="SYSTEM",
                detected_at=start_time,
                message=f"Reconciliation error: {e}",
            ))

        # Calculate stats
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        critical_count = sum(1 for d in discrepancies if d.severity == DiscrepancySeverity.CRITICAL)
        warning_count = sum(1 for d in discrepancies if d.severity == DiscrepancySeverity.WARNING)
        info_count = sum(1 for d in discrepancies if d.severity == DiscrepancySeverity.INFO)

        result = ReconciliationResult(
            run_id=run_id,
            run_time=start_time,
            duration_ms=duration_ms,
            positions_checked=len(theoretical) + len(actual),
            discrepancies_found=len(discrepancies),
            critical_count=critical_count,
            warning_count=warning_count,
            info_count=info_count,
            discrepancies=discrepancies,
        )

        # Update tracking
        self._total_reconciliations += 1
        self._total_discrepancies_found += len(discrepancies)
        self._total_critical_alerts += critical_count
        self._last_reconciliation = start_time
        self._reconciliation_history.append(result)

        # Log summary
        if discrepancies:
            logger.warning(
                f"Reconciliation {run_id}: {len(discrepancies)} discrepancies "
                f"(critical={critical_count}, warning={warning_count}, info={info_count})"
            )
        else:
            logger.debug(f"Reconciliation {run_id}: All positions match")

        return result

    async def _get_theoretical_positions_safe(self) -> List[TheoreticalPosition]:
        """Get theoretical positions with error handling."""
        if self._get_theoretical_positions is None:
            return []

        try:
            positions = await self._get_theoretical_positions()
            return positions if positions else []
        except Exception as e:
            logger.error(f"Failed to get theoretical positions: {e}")
            return []

    async def _get_broker_positions_safe(self) -> List[BrokerPosition]:
        """Get broker positions with error handling."""
        try:
            raw_positions = await self._broker.get_positions()

            positions = []
            for symbol, data in raw_positions.items():
                positions.append(BrokerPosition(
                    symbol=symbol,
                    quantity=data.get("position", 0),
                    avg_cost=data.get("avg_cost", 0),
                    market_value=data.get("market_value", 0),
                    unrealized_pnl=data.get("unrealized_pnl", 0),
                    realized_pnl=data.get("realized_pnl", 0),
                ))
            return positions

        except Exception as e:
            logger.error(f"Failed to get broker positions: {e}")
            return []

    def _compare_positions(
        self,
        symbol: str,
        theoretical: TheoreticalPosition,
        actual: BrokerPosition,
    ) -> Optional[Discrepancy]:
        """Compare theoretical vs actual position and return discrepancy if any."""
        now = datetime.now(timezone.utc)

        # Determine theoretical quantity with sign
        theo_qty = theoretical.quantity
        if theoretical.side == "short":
            theo_qty = -theo_qty

        # Check quantity mismatch
        # CRITICAL FIX: Use epsilon to avoid division by zero / numerical instability
        EPSILON = 1e-8
        if actual.quantity != 0 and abs(theo_qty) > EPSILON:
            pct_diff = abs(actual.quantity - theo_qty) / abs(theo_qty) * 100
            if pct_diff > self._discrepancy_threshold_pct:
                severity = (
                    DiscrepancySeverity.CRITICAL if pct_diff > 10
                    else DiscrepancySeverity.WARNING if pct_diff > 5
                    else DiscrepancySeverity.INFO
                )
                return Discrepancy(
                    discrepancy_id=f"qty_{symbol}_{now.timestamp()}",
                    discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
                    severity=severity,
                    symbol=symbol,
                    detected_at=now,
                    message=f"Quantity mismatch: theoretical={theo_qty}, actual={actual.quantity}",
                    theoretical_quantity=theo_qty,
                    actual_quantity=actual.quantity,
                )

        # Check side mismatch
        actual_side = "long" if actual.quantity > 0 else "short" if actual.quantity < 0 else "flat"
        if theoretical.side != actual_side and actual.quantity != 0:
            return Discrepancy(
                discrepancy_id=f"side_{symbol}_{now.timestamp()}",
                discrepancy_type=DiscrepancyType.SIDE_MISMATCH,
                severity=DiscrepancySeverity.CRITICAL,
                symbol=symbol,
                detected_at=now,
                message=f"Side mismatch: theoretical={theoretical.side}, actual={actual_side}",
                theoretical_side=theoretical.side,
                actual_side=actual_side,
            )

        # Check cost basis drift
        if theoretical.entry_price > 0 and actual.avg_cost > 0:
            cost_pct_diff = abs(theoretical.entry_price - actual.avg_cost) / theoretical.entry_price * 100
            if cost_pct_diff > self._cost_basis_threshold_pct:
                return Discrepancy(
                    discrepancy_id=f"cost_{symbol}_{now.timestamp()}",
                    discrepancy_type=DiscrepancyType.COST_BASIS_DRIFT,
                    severity=DiscrepancySeverity.INFO,
                    symbol=symbol,
                    detected_at=now,
                    message=f"Cost basis drift: theoretical={theoretical.entry_price:.2f}, actual={actual.avg_cost:.2f}",
                    theoretical_cost=theoretical.entry_price,
                    actual_cost=actual.avg_cost,
                )

        return None

    def _create_missing_position_discrepancy(
        self,
        symbol: str,
        theoretical: TheoreticalPosition,
    ) -> Discrepancy:
        """Create discrepancy for position tracked but missing from broker."""
        now = datetime.now(timezone.utc)
        return Discrepancy(
            discrepancy_id=f"missing_{symbol}_{now.timestamp()}",
            discrepancy_type=DiscrepancyType.MISSING_POSITION,
            severity=DiscrepancySeverity.CRITICAL,
            symbol=symbol,
            detected_at=now,
            message=f"Position tracked but not in broker: {theoretical.side} {theoretical.quantity}",
            theoretical_quantity=theoretical.quantity,
            actual_quantity=0,
            theoretical_side=theoretical.side,
        )

    def _create_unexpected_position_discrepancy(
        self,
        symbol: str,
        actual: BrokerPosition,
    ) -> Discrepancy:
        """Create discrepancy for position in broker but not tracked."""
        now = datetime.now(timezone.utc)
        side = "long" if actual.quantity > 0 else "short"
        return Discrepancy(
            discrepancy_id=f"unexpected_{symbol}_{now.timestamp()}",
            discrepancy_type=DiscrepancyType.UNEXPECTED_POSITION,
            severity=DiscrepancySeverity.WARNING,
            symbol=symbol,
            detected_at=now,
            message=f"Unexpected position in broker: {side} {abs(actual.quantity)} @ {actual.avg_cost}",
            actual_quantity=actual.quantity,
            actual_side=side,
        )

    async def _handle_discrepancy(self, discrepancy: Discrepancy) -> None:
        """Handle a detected discrepancy."""
        # Track active discrepancies
        self._active_discrepancies[discrepancy.symbol] = discrepancy
        self._discrepancy_history.append(discrepancy)

        # Emit alert for critical discrepancies
        if discrepancy.severity == DiscrepancySeverity.CRITICAL:
            await self._emit_critical_alert(discrepancy)

        # Auto-correct if enabled
        if self._auto_correct_enabled and discrepancy.severity == DiscrepancySeverity.CRITICAL:
            await self._attempt_auto_correction(discrepancy)

    async def _emit_critical_alert(self, discrepancy: Discrepancy) -> None:
        """Emit a critical alert event for the discrepancy."""
        alert = RiskAlertEvent(
            alert_type="RECONCILIATION_CRITICAL",
            message=f"[RECONCILIATION] {discrepancy.message}",
            severity="critical",
            source_agent=self._config.name,
            metadata={
                "discrepancy_id": discrepancy.discrepancy_id,
                "discrepancy_type": discrepancy.discrepancy_type.value,
                "symbol": discrepancy.symbol,
            },
        )
        await self._event_bus.publish(alert, priority=True)
        logger.critical(f"RECONCILIATION ALERT: {discrepancy.message}")

    async def _attempt_auto_correction(self, discrepancy: Discrepancy) -> bool:
        """
        Attempt to auto-correct a discrepancy.

        Currently only generates alerts - actual correction orders
        require manual review.
        """
        # For safety, we log the recommended correction but don't execute
        if discrepancy.discrepancy_type == DiscrepancyType.QUANTITY_MISMATCH:
            correction_qty = (discrepancy.actual_quantity or 0) - (discrepancy.theoretical_quantity or 0)
            logger.warning(
                f"AUTO-CORRECT RECOMMENDED for {discrepancy.symbol}: "
                f"Adjust position by {correction_qty} to match broker"
            )

        elif discrepancy.discrepancy_type == DiscrepancyType.MISSING_POSITION:
            logger.warning(
                f"AUTO-CORRECT RECOMMENDED for {discrepancy.symbol}: "
                f"Position exists in tracking but not in broker - verify and remove from tracking"
            )

        elif discrepancy.discrepancy_type == DiscrepancyType.UNEXPECTED_POSITION:
            logger.warning(
                f"AUTO-CORRECT RECOMMENDED for {discrepancy.symbol}: "
                f"Unexpected position in broker - verify and add to tracking or close"
            )

        self._total_auto_corrections += 1
        return False  # Return False as we don't actually execute corrections

    def resolve_discrepancy(self, symbol: str, resolution_method: str = "manual") -> bool:
        """
        Mark a discrepancy as resolved.

        Args:
            symbol: Symbol to resolve
            resolution_method: How it was resolved

        Returns:
            True if discrepancy was found and resolved
        """
        if symbol not in self._active_discrepancies:
            return False

        discrepancy = self._active_discrepancies[symbol]
        discrepancy.resolved = True
        discrepancy.resolved_at = datetime.now(timezone.utc)
        discrepancy.resolution_method = resolution_method

        del self._active_discrepancies[symbol]
        logger.info(f"Discrepancy resolved for {symbol}: {resolution_method}")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get reconciliation statistics."""
        return {
            "total_reconciliations": self._total_reconciliations,
            "total_discrepancies_found": self._total_discrepancies_found,
            "total_critical_alerts": self._total_critical_alerts,
            "total_auto_corrections": self._total_auto_corrections,
            "active_discrepancies": len(self._active_discrepancies),
            "last_reconciliation": self._last_reconciliation.isoformat() if self._last_reconciliation else None,
            "reconcile_interval_seconds": self._reconcile_interval_seconds,
            "auto_correct_enabled": self._auto_correct_enabled,
            "loop_running": self._reconciliation_task is not None,
        }

    def get_active_discrepancies(self) -> List[Dict[str, Any]]:
        """Get list of currently active discrepancies."""
        return [d.to_dict() for d in self._active_discrepancies.values()]

    def get_recent_reconciliations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent reconciliation results."""
        recent = list(self._reconciliation_history)[-limit:]
        return [r.to_dict() for r in recent]

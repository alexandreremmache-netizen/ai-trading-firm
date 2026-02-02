"""
Audit Logger
============

Comprehensive logging for regulatory compliance (EU/AMF).
All decisions must be logged with timestamp, data sources, rationale, and responsible agent.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any

from core.events import Event


logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: str
    entry_type: str
    agent_name: str
    event_id: str | None
    details: dict[str, Any]

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)


class AuditLogger:
    """
    Audit logger for regulatory compliance.

    Requirements (EU/AMF):
    - All decisions logged with timestamp
    - Data sources recorded
    - Rationale documented
    - Responsible agent identified
    - Retention: 7 years (MiFID II)

    Features:
    - Rotating file handlers to prevent unbounded log growth
    - Configurable max file size and backup count
    - Both size-based and time-based rotation options
    """

    # Default rotation settings
    DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50 MB per file
    DEFAULT_BACKUP_COUNT = 100  # Keep 100 backup files (for 7 year retention)
    DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB for system.log
    DEFAULT_LOG_BACKUP_COUNT = 30  # Keep 30 backup files for system logs

    def __init__(
        self,
        audit_file: str = "logs/audit.jsonl",
        trade_file: str = "logs/trades.jsonl",
        decision_file: str = "logs/decisions.jsonl",
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        log_max_bytes: int = DEFAULT_LOG_MAX_BYTES,
        log_backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
    ):
        self._audit_file = Path(audit_file)
        self._trade_file = Path(trade_file)
        self._decision_file = Path(decision_file)

        # Rotation settings
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._log_max_bytes = log_max_bytes
        self._log_backup_count = log_backup_count

        # File handles for rotation checking
        self._file_sizes: dict[Path, int] = {}

        # Ensure directories exist
        for file_path in [self._audit_file, self._trade_file, self._decision_file]:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        self._setup_python_logging()

    def _setup_python_logging(self) -> None:
        """
        Configure Python logging with rotation.

        Uses RotatingFileHandler to prevent unbounded log growth.
        """
        log_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

        # Ensure logs directory exists
        Path("logs").mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        file_handler = RotatingFileHandler(
            "logs/system.log",
            maxBytes=self._log_max_bytes,
            backupCount=self._log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        console_handler.setLevel(logging.INFO)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        root_logger.handlers = []

        # Add handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        logger.info(
            f"Logging initialized with rotation: "
            f"max_size={self._log_max_bytes/1024/1024:.0f}MB, "
            f"backups={self._log_backup_count}"
        )

    def _write_entry(self, file_path: Path, entry: AuditEntry) -> None:
        """
        Write entry to JSONL file with rotation support.

        Checks file size and rotates if necessary.
        """
        try:
            # Check if rotation is needed
            self._check_rotation(file_path)

            with open(file_path, "a", encoding="utf-8") as f:
                line = entry.to_json() + "\n"
                f.write(line)

                # Update tracked file size
                self._file_sizes[file_path] = self._file_sizes.get(file_path, 0) + len(line.encode('utf-8'))

        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")

    def _check_rotation(self, file_path: Path) -> None:
        """
        Check if file needs rotation and perform if necessary.

        Files are rotated when they exceed max_bytes.
        Old files are renamed with numeric suffixes: file.1.jsonl, file.2.jsonl, etc.
        """
        if not file_path.exists():
            self._file_sizes[file_path] = 0
            return

        # Get current file size
        current_size = file_path.stat().st_size
        self._file_sizes[file_path] = current_size

        if current_size < self._max_bytes:
            return

        # Need to rotate
        logger.info(f"Rotating log file: {file_path} (size={current_size/1024/1024:.1f}MB)")

        # Rotate existing backups
        for i in range(self._backup_count - 1, 0, -1):
            src = file_path.with_suffix(f".{i}.jsonl")
            dst = file_path.with_suffix(f".{i+1}.jsonl")

            if src.exists():
                if i + 1 > self._backup_count:
                    # Delete oldest backup
                    src.unlink()
                else:
                    # Rename to next number
                    if dst.exists():
                        dst.unlink()
                    src.rename(dst)

        # Rename current file to .1
        backup_path = file_path.with_suffix(".1.jsonl")
        if backup_path.exists():
            backup_path.unlink()
        file_path.rename(backup_path)

        # Reset size tracking
        self._file_sizes[file_path] = 0

    def get_log_stats(self) -> dict[str, Any]:
        """Get statistics about log files."""
        stats = {}

        for name, file_path in [
            ("audit", self._audit_file),
            ("trade", self._trade_file),
            ("decision", self._decision_file),
        ]:
            if file_path.exists():
                size = file_path.stat().st_size
                # Count backup files
                backup_count = len(list(file_path.parent.glob(f"{file_path.stem}.*.jsonl")))

                stats[name] = {
                    "current_size_mb": size / 1024 / 1024,
                    "max_size_mb": self._max_bytes / 1024 / 1024,
                    "usage_pct": (size / self._max_bytes) * 100,
                    "backup_count": backup_count,
                    "max_backups": self._backup_count,
                }
            else:
                stats[name] = {"exists": False}

        return stats

    def log_event(self, event: Event) -> None:
        """Log any event."""
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="event",
            agent_name=event.source_agent,
            event_id=event.event_id,
            details=event.to_audit_dict(),
        )
        self._write_entry(self._audit_file, entry)

    def log_decision(
        self,
        agent_name: str,
        decision_id: str,
        symbol: str,
        action: str,
        quantity: int,
        rationale: str,
        data_sources: list[str],
        contributing_signals: list[str],
        conviction_score: float,
    ) -> None:
        """
        Log a trading decision with full compliance data.

        This is a critical audit function - all decisions MUST be logged.
        """
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="decision",
            agent_name=agent_name,
            event_id=decision_id,
            details={
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "rationale": rationale,
                "data_sources": data_sources,
                "contributing_signals": contributing_signals,
                "conviction_score": conviction_score,
            },
        )
        self._write_entry(self._decision_file, entry)
        logger.info(f"Decision logged: {agent_name} - {action} {quantity} {symbol}")

    def log_trade(
        self,
        agent_name: str,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        commission: float,
        decision_id: str,
    ) -> None:
        """
        Log a trade execution.

        Links back to the original decision for audit trail.
        """
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="trade",
            agent_name=agent_name,
            event_id=order_id,
            details={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "commission": commission,
                "decision_id": decision_id,
                "total_value": quantity * price,
            },
        )
        self._write_entry(self._trade_file, entry)
        logger.info(f"Trade logged: {side} {quantity} {symbol} @ {price}")

    def log_risk_alert(
        self,
        agent_name: str,
        alert_type: str,
        severity: str,
        message: str,
        current_value: float,
        threshold_value: float,
        halt_trading: bool,
    ) -> None:
        """Log a risk alert."""
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="risk_alert",
            agent_name=agent_name,
            event_id=None,
            details={
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "current_value": current_value,
                "threshold_value": threshold_value,
                "halt_trading": halt_trading,
            },
        )
        self._write_entry(self._audit_file, entry)

        if halt_trading:
            logger.critical(f"TRADING HALTED: {message}")
        else:
            logger.warning(f"Risk alert: {message}")

    def log_compliance_check(
        self,
        decision_id: str,
        agent_name: str,
        approved: bool,
        checks: list[dict[str, Any]],
        rejection_code: str | None = None,
        rejection_reason: str | None = None,
    ) -> None:
        """
        Log a compliance check result.

        Required for EU/AMF regulatory compliance - all compliance
        decisions must be fully documented.
        """
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="compliance_check",
            agent_name=agent_name,
            event_id=decision_id,
            details={
                "approved": approved,
                "checks": checks,
                "rejection_code": rejection_code,
                "rejection_reason": rejection_reason,
            },
        )
        self._write_entry(self._audit_file, entry)

        if not approved:
            logger.warning(
                f"Compliance REJECTED {decision_id}: {rejection_code} - {rejection_reason}"
            )

    def log_agent_event(
        self,
        agent_name: str,
        event_type: str,
        details: dict[str, Any],
    ) -> None:
        """Log an agent lifecycle event."""
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type=f"agent_{event_type}",
            agent_name=agent_name,
            event_id=None,
            details=details,
        )
        self._write_entry(self._audit_file, entry)

    def log_system_event(
        self,
        event_type: str,
        details: dict[str, Any],
    ) -> None:
        """Log a system-level event."""
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type=f"system_{event_type}",
            agent_name="system",
            event_id=None,
            details=details,
        )
        self._write_entry(self._audit_file, entry)
        logger.info(f"System event: {event_type}")

    def get_decisions(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbol: str | None = None,
    ) -> list[dict]:
        """
        Query decision history for audit.

        Returns decisions matching the criteria.
        """
        decisions = []

        try:
            with open(self._decision_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)

                    # Filter by date
                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    if start_date and entry_time < start_date:
                        continue
                    if end_date and entry_time > end_date:
                        continue

                    # Filter by symbol
                    if symbol and entry["details"].get("symbol") != symbol:
                        continue

                    decisions.append(entry)

        except FileNotFoundError:
            pass

        return decisions

    def get_trades(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict]:
        """Query trade history for audit."""
        trades = []

        try:
            with open(self._trade_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)

                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    if start_date and entry_time < start_date:
                        continue
                    if end_date and entry_time > end_date:
                        continue

                    trades.append(entry)

        except FileNotFoundError:
            pass

        return trades

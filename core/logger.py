"""
Audit Logger
============

Comprehensive logging for regulatory compliance (EU/AMF).
All decisions must be logged with timestamp, data sources, rationale, and responsible agent.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any

from core.events import Event


logger = logging.getLogger(__name__)

# P2: Correlation ID tracking across async calls
# Context variable for correlation IDs that propagates across async boundaries
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    'correlation_id', default=None
)


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context."""
    return _correlation_id.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """
    Set a correlation ID in context. Generates one if not provided.

    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())[:8]
    _correlation_id.set(correlation_id)
    return correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID from context."""
    _correlation_id.set(None)


class CorrelationIdFilter(logging.Filter):
    """
    Logging filter that adds correlation ID to log records.

    P2: Enables tracking of related log entries across async calls.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id to log record."""
        record.correlation_id = get_correlation_id() or "-"
        return True


class StructuredJsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    P2: Enables machine-readable log output for log aggregation systems.

    Output format:
    {
        "timestamp": "2024-01-15T10:30:00.123456Z",
        "level": "INFO",
        "logger": "core.logger",
        "message": "Log message",
        "correlation_id": "abc12345",
        "extra": {...}
    }
    """

    def __init__(
        self,
        include_extra: bool = True,
        timestamp_format: str = "iso",
    ):
        """
        Initialize JSON formatter.

        Args:
            include_extra: Include extra fields from log record
            timestamp_format: "iso" for ISO8601, "epoch" for Unix timestamp
        """
        super().__init__()
        self.include_extra = include_extra
        self.timestamp_format = timestamp_format

        # Standard LogRecord attributes to exclude from extra
        self._standard_attrs = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'message', 'correlation_id', 'taskName',
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Get timestamp
        if self.timestamp_format == "iso":
            timestamp = datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat()
        else:
            timestamp = record.created

        # Build base log entry
        log_entry = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, 'correlation_id', None),
        }

        # Add location info for errors
        if record.levelno >= logging.WARNING:
            log_entry["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if self.include_extra:
            extra = {}
            for key, value in record.__dict__.items():
                if key not in self._standard_attrs:
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        extra[key] = value
                    except (TypeError, ValueError):
                        extra[key] = str(value)

            if extra:
                log_entry["extra"] = extra

        return json.dumps(log_entry, default=str)


@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: str
    entry_type: str
    agent_name: str
    event_id: str | None
    details: dict[str, Any]
    correlation_id: str | None = None  # P2: Track correlation across async calls

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

    # Async write settings
    FLUSH_INTERVAL_MS = 100  # Flush every 100ms
    FLUSH_BATCH_SIZE = 100   # Or every 100 entries
    CONSECUTIVE_FAILURE_ALERT_THRESHOLD = 5  # Alert after 5 consecutive failures

    def __init__(
        self,
        audit_file: str = "logs/audit.jsonl",
        trade_file: str = "logs/trades.jsonl",
        decision_file: str = "logs/decisions.jsonl",
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        log_max_bytes: int = DEFAULT_LOG_MAX_BYTES,
        log_backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
        use_json_logging: bool = False,
    ):
        self._audit_file = Path(audit_file)
        self._trade_file = Path(trade_file)
        self._decision_file = Path(decision_file)

        # Rotation settings
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._log_max_bytes = log_max_bytes
        self._log_backup_count = log_backup_count

        # P2: JSON logging option
        self._use_json_logging = use_json_logging

        # File handles for rotation checking
        self._file_sizes: dict[Path, int] = {}

        # MON-001: Write metrics counters
        self._write_success: int = 0
        self._write_failures: int = 0
        self._consecutive_failures: int = 0

        # RT-P0-1: Async write queue and background writer
        self._write_queue: asyncio.Queue[tuple[Path, AuditEntry]] | None = None
        self._writer_task: asyncio.Task | None = None
        self._running: bool = False

        # Ensure directories exist
        for file_path in [self._audit_file, self._trade_file, self._decision_file]:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        self._setup_python_logging()

    def _setup_python_logging(self) -> None:
        """
        Configure Python logging with rotation.

        Uses RotatingFileHandler to prevent unbounded log growth.
        P2: Supports structured JSON logging and correlation ID tracking.
        """
        # Ensure logs directory exists
        Path("logs").mkdir(parents=True, exist_ok=True)

        # P2: Create correlation ID filter
        correlation_filter = CorrelationIdFilter()

        # Create rotating file handler
        file_handler = RotatingFileHandler(
            "logs/system.log",
            maxBytes=self._log_max_bytes,
            backupCount=self._log_backup_count,
            encoding="utf-8",
        )
        file_handler.addFilter(correlation_filter)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.addFilter(correlation_filter)
        console_handler.setLevel(logging.INFO)

        # P2: Choose formatter based on configuration
        if self._use_json_logging:
            json_formatter = StructuredJsonFormatter()
            file_handler.setFormatter(json_formatter)
            console_handler.setFormatter(json_formatter)
        else:
            # Include correlation_id in standard format
            log_format = "%(asctime)s | %(correlation_id)s | %(name)s | %(levelname)s | %(message)s"
            file_handler.setFormatter(logging.Formatter(log_format))
            console_handler.setFormatter(logging.Formatter(log_format))

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        root_logger.handlers = []

        # Add handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        log_mode = "JSON" if self._use_json_logging else "text"
        logger.info(
            f"Logging initialized with rotation: "
            f"max_size={self._log_max_bytes/1024/1024:.0f}MB, "
            f"backups={self._log_backup_count}, "
            f"format={log_mode}"
        )

    async def start_async_writer(self) -> None:
        """
        Start the async background writer task.

        RT-P0-1: Non-blocking I/O for audit log writes.
        Call this when starting the system to enable async writes.
        """
        if self._running:
            return

        self._write_queue = asyncio.Queue()
        self._running = True
        self._writer_task = asyncio.create_task(self._background_writer())
        logger.info("Async audit log writer started")

    async def stop_async_writer(self) -> None:
        """Stop the async background writer and flush remaining entries."""
        if not self._running:
            return

        self._running = False

        # Flush remaining entries
        if self._write_queue:
            while not self._write_queue.empty():
                try:
                    file_path, entry = self._write_queue.get_nowait()
                    self._write_entry_sync(file_path, entry)
                except asyncio.QueueEmpty:
                    break

        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Async audit log writer stopped. Stats: success={self._write_success}, failures={self._write_failures}")

    async def _background_writer(self) -> None:
        """
        Background task that flushes write queue every 100ms or 100 entries.

        RT-P0-1: Decouples file I/O from event dispatch path.
        """
        batch: list[tuple[Path, AuditEntry]] = []
        last_flush = asyncio.get_event_loop().time()

        while self._running:
            try:
                # Wait for entry with timeout
                try:
                    entry = await asyncio.wait_for(
                        self._write_queue.get(),
                        timeout=self.FLUSH_INTERVAL_MS / 1000.0
                    )
                    batch.append(entry)
                except asyncio.TimeoutError:
                    pass

                # Flush if batch is full or timeout elapsed
                current_time = asyncio.get_event_loop().time()
                time_since_flush = (current_time - last_flush) * 1000  # ms

                if len(batch) >= self.FLUSH_BATCH_SIZE or time_since_flush >= self.FLUSH_INTERVAL_MS:
                    if batch:
                        await self._flush_batch(batch)
                        batch = []
                        last_flush = current_time

            except asyncio.CancelledError:
                # Flush remaining on shutdown
                if batch:
                    await self._flush_batch(batch)
                raise
            except Exception as e:
                logger.error(f"Background writer error: {e}")
                await asyncio.sleep(0.1)

    async def _flush_batch(self, batch: list[tuple[Path, AuditEntry]]) -> None:
        """Flush a batch of entries to disk."""
        # Run blocking I/O in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._flush_batch_sync, batch)

    def _flush_batch_sync(self, batch: list[tuple[Path, AuditEntry]]) -> None:
        """Synchronously flush batch of entries (runs in thread pool)."""
        for file_path, entry in batch:
            self._write_entry_sync(file_path, entry)

    def _write_entry(self, file_path: Path, entry: AuditEntry) -> None:
        """
        Write entry to JSONL file - queues for async write if available.

        RT-P0-1: Uses async queue when background writer is running,
        falls back to sync write otherwise.
        """
        if self._running and self._write_queue:
            # Non-blocking: queue for async write
            try:
                self._write_queue.put_nowait((file_path, entry))
            except asyncio.QueueFull:
                # Queue full - fall back to sync write
                logger.warning("Audit write queue full, falling back to sync write")
                self._write_entry_sync(file_path, entry)
        else:
            # Sync fallback when async writer not running
            self._write_entry_sync(file_path, entry)

    def _write_entry_sync(self, file_path: Path, entry: AuditEntry) -> None:
        """
        Synchronously write entry to JSONL file with rotation support.

        MON-001: Tracks success/failure metrics and alerts on consecutive failures.
        """
        try:
            # Check if rotation is needed
            self._check_rotation(file_path)

            with open(file_path, "a", encoding="utf-8") as f:
                line = entry.to_json() + "\n"
                f.write(line)

                # Update tracked file size
                self._file_sizes[file_path] = self._file_sizes.get(file_path, 0) + len(line.encode('utf-8'))

            # MON-001: Track success
            self._write_success += 1
            self._consecutive_failures = 0

        except Exception as e:
            # MON-001: Track failure and alert after threshold
            self._write_failures += 1
            self._consecutive_failures += 1

            logger.error(f"Failed to write audit entry: {e}")

            if self._consecutive_failures >= self.CONSECUTIVE_FAILURE_ALERT_THRESHOLD:
                logger.critical(
                    f"ALERT: {self._consecutive_failures} consecutive audit log write failures! "
                    f"Total failures: {self._write_failures}, Success: {self._write_success}"
                )

    def get_write_metrics(self) -> dict[str, int]:
        """Get audit log write metrics (MON-001)."""
        return {
            "write_success": self._write_success,
            "write_failures": self._write_failures,
            "consecutive_failures": self._consecutive_failures,
            "queue_size": self._write_queue.qsize() if self._write_queue else 0,
        }

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
            correlation_id=get_correlation_id(),  # P2: Track correlation
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
            correlation_id=get_correlation_id(),  # P2: Track correlation
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
            correlation_id=get_correlation_id(),  # P2: Track correlation
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
            correlation_id=get_correlation_id(),  # P2: Track correlation
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
            correlation_id=get_correlation_id(),  # P2: Track correlation
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
            correlation_id=get_correlation_id(),  # P2: Track correlation
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
            correlation_id=get_correlation_id(),  # P2: Track correlation
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
            logger.debug(f"Decision file not found: {self._decision_file}")

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
            logger.debug(f"Trade file not found: {self._trade_file}")

        return trades

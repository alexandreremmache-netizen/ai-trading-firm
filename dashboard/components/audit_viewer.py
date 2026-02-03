"""
Audit Viewer
============

View and search decision audit logs for compliance reporting and analysis.

Features:
- Load and parse audit logs (audit.jsonl, trades.jsonl, decisions.jsonl)
- Full-text search in rationale and details
- Filter by time range, entry type, agent, symbol, action
- Link related entries (decision -> risk_check -> compliance_check -> execution -> fill)
- Export for audit table display and compliance reporting
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class AuditEntryType(Enum):
    """Types of audit log entries."""
    DECISION = "decision"
    TRADE = "trade"
    RISK_ALERT = "risk_alert"
    COMPLIANCE_CHECK = "compliance_check"
    EVENT = "event"
    AGENT_START = "agent_start"
    AGENT_STOP = "agent_stop"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> AuditEntryType:
        """Convert string to AuditEntryType, handling prefixed types."""
        # Handle exact matches
        for member in cls:
            if member.value == value:
                return member

        # Handle prefixed types (e.g., "agent_start", "system_shutdown")
        if value.startswith("agent_"):
            if "start" in value:
                return cls.AGENT_START
            elif "stop" in value or "shutdown" in value:
                return cls.AGENT_STOP
            return cls.EVENT

        if value.startswith("system_"):
            if "start" in value:
                return cls.SYSTEM_START
            elif "stop" in value or "shutdown" in value:
                return cls.SYSTEM_STOP
            return cls.EVENT

        return cls.UNKNOWN


@dataclass
class AuditEntry:
    """
    Single audit log entry with normalized structure.

    Represents any entry from the audit log files with common fields
    extracted for easy querying and display.
    """
    entry_id: str  # event_id from log or generated
    timestamp: datetime
    entry_type: AuditEntryType
    agent_name: str
    action: str | None  # buy/sell/hold for decisions/trades
    symbol: str | None
    details: dict[str, Any]
    data_sources: list[str]
    rationale: str | None
    correlation_id: str | None = None
    related_entries: list[str] = field(default_factory=list)  # IDs of related entries

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "entry_type": self.entry_type.value,
            "agent_name": self.agent_name,
            "action": self.action,
            "symbol": self.symbol,
            "details": self.details,
            "data_sources": self.data_sources,
            "rationale": self.rationale,
            "correlation_id": self.correlation_id,
            "related_entries": self.related_entries,
        }

    def matches_text(self, search_text: str) -> bool:
        """Check if entry matches full-text search."""
        search_lower = search_text.lower()

        # Search in rationale
        if self.rationale and search_lower in self.rationale.lower():
            return True

        # Search in agent name
        if search_lower in self.agent_name.lower():
            return True

        # Search in symbol
        if self.symbol and search_lower in self.symbol.lower():
            return True

        # Search in action
        if self.action and search_lower in self.action.lower():
            return True

        # Search in details (convert to string)
        details_str = json.dumps(self.details, default=str).lower()
        if search_lower in details_str:
            return True

        # Search in data sources
        for source in self.data_sources:
            if search_lower in source.lower():
                return True

        return False


@dataclass
class AuditFilter:
    """
    Filter criteria for searching audit entries.

    All filters are optional - unset filters match all entries.
    """
    start_time: datetime | None = None
    end_time: datetime | None = None
    entry_types: list[AuditEntryType] | None = None
    agents: list[str] | None = None
    symbols: list[str] | None = None
    actions: list[str] | None = None
    search_text: str | None = None  # Full-text search
    correlation_id: str | None = None
    decision_id: str | None = None  # Find entries related to specific decision

    def matches(self, entry: AuditEntry) -> bool:
        """Check if an entry matches this filter."""
        # Time range filter
        if self.start_time and entry.timestamp < self.start_time:
            return False
        if self.end_time and entry.timestamp > self.end_time:
            return False

        # Entry type filter
        if self.entry_types and entry.entry_type not in self.entry_types:
            return False

        # Agent filter
        if self.agents and entry.agent_name not in self.agents:
            return False

        # Symbol filter
        if self.symbols:
            if not entry.symbol or entry.symbol not in self.symbols:
                return False

        # Action filter
        if self.actions:
            if not entry.action or entry.action not in self.actions:
                return False

        # Correlation ID filter
        if self.correlation_id:
            if entry.correlation_id != self.correlation_id:
                return False

        # Decision ID filter (check if entry is related to decision)
        if self.decision_id:
            if entry.entry_id != self.decision_id and self.decision_id not in entry.related_entries:
                # Also check details for decision_id reference
                if entry.details.get("decision_id") != self.decision_id:
                    return False

        # Full-text search
        if self.search_text:
            if not entry.matches_text(self.search_text):
                return False

        return True


class AuditViewer:
    """
    View and search decision audit logs.

    Provides comprehensive audit log viewing capabilities for compliance
    reporting, decision analysis, and trade history investigation.

    Usage:
        viewer = AuditViewer()
        viewer.load_audit_log()

        # Search for specific entries
        filter = AuditFilter(
            start_time=datetime(2024, 1, 1),
            symbols=["AAPL", "MSFT"],
            entry_types=[AuditEntryType.DECISION],
        )
        results = viewer.search_entries(filter)

        # Get entries related to a specific decision
        related = viewer.get_entries_by_decision("decision-uuid")

        # Get trade history for a symbol
        trades = viewer.get_trade_history(symbol="AAPL")

        # Export for compliance report
        export = viewer.export_filtered(filter, format="csv")
    """

    # Default log file paths
    DEFAULT_AUDIT_FILE = "logs/audit.jsonl"
    DEFAULT_TRADE_FILE = "logs/trades.jsonl"
    DEFAULT_DECISION_FILE = "logs/decisions.jsonl"

    def __init__(
        self,
        audit_file: str = DEFAULT_AUDIT_FILE,
        trade_file: str = DEFAULT_TRADE_FILE,
        decision_file: str = DEFAULT_DECISION_FILE,
        logs_dir: str = "logs",
    ):
        """
        Initialize the audit viewer.

        Args:
            audit_file: Path to main audit log file
            trade_file: Path to trades log file
            decision_file: Path to decisions log file
            logs_dir: Directory containing log files (for backup file discovery)
        """
        self._audit_file = Path(audit_file)
        self._trade_file = Path(trade_file)
        self._decision_file = Path(decision_file)
        self._logs_dir = Path(logs_dir)

        # In-memory entry storage
        self._entries: list[AuditEntry] = []
        self._entries_by_id: dict[str, AuditEntry] = {}
        self._entries_by_decision: dict[str, list[AuditEntry]] = {}

        # Index for faster lookups
        self._entries_by_symbol: dict[str, list[AuditEntry]] = {}
        self._entries_by_agent: dict[str, list[AuditEntry]] = {}
        self._entries_by_type: dict[AuditEntryType, list[AuditEntry]] = {}

        # Stats
        self._total_loaded = 0
        self._load_errors = 0

        logger.info(
            f"AuditViewer initialized: audit={audit_file}, "
            f"trades={trade_file}, decisions={decision_file}"
        )

    def load_audit_log(
        self,
        include_backups: bool = False,
        max_entries: int | None = None,
    ) -> int:
        """
        Load audit log entries from all log files.

        Loads from all three log files (audit, trades, decisions) ensuring
        entries from each file type are included. If max_entries is specified,
        the most recent entries are kept after loading all files.

        Args:
            include_backups: Include rotated backup files (e.g., audit.1.jsonl)
            max_entries: Maximum entries to keep (None for unlimited)

        Returns:
            Number of entries loaded
        """
        self._entries.clear()
        self._entries_by_id.clear()
        self._entries_by_decision.clear()
        self._entries_by_symbol.clear()
        self._entries_by_agent.clear()
        self._entries_by_type.clear()
        self._total_loaded = 0
        self._load_errors = 0

        # Collect all files to load - prioritize decisions and trades
        files_to_load: list[tuple[Path, str]] = []

        # Load decisions and trades first (they're smaller and more important)
        for file_path, source in [
            (self._decision_file, "decision"),
            (self._trade_file, "trade"),
            (self._audit_file, "audit"),
        ]:
            if file_path.exists():
                files_to_load.append((file_path, source))

            # Include backup files if requested
            if include_backups and self._logs_dir.exists():
                backup_pattern = f"{file_path.stem}.*.jsonl"
                for backup_file in sorted(self._logs_dir.glob(backup_pattern)):
                    files_to_load.append((backup_file, source))

        # Load entries from all files (no limit during loading)
        for file_path, source in files_to_load:
            self._load_file(file_path, source, max_entries=None)

        # Build relationship index
        self._build_relationship_index()

        # Sort entries by timestamp (most recent first)
        self._entries.sort(key=lambda e: e.timestamp, reverse=True)

        # Apply max_entries limit after sorting (keep most recent)
        if max_entries and len(self._entries) > max_entries:
            # Keep the most recent entries
            removed_entries = self._entries[max_entries:]
            self._entries = self._entries[:max_entries]

            # Update indexes to remove references to removed entries
            removed_ids = {e.entry_id for e in removed_entries}
            for entry_id in removed_ids:
                self._entries_by_id.pop(entry_id, None)

            # Rebuild type/symbol/agent indexes
            self._entries_by_symbol.clear()
            self._entries_by_agent.clear()
            self._entries_by_type.clear()
            for entry in self._entries:
                if entry.symbol:
                    if entry.symbol not in self._entries_by_symbol:
                        self._entries_by_symbol[entry.symbol] = []
                    self._entries_by_symbol[entry.symbol].append(entry)
                if entry.agent_name not in self._entries_by_agent:
                    self._entries_by_agent[entry.agent_name] = []
                self._entries_by_agent[entry.agent_name].append(entry)
                if entry.entry_type not in self._entries_by_type:
                    self._entries_by_type[entry.entry_type] = []
                self._entries_by_type[entry.entry_type].append(entry)

            self._total_loaded = len(self._entries)

        logger.info(
            f"Loaded {self._total_loaded} audit entries "
            f"({self._load_errors} errors) from {len(files_to_load)} files"
        )

        return self._total_loaded

    def _load_file(
        self,
        file_path: Path,
        source: str,
        max_entries: int | None,
    ) -> int:
        """Load entries from a single file."""
        count = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if max_entries and self._total_loaded >= max_entries:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        raw_entry = json.loads(line)
                        entry = self._parse_entry(raw_entry, source)
                        if entry:
                            self._add_entry(entry)
                            count += 1
                            self._total_loaded += 1
                    except json.JSONDecodeError as e:
                        self._load_errors += 1
                        logger.warning(
                            f"JSON parse error in {file_path}:{line_num}: {e}"
                        )
                    except Exception as e:
                        self._load_errors += 1
                        logger.warning(
                            f"Error parsing entry in {file_path}:{line_num}: {e}"
                        )

        except FileNotFoundError:
            logger.debug(f"Audit file not found: {file_path}")
        except Exception as e:
            logger.exception(f"Error reading audit file {file_path}: {e}")

        return count

    def _parse_entry(self, raw: dict[str, Any], source: str) -> AuditEntry | None:
        """Parse a raw JSON entry into an AuditEntry."""
        try:
            # Extract common fields
            timestamp_str = raw.get("timestamp", "")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
            else:
                timestamp = datetime.now(timezone.utc)

            entry_type_str = raw.get("entry_type", "unknown")
            entry_type = AuditEntryType.from_string(entry_type_str)

            agent_name = raw.get("agent_name", "unknown")
            event_id = raw.get("event_id") or f"{source}-{hash(timestamp_str)}"
            details = raw.get("details", {})
            correlation_id = raw.get("correlation_id")

            # Extract type-specific fields
            action = None
            symbol = None
            rationale = None
            data_sources: list[str] = []

            if isinstance(details, dict):
                action = details.get("action") or details.get("side")
                symbol = details.get("symbol")
                rationale = details.get("rationale")
                data_sources = details.get("data_sources", [])

                # Handle nested event details
                if "event_class" in details:
                    symbol = symbol or details.get("symbol")
                    rationale = rationale or details.get("rationale")
                    data_sources = data_sources or details.get("data_sources", [])

            return AuditEntry(
                entry_id=event_id,
                timestamp=timestamp,
                entry_type=entry_type,
                agent_name=agent_name,
                action=action,
                symbol=symbol,
                details=details,
                data_sources=data_sources,
                rationale=rationale,
                correlation_id=correlation_id,
            )

        except Exception as e:
            logger.warning(f"Failed to parse audit entry: {e}")
            return None

    def _add_entry(self, entry: AuditEntry) -> None:
        """Add entry to storage and indexes."""
        self._entries.append(entry)
        self._entries_by_id[entry.entry_id] = entry

        # Index by symbol
        if entry.symbol:
            if entry.symbol not in self._entries_by_symbol:
                self._entries_by_symbol[entry.symbol] = []
            self._entries_by_symbol[entry.symbol].append(entry)

        # Index by agent
        if entry.agent_name not in self._entries_by_agent:
            self._entries_by_agent[entry.agent_name] = []
        self._entries_by_agent[entry.agent_name].append(entry)

        # Index by type
        if entry.entry_type not in self._entries_by_type:
            self._entries_by_type[entry.entry_type] = []
        self._entries_by_type[entry.entry_type].append(entry)

    def _build_relationship_index(self) -> None:
        """Build index linking related entries (decision -> checks -> trades)."""
        # Index trades and compliance checks by decision_id
        for entry in self._entries:
            decision_id = entry.details.get("decision_id")
            if decision_id:
                entry.related_entries.append(decision_id)

                if decision_id not in self._entries_by_decision:
                    self._entries_by_decision[decision_id] = []
                self._entries_by_decision[decision_id].append(entry)

        # Link compliance checks to decisions by event_id matching
        for entry in self._entries:
            if entry.entry_type == AuditEntryType.DECISION:
                # Find compliance checks with matching event_id
                for other in self._entries:
                    if other.entry_type == AuditEntryType.COMPLIANCE_CHECK:
                        if other.entry_id == entry.entry_id:
                            if entry.entry_id not in self._entries_by_decision:
                                self._entries_by_decision[entry.entry_id] = []
                            self._entries_by_decision[entry.entry_id].append(other)
                            other.related_entries.append(entry.entry_id)

    def search_entries(
        self,
        filter: AuditFilter | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """
        Search audit entries with filtering.

        Args:
            filter: Filter criteria (None returns all)
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of matching AuditEntry objects
        """
        if filter is None:
            # Return all entries (paginated)
            return self._entries[offset:offset + limit]

        # Apply filter
        matching = []
        skipped = 0

        for entry in self._entries:
            if filter.matches(entry):
                if skipped < offset:
                    skipped += 1
                    continue

                matching.append(entry)

                if len(matching) >= limit:
                    break

        return matching

    def get_entries_by_decision(self, decision_id: str) -> list[AuditEntry]:
        """
        Get all entries related to a specific decision.

        Returns entries in chronological order:
        decision -> risk_check -> compliance_check -> execution -> fill

        Args:
            decision_id: The decision event ID

        Returns:
            List of related AuditEntry objects
        """
        related = []

        # Get the decision entry itself
        if decision_id in self._entries_by_id:
            related.append(self._entries_by_id[decision_id])

        # Get entries referencing this decision
        if decision_id in self._entries_by_decision:
            related.extend(self._entries_by_decision[decision_id])

        # Sort by timestamp for chronological order
        related.sort(key=lambda e: e.timestamp)

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for entry in related:
            if entry.entry_id not in seen:
                seen.add(entry.entry_id)
                unique.append(entry)

        return unique

    def get_trade_history(
        self,
        symbol: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """
        Get trade history with optional filtering.

        Args:
            symbol: Filter by symbol (None for all)
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum trades to return

        Returns:
            List of trade AuditEntry objects
        """
        filter = AuditFilter(
            start_time=start_time,
            end_time=end_time,
            entry_types=[AuditEntryType.TRADE],
            symbols=[symbol] if symbol else None,
        )

        return self.search_entries(filter, limit=limit)

    def get_decision_history(
        self,
        symbol: str | None = None,
        agent: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """
        Get decision history with optional filtering.

        Args:
            symbol: Filter by symbol
            agent: Filter by agent name
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum decisions to return

        Returns:
            List of decision AuditEntry objects
        """
        filter = AuditFilter(
            start_time=start_time,
            end_time=end_time,
            entry_types=[AuditEntryType.DECISION],
            symbols=[symbol] if symbol else None,
            agents=[agent] if agent else None,
        )

        return self.search_entries(filter, limit=limit)

    def get_compliance_checks(
        self,
        approved: bool | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """
        Get compliance check history.

        Args:
            approved: Filter by approval status (None for all)
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum results to return

        Returns:
            List of compliance check AuditEntry objects
        """
        filter = AuditFilter(
            start_time=start_time,
            end_time=end_time,
            entry_types=[AuditEntryType.COMPLIANCE_CHECK],
        )

        results = self.search_entries(filter, limit=limit * 2)  # Get extra for filtering

        if approved is not None:
            results = [
                e for e in results
                if e.details.get("approved") == approved
            ][:limit]

        return results[:limit]

    def get_risk_alerts(
        self,
        severity: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """
        Get risk alert history.

        Args:
            severity: Filter by severity (e.g., "warning", "critical")
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum results to return

        Returns:
            List of risk alert AuditEntry objects
        """
        filter = AuditFilter(
            start_time=start_time,
            end_time=end_time,
            entry_types=[AuditEntryType.RISK_ALERT],
        )

        results = self.search_entries(filter, limit=limit * 2)

        if severity:
            results = [
                e for e in results
                if e.details.get("severity", "").lower() == severity.lower()
            ]

        return results[:limit]

    def export_filtered(
        self,
        filter: AuditFilter | None = None,
        format: str = "json",
        limit: int = 10000,
    ) -> str | list[dict]:
        """
        Export filtered audit entries for compliance reporting.

        Args:
            filter: Filter criteria
            format: Output format ("json", "csv", "dict")
            limit: Maximum entries to export

        Returns:
            Formatted export string or list of dicts
        """
        entries = self.search_entries(filter, limit=limit)

        if format == "dict":
            return [e.to_dict() for e in entries]

        if format == "csv":
            return self._export_csv(entries)

        # Default to JSON
        return json.dumps(
            [e.to_dict() for e in entries],
            indent=2,
            default=str,
        )

    def _export_csv(self, entries: list[AuditEntry]) -> str:
        """Export entries as CSV format."""
        if not entries:
            return ""

        # CSV header
        headers = [
            "timestamp",
            "entry_type",
            "agent_name",
            "symbol",
            "action",
            "rationale",
            "data_sources",
            "entry_id",
            "correlation_id",
        ]

        lines = [",".join(headers)]

        for entry in entries:
            row = [
                entry.timestamp.isoformat(),
                entry.entry_type.value,
                entry.agent_name,
                entry.symbol or "",
                entry.action or "",
                self._csv_escape(entry.rationale or ""),
                ";".join(entry.data_sources),
                entry.entry_id,
                entry.correlation_id or "",
            ]
            lines.append(",".join(row))

        return "\n".join(lines)

    def _csv_escape(self, value: str) -> str:
        """Escape a value for CSV output."""
        if "," in value or '"' in value or "\n" in value:
            return '"' + value.replace('"', '""') + '"'
        return value

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about loaded audit entries.

        Returns:
            Dictionary with audit statistics
        """
        stats: dict[str, Any] = {
            "total_entries": len(self._entries),
            "load_errors": self._load_errors,
            "entries_by_type": {},
            "entries_by_agent": {},
            "unique_symbols": len(self._entries_by_symbol),
            "time_range": {
                "earliest": None,
                "latest": None,
            },
        }

        # Count by type
        for entry_type, entries in self._entries_by_type.items():
            stats["entries_by_type"][entry_type.value] = len(entries)

        # Count by agent
        for agent, entries in self._entries_by_agent.items():
            stats["entries_by_agent"][agent] = len(entries)

        # Time range
        if self._entries:
            timestamps = [e.timestamp for e in self._entries]
            stats["time_range"]["earliest"] = min(timestamps).isoformat()
            stats["time_range"]["latest"] = max(timestamps).isoformat()

        return stats

    def get_unique_values(self, field: str) -> list[str]:
        """
        Get unique values for a field (for filter dropdowns).

        Args:
            field: Field name ("symbol", "agent", "action")

        Returns:
            List of unique values
        """
        if field == "symbol":
            return sorted(self._entries_by_symbol.keys())

        if field == "agent":
            return sorted(self._entries_by_agent.keys())

        if field == "action":
            actions = set()
            for entry in self._entries:
                if entry.action:
                    actions.add(entry.action)
            return sorted(actions)

        if field == "entry_type":
            return [et.value for et in self._entries_by_type.keys()]

        return []

    def to_dict(self) -> dict[str, Any]:
        """
        Export viewer state to dictionary for WebSocket streaming.

        Returns:
            Dictionary with viewer state
        """
        return {
            "statistics": self.get_statistics(),
            "recent_entries": [e.to_dict() for e in self._entries[:50]],
            "unique_symbols": self.get_unique_values("symbol"),
            "unique_agents": self.get_unique_values("agent"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @property
    def entry_count(self) -> int:
        """Total number of loaded entries."""
        return len(self._entries)

    @property
    def is_loaded(self) -> bool:
        """Check if audit logs have been loaded."""
        return self._total_loaded > 0

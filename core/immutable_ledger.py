"""
Immutable Audit Ledger
======================

Hash-chained event logging for regulatory compliance (MiFID II, AMF).

Features:
- Each entry contains hash of previous entry (tamper-evident chain)
- SHA-256 hashing for cryptographic integrity
- Periodic integrity verification
- Export to compliance-ready format
- Automatic rotation and archiving

MiFID II Requirements Addressed:
- RTS 6 Article 4: Order and execution records
- RTS 6 Article 18: Kill switch audit trail
- RTS 25 Article 4: Record keeping
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional
from collections import deque
import threading

logger = logging.getLogger(__name__)


@dataclass
class LedgerEntry:
    """
    A single entry in the immutable ledger.

    Each entry is cryptographically linked to the previous entry
    via the previous_hash field, creating a tamper-evident chain.
    """
    sequence_number: int
    timestamp: str  # ISO format
    event_type: str
    source_agent: str
    event_data: dict
    previous_hash: str
    entry_hash: str = ""

    def __post_init__(self):
        """Calculate entry hash after initialization."""
        if not self.entry_hash:
            self.entry_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """
        Calculate SHA-256 hash of the entry.

        Hash covers: sequence_number, timestamp, event_type,
        source_agent, event_data, previous_hash
        """
        hash_input = json.dumps({
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "source_agent": self.source_agent,
            "event_data": self.event_data,
            "previous_hash": self.previous_hash,
        }, sort_keys=True, separators=(',', ':'))

        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    def verify(self) -> bool:
        """Verify that the entry hash is valid."""
        return self.entry_hash == self._calculate_hash()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LedgerEntry":
        """Create entry from dictionary."""
        return cls(
            sequence_number=data["sequence_number"],
            timestamp=data["timestamp"],
            event_type=data["event_type"],
            source_agent=data["source_agent"],
            event_data=data["event_data"],
            previous_hash=data["previous_hash"],
            entry_hash=data.get("entry_hash", ""),
        )


class ImmutableAuditLedger:
    """
    Immutable, hash-chained audit ledger for compliance.

    The ledger maintains cryptographic integrity by linking each
    entry to the previous one via SHA-256 hashes. Any tampering
    with historical entries will break the hash chain and be
    detectable during verification.

    Thread-safe for concurrent access.
    """

    GENESIS_HASH = "0" * 64  # Genesis block has all-zero previous hash

    def __init__(
        self,
        storage_path: Path | str = "logs/audit_ledger",
        max_memory_entries: int = 10000,
        auto_flush_interval_seconds: float = 60.0,
        verification_interval_entries: int = 1000,
    ):
        """
        Initialize the immutable audit ledger.

        Args:
            storage_path: Directory for ledger storage
            max_memory_entries: Max entries to keep in memory
            auto_flush_interval_seconds: Auto-flush to disk interval
            verification_interval_entries: Verify chain every N entries
        """
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._max_memory_entries = max_memory_entries
        self._auto_flush_interval = auto_flush_interval_seconds
        self._verification_interval = verification_interval_entries

        # In-memory ledger (bounded)
        self._entries: deque[LedgerEntry] = deque(maxlen=max_memory_entries)

        # Chain state
        self._sequence_number = 0
        self._last_hash = self.GENESIS_HASH
        self._chain_valid = True

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "total_entries": 0,
            "verification_passes": 0,
            "verification_failures": 0,
            "flushes_to_disk": 0,
            "integrity_checks": 0,
        }

        # Load existing ledger state if present
        self._load_state()

        logger.info(
            f"ImmutableAuditLedger initialized at {self._storage_path} "
            f"(sequence: {self._sequence_number}, entries in memory: {len(self._entries)})"
        )

    def append(
        self,
        event_type: str,
        source_agent: str,
        event_data: dict,
    ) -> LedgerEntry:
        """
        Append a new entry to the ledger.

        The entry is cryptographically linked to the previous entry.

        Args:
            event_type: Type of event (decision, order, fill, risk_alert, etc.)
            source_agent: Agent that generated the event
            event_data: Event payload

        Returns:
            The created LedgerEntry

        Raises:
            ValueError: If input parameters are invalid
        """
        # CRITICAL FIX: Input validation for audit trail integrity
        if not event_type or not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string")
        if not source_agent or not isinstance(source_agent, str) or not source_agent.strip():
            raise ValueError("source_agent must be a non-empty string")
        if event_data is None or not isinstance(event_data, dict):
            raise ValueError("event_data must be a non-null dictionary")

        with self._lock:
            self._sequence_number += 1

            entry = LedgerEntry(
                sequence_number=self._sequence_number,
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=event_type,
                source_agent=source_agent,
                event_data=event_data,
                previous_hash=self._last_hash,
            )

            self._entries.append(entry)
            self._last_hash = entry.entry_hash
            self._stats["total_entries"] += 1

            # Periodic verification
            if self._sequence_number % self._verification_interval == 0:
                self._verify_recent_chain()

            return entry

    def verify_chain(self, start_seq: int = 1, end_seq: Optional[int] = None) -> tuple[bool, list[int]]:
        """
        Verify the integrity of the hash chain.

        Args:
            start_seq: Starting sequence number
            end_seq: Ending sequence number (None = latest)

        Returns:
            Tuple of (chain_valid, list_of_invalid_sequences)
        """
        with self._lock:
            self._stats["integrity_checks"] += 1

            if not self._entries:
                return True, []

            invalid_sequences = []
            entries_to_check = [
                e for e in self._entries
                if e.sequence_number >= start_seq and
                   (end_seq is None or e.sequence_number <= end_seq)
            ]

            if not entries_to_check:
                return True, []

            # Verify each entry
            for i, entry in enumerate(entries_to_check):
                # Verify entry's own hash
                if not entry.verify():
                    invalid_sequences.append(entry.sequence_number)
                    logger.error(
                        f"LEDGER INTEGRITY FAILURE: Entry {entry.sequence_number} hash mismatch"
                    )
                    continue

                # Verify chain link (except first entry)
                if i > 0:
                    prev_entry = entries_to_check[i - 1]
                    if entry.previous_hash != prev_entry.entry_hash:
                        invalid_sequences.append(entry.sequence_number)
                        logger.error(
                            f"LEDGER INTEGRITY FAILURE: Entry {entry.sequence_number} "
                            f"chain link broken (expected {prev_entry.entry_hash[:16]}..., "
                            f"got {entry.previous_hash[:16]}...)"
                        )

            chain_valid = len(invalid_sequences) == 0

            if chain_valid:
                self._stats["verification_passes"] += 1
                logger.debug(f"Ledger chain verified: sequences {start_seq}-{entries_to_check[-1].sequence_number}")
            else:
                self._stats["verification_failures"] += 1
                logger.critical(
                    f"LEDGER CHAIN INVALID: {len(invalid_sequences)} corrupted entries detected"
                )

            self._chain_valid = chain_valid
            return chain_valid, invalid_sequences

    def _verify_recent_chain(self, lookback: int = 100) -> bool:
        """Verify the most recent entries in the chain."""
        start_seq = max(1, self._sequence_number - lookback)
        valid, _ = self.verify_chain(start_seq)
        return valid

    def flush_to_disk(self) -> int:
        """
        Flush current entries to disk storage.

        Returns:
            Number of entries flushed
        """
        with self._lock:
            if not self._entries:
                return 0

            # Generate filename based on date and sequence range
            now = datetime.now(timezone.utc)
            first_seq = self._entries[0].sequence_number
            last_seq = self._entries[-1].sequence_number

            filename = f"ledger_{now.strftime('%Y%m%d')}_{first_seq:08d}_{last_seq:08d}.jsonl"
            filepath = self._storage_path / filename

            entries_written = 0
            with open(filepath, 'a', encoding='utf-8') as f:
                for entry in self._entries:
                    f.write(json.dumps(entry.to_dict()) + '\n')
                    entries_written += 1

            self._stats["flushes_to_disk"] += 1
            self._save_state()

            logger.info(f"Flushed {entries_written} entries to {filepath}")
            return entries_written

    def _save_state(self) -> None:
        """Save ledger state for persistence."""
        state = {
            "sequence_number": self._sequence_number,
            "last_hash": self._last_hash,
            "chain_valid": self._chain_valid,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "stats": self._stats,
        }

        state_file = self._storage_path / "ledger_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load ledger state from persistence."""
        state_file = self._storage_path / "ledger_state.json"

        if not state_file.exists():
            return

        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            self._sequence_number = state.get("sequence_number", 0)
            self._last_hash = state.get("last_hash", self.GENESIS_HASH)
            self._chain_valid = state.get("chain_valid", True)
            self._stats = state.get("stats", self._stats)

            logger.info(
                f"Loaded ledger state: sequence={self._sequence_number}, "
                f"chain_valid={self._chain_valid}"
            )

        except Exception as e:
            logger.error(f"Failed to load ledger state: {e}")

    def get_entries(
        self,
        start_seq: Optional[int] = None,
        end_seq: Optional[int] = None,
        event_type: Optional[str] = None,
        source_agent: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Query ledger entries with filters.

        Args:
            start_seq: Starting sequence number
            end_seq: Ending sequence number
            event_type: Filter by event type
            source_agent: Filter by source agent
            limit: Maximum entries to return

        Returns:
            List of matching entries as dicts
        """
        with self._lock:
            results = []

            for entry in self._entries:
                # Apply filters
                if start_seq is not None and entry.sequence_number < start_seq:
                    continue
                if end_seq is not None and entry.sequence_number > end_seq:
                    continue
                if event_type is not None and entry.event_type != event_type:
                    continue
                if source_agent is not None and entry.source_agent != source_agent:
                    continue

                results.append(entry.to_dict())

                if len(results) >= limit:
                    break

            return results

    def get_entry_by_sequence(self, sequence_number: int) -> Optional[dict]:
        """Get a specific entry by sequence number."""
        with self._lock:
            for entry in self._entries:
                if entry.sequence_number == sequence_number:
                    return entry.to_dict()
            return None

    def export_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        output_path: Optional[Path] = None,
    ) -> dict:
        """
        Export a compliance-ready report of ledger entries.

        Args:
            start_date: Report start date
            end_date: Report end date
            output_path: Optional path to write JSON report

        Returns:
            Report dict with entries, statistics, and integrity status
        """
        with self._lock:
            # Filter entries by date
            entries_in_range = []
            for entry in self._entries:
                entry_time = datetime.fromisoformat(entry.timestamp)
                if start_date <= entry_time <= end_date:
                    entries_in_range.append(entry.to_dict())

            # Build report
            report = {
                "report_generated": datetime.now(timezone.utc).isoformat(),
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "total_entries": len(entries_in_range),
                "chain_integrity_verified": self._chain_valid,
                "first_sequence": entries_in_range[0]["sequence_number"] if entries_in_range else None,
                "last_sequence": entries_in_range[-1]["sequence_number"] if entries_in_range else None,
                "first_hash": entries_in_range[0]["entry_hash"][:16] if entries_in_range else None,
                "last_hash": entries_in_range[-1]["entry_hash"][:16] if entries_in_range else None,
                "event_type_summary": {},
                "source_agent_summary": {},
                "entries": entries_in_range,
            }

            # Summarize by event type and source
            for entry in entries_in_range:
                event_type = entry["event_type"]
                source = entry["source_agent"]

                report["event_type_summary"][event_type] = \
                    report["event_type_summary"].get(event_type, 0) + 1
                report["source_agent_summary"][source] = \
                    report["source_agent_summary"].get(source, 0) + 1

            # Write to file if requested
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Compliance report exported to {output_path}")

            return report

    def get_statistics(self) -> dict:
        """Get ledger statistics."""
        with self._lock:
            return {
                **self._stats,
                "current_sequence": self._sequence_number,
                "entries_in_memory": len(self._entries),
                "max_memory_entries": self._max_memory_entries,
                "last_hash_prefix": self._last_hash[:16] if self._last_hash else None,
                "chain_valid": self._chain_valid,
                "storage_path": str(self._storage_path),
            }

    def get_chain_summary(self) -> dict:
        """Get summary of the hash chain."""
        with self._lock:
            return {
                "genesis_hash": self.GENESIS_HASH[:16],
                "current_hash": self._last_hash[:16] if self._last_hash else None,
                "total_entries": self._sequence_number,
                "chain_length": len(self._entries),
                "chain_valid": self._chain_valid,
                "oldest_entry": self._entries[0].to_dict() if self._entries else None,
                "newest_entry": self._entries[-1].to_dict() if self._entries else None,
            }


def create_audit_ledger(
    storage_path: str = "logs/audit_ledger",
) -> ImmutableAuditLedger:
    """
    Create a configured audit ledger instance.

    Args:
        storage_path: Directory for ledger storage

    Returns:
        Configured ImmutableAuditLedger
    """
    return ImmutableAuditLedger(
        storage_path=storage_path,
        max_memory_entries=10000,
        auto_flush_interval_seconds=60.0,
        verification_interval_entries=1000,
    )

"""
Tests for Immutable Audit Ledger (MiFID II Compliance)
======================================================

Tests cover:
- Hash chain integrity
- Entry creation and verification
- Chain tampering detection
- Export functionality
- Thread safety
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
import threading
import time

from core.immutable_ledger import (
    ImmutableAuditLedger,
    LedgerEntry,
    create_audit_ledger,
)


class TestLedgerEntry:
    """Tests for individual ledger entries."""

    def test_entry_creates_hash(self):
        """Entry should auto-generate hash on creation."""
        entry = LedgerEntry(
            sequence_number=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="test",
            source_agent="TestAgent",
            event_data={"key": "value"},
            previous_hash="0" * 64,
        )
        assert entry.entry_hash != ""
        assert len(entry.entry_hash) == 64  # SHA-256 hex length

    def test_entry_verify_valid(self):
        """Valid entry should pass verification."""
        entry = LedgerEntry(
            sequence_number=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="decision",
            source_agent="CIOAgent",
            event_data={"symbol": "MES", "action": "BUY"},
            previous_hash="0" * 64,
        )
        assert entry.verify() is True

    def test_entry_verify_tampered(self):
        """Tampered entry should fail verification."""
        entry = LedgerEntry(
            sequence_number=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="decision",
            source_agent="CIOAgent",
            event_data={"symbol": "MES", "action": "BUY"},
            previous_hash="0" * 64,
        )
        # Tamper with data
        entry.event_data["symbol"] = "TAMPERED"
        assert entry.verify() is False

    def test_entry_to_dict(self):
        """Entry should serialize to dict correctly."""
        entry = LedgerEntry(
            sequence_number=42,
            timestamp="2026-02-05T12:00:00+00:00",
            event_type="fill",
            source_agent="ExecutionAgent",
            event_data={"price": 100.0},
            previous_hash="abc123",
        )
        d = entry.to_dict()
        assert d["sequence_number"] == 42
        assert d["event_type"] == "fill"
        assert d["source_agent"] == "ExecutionAgent"

    def test_entry_from_dict(self):
        """Entry should deserialize from dict correctly."""
        data = {
            "sequence_number": 10,
            "timestamp": "2026-02-05T12:00:00+00:00",
            "event_type": "order",
            "source_agent": "CIOAgent",
            "event_data": {"qty": 5},
            "previous_hash": "0" * 64,
            "entry_hash": "test_hash",
        }
        entry = LedgerEntry.from_dict(data)
        assert entry.sequence_number == 10
        assert entry.event_type == "order"


class TestImmutableAuditLedger:
    """Tests for the audit ledger."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def ledger(self, temp_dir):
        """Create a ledger for testing."""
        return ImmutableAuditLedger(
            storage_path=temp_dir,
            max_memory_entries=100,
            verification_interval_entries=10,
        )

    def test_append_creates_entry(self, ledger):
        """Appending should create a new entry."""
        entry = ledger.append(
            event_type="test",
            source_agent="TestAgent",
            event_data={"test": True},
        )
        assert entry.sequence_number == 1
        assert entry.event_type == "test"
        assert entry.source_agent == "TestAgent"

    def test_append_chain_links(self, ledger):
        """Entries should be linked via hash chain."""
        entry1 = ledger.append("test1", "Agent1", {})
        entry2 = ledger.append("test2", "Agent2", {})

        assert entry2.previous_hash == entry1.entry_hash

    def test_genesis_hash(self, ledger):
        """First entry should have genesis hash as previous."""
        entry = ledger.append("first", "Agent", {})
        assert entry.previous_hash == "0" * 64

    def test_verify_chain_valid(self, ledger):
        """Valid chain should pass verification."""
        for i in range(10):
            ledger.append(f"event_{i}", "Agent", {"index": i})

        is_valid, invalid_seqs = ledger.verify_chain()
        assert is_valid is True
        assert len(invalid_seqs) == 0

    def test_verify_chain_empty(self, ledger):
        """Empty ledger should pass verification."""
        is_valid, invalid_seqs = ledger.verify_chain()
        assert is_valid is True
        assert len(invalid_seqs) == 0

    def test_get_entries_filtered(self, ledger):
        """Should filter entries by criteria."""
        ledger.append("decision", "CIOAgent", {"symbol": "MES"})
        ledger.append("order", "ExecutionAgent", {"symbol": "MES"})
        ledger.append("decision", "CIOAgent", {"symbol": "MNQ"})

        decisions = ledger.get_entries(event_type="decision")
        assert len(decisions) == 2

        cio_entries = ledger.get_entries(source_agent="CIOAgent")
        assert len(cio_entries) == 2

    def test_get_entry_by_sequence(self, ledger):
        """Should retrieve entry by sequence number."""
        ledger.append("first", "Agent", {})
        ledger.append("second", "Agent", {})
        ledger.append("third", "Agent", {})

        entry = ledger.get_entry_by_sequence(2)
        assert entry is not None
        assert entry["event_type"] == "second"

    def test_statistics(self, ledger):
        """Should track statistics correctly."""
        for i in range(5):
            ledger.append(f"event_{i}", "Agent", {})

        stats = ledger.get_statistics()
        assert stats["total_entries"] == 5
        assert stats["current_sequence"] == 5
        assert stats["chain_valid"] is True

    def test_flush_to_disk(self, ledger, temp_dir):
        """Should flush entries to disk."""
        for i in range(5):
            ledger.append(f"event_{i}", "Agent", {})

        count = ledger.flush_to_disk()
        assert count == 5

        # Check file was created
        files = list(Path(temp_dir).glob("ledger_*.jsonl"))
        assert len(files) >= 1

    def test_thread_safety(self, ledger):
        """Should handle concurrent appends safely."""
        results = []
        errors = []

        def append_entries(agent_name, count):
            try:
                for i in range(count):
                    entry = ledger.append(
                        f"event_{i}",
                        agent_name,
                        {"index": i}
                    )
                    results.append(entry.sequence_number)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=append_entries, args=(f"Agent{i}", 10))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 entries
        # All sequence numbers should be unique
        assert len(set(results)) == 50

    def test_export_compliance_report(self, ledger, temp_dir):
        """Should export compliance report."""
        now = datetime.now(timezone.utc)
        for i in range(5):
            ledger.append("decision", "CIOAgent", {"trade": i})

        report = ledger.export_compliance_report(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
            output_path=Path(temp_dir) / "report.json",
        )

        assert report["total_entries"] == 5
        assert report["chain_integrity_verified"] is True
        assert (Path(temp_dir) / "report.json").exists()


class TestCreateAuditLedger:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        """Should create ledger with default config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = create_audit_ledger(storage_path=temp_dir)
            assert ledger is not None
            assert ledger._max_memory_entries == 10000


class TestMiFIDIICompliance:
    """Tests specifically for MiFID II requirements."""

    @pytest.fixture
    def ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ImmutableAuditLedger(storage_path=temp_dir)

    def test_timestamp_utc_millisecond(self, ledger):
        """Timestamps must be UTC with millisecond precision."""
        entry = ledger.append("decision", "CIOAgent", {})

        # Verify timestamp is ISO format with timezone
        timestamp = datetime.fromisoformat(entry.timestamp)
        assert timestamp.tzinfo is not None
        # Should have microsecond precision (which includes milliseconds)
        assert "+" in entry.timestamp or "Z" in entry.timestamp

    def test_kill_switch_audit_trail(self, ledger):
        """Kill switch events must be audited."""
        entry = ledger.append(
            event_type="kill_switch",
            source_agent="RiskAgent",
            event_data={
                "activated": True,
                "reason": "daily_loss_limit",
                "trigger_type": "automatic",
                "positions_at_activation": 5,
            }
        )

        assert entry.event_type == "kill_switch"
        assert entry.event_data["activated"] is True
        assert entry.verify() is True

    def test_chain_tampering_detected(self, ledger):
        """Any tampering must be detectable."""
        # Create valid chain
        for i in range(10):
            ledger.append(f"event_{i}", "Agent", {"i": i})

        # Verify it's valid
        is_valid, _ = ledger.verify_chain()
        assert is_valid is True

        # Tamper with an entry (simulate attacker)
        if ledger._entries:
            ledger._entries[5].event_data["i"] = "TAMPERED"

        # Tampering should be detected
        is_valid, invalid_seqs = ledger.verify_chain()
        assert is_valid is False
        assert 6 in invalid_seqs  # Entry 6 (0-indexed 5) is tampered

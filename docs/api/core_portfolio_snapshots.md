# portfolio_snapshots

**Path**: `C:\Users\Alexa\ai-trading-firm\core\portfolio_snapshots.py`

## Overview

Historical Portfolio Snapshots Module
=====================================

Store and retrieve historical portfolio snapshots (Issue #P17).

Features:
- Periodic portfolio state capture
- Position history tracking
- P&L history with attribution
- Snapshot comparison tools
- Export to various formats

## Classes

### SnapshotType

**Inherits from**: str, Enum

Type of portfolio snapshot.

### PositionSnapshot

Snapshot of a single position.

### PortfolioSnapshot

Complete portfolio snapshot at a point in time.

#### Methods

##### `def to_dict(self) -> dict`

Convert to dictionary.

##### `def from_dict(cls, data: dict) -> PortfolioSnapshot`

Create from dictionary.

##### `def get_position(self, symbol: str)`

Get position by symbol.

##### `def get_sector_exposure(self) -> dict[str, float]`

Get exposure by sector.

### PortfolioSnapshotStore

Persistent storage for portfolio snapshots (#P17).

Uses SQLite for efficient querying and storage.

#### Methods

##### `def __init__(self, db_path: str)`

##### `def save_snapshot(self, snapshot: PortfolioSnapshot) -> None`

Save a portfolio snapshot.

##### `def get_snapshot(self, snapshot_id: str)`

Get a specific snapshot by ID.

##### `def get_snapshots(self, start_date: , end_date: , snapshot_type: , limit: int) -> list[PortfolioSnapshot]`

Query snapshots with filters.

##### `def get_latest_snapshot(self, snapshot_type: )`

Get the most recent snapshot.

##### `def get_eod_snapshots(self, days: int) -> list[PortfolioSnapshot]`

Get end-of-day snapshots for the last N days.

##### `def delete_old_snapshots(self, retention_days: int, keep_eod: bool) -> int`

Delete old snapshots while respecting retention policy.

##### `def export_to_json(self, filepath: str, start_date: , end_date: ) -> int`

Export snapshots to JSON file.

##### `def get_statistics(self) -> dict`

Get storage statistics.

### PortfolioSnapshotManager

High-level manager for portfolio snapshots (#P17).

Handles automatic snapshot scheduling and comparison.

#### Methods

##### `def __init__(self, store: , intraday_interval_minutes: int)`

##### `def should_take_intraday_snapshot(self) -> bool`

Check if it's time for an intraday snapshot.

##### `def create_snapshot(self, snapshot_type: SnapshotType, risk_state: Any, positions: dict[str, Any], trigger_reason: str, notes: str) -> PortfolioSnapshot`

Create a new portfolio snapshot from current state.

##### `def compare_snapshots(self, snapshot1: PortfolioSnapshot, snapshot2: PortfolioSnapshot) -> dict`

Compare two snapshots and return differences (#P18 partial).

##### `def get_performance_history(self, days: int) -> list[dict]`

Get daily performance history.

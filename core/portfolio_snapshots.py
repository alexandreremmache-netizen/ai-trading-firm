"""
Historical Portfolio Snapshots Module
=====================================

Store and retrieve historical portfolio snapshots (Issue #P17).

Features:
- Periodic portfolio state capture
- Position history tracking
- P&L history with attribution
- Snapshot comparison tools
- Export to various formats
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterator
from enum import Enum

logger = logging.getLogger(__name__)


class SnapshotType(str, Enum):
    """Type of portfolio snapshot."""
    INTRADAY = "intraday"  # Periodic intraday snapshots
    EOD = "eod"  # End of day
    MONTHLY = "monthly"  # Month-end
    QUARTERLY = "quarterly"  # Quarter-end
    ANNUAL = "annual"  # Year-end
    EVENT = "event"  # Triggered by specific event (rebalance, large trade)


@dataclass
class PositionSnapshot:
    """Snapshot of a single position."""
    symbol: str
    quantity: int
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    weight_pct: float
    sector: str = ""
    asset_class: str = ""


@dataclass
class PortfolioSnapshot:
    """Complete portfolio snapshot at a point in time."""
    snapshot_id: str
    timestamp: datetime
    snapshot_type: SnapshotType

    # Portfolio values
    net_liquidation: float
    gross_exposure: float
    net_exposure: float
    cash: float

    # P&L
    daily_pnl: float
    mtd_pnl: float
    ytd_pnl: float
    inception_pnl: float

    # Risk metrics
    var_95: float
    var_99: float
    expected_shortfall: float
    current_drawdown: float
    max_drawdown: float

    # Greeks (if applicable)
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    portfolio_theta: float = 0.0

    # Positions
    positions: list[PositionSnapshot] = field(default_factory=list)

    # Metadata
    trigger_reason: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['snapshot_type'] = self.snapshot_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'PortfolioSnapshot':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['snapshot_type'] = SnapshotType(data['snapshot_type'])
        data['positions'] = [PositionSnapshot(**p) for p in data.get('positions', [])]
        return cls(**data)

    def get_position(self, symbol: str) -> PositionSnapshot | None:
        """Get position by symbol."""
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None

    def get_sector_exposure(self) -> dict[str, float]:
        """Get exposure by sector."""
        sectors: dict[str, float] = {}
        for pos in self.positions:
            sector = pos.sector or "Unknown"
            sectors[sector] = sectors.get(sector, 0) + pos.market_value
        return sectors


class PortfolioSnapshotStore:
    """
    Persistent storage for portfolio snapshots (#P17).

    Uses SQLite for efficient querying and storage.
    """

    def __init__(self, db_path: str = "data/portfolio_snapshots.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    snapshot_type TEXT NOT NULL,
                    net_liquidation REAL,
                    gross_exposure REAL,
                    net_exposure REAL,
                    cash REAL,
                    daily_pnl REAL,
                    mtd_pnl REAL,
                    ytd_pnl REAL,
                    inception_pnl REAL,
                    var_95 REAL,
                    var_99 REAL,
                    expected_shortfall REAL,
                    current_drawdown REAL,
                    max_drawdown REAL,
                    portfolio_delta REAL,
                    portfolio_gamma REAL,
                    portfolio_vega REAL,
                    portfolio_theta REAL,
                    trigger_reason TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity INTEGER,
                    avg_cost REAL,
                    market_price REAL,
                    market_value REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    weight_pct REAL,
                    sector TEXT,
                    asset_class TEXT,
                    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp
                ON snapshots(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_type
                ON snapshots(snapshot_type)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_snapshot
                ON positions(snapshot_id)
            """)

    def save_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Save a portfolio snapshot."""
        with sqlite3.connect(self.db_path) as conn:
            # Insert snapshot
            conn.execute("""
                INSERT OR REPLACE INTO snapshots (
                    snapshot_id, timestamp, snapshot_type,
                    net_liquidation, gross_exposure, net_exposure, cash,
                    daily_pnl, mtd_pnl, ytd_pnl, inception_pnl,
                    var_95, var_99, expected_shortfall,
                    current_drawdown, max_drawdown,
                    portfolio_delta, portfolio_gamma, portfolio_vega, portfolio_theta,
                    trigger_reason, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.snapshot_id,
                snapshot.timestamp.isoformat(),
                snapshot.snapshot_type.value,
                snapshot.net_liquidation,
                snapshot.gross_exposure,
                snapshot.net_exposure,
                snapshot.cash,
                snapshot.daily_pnl,
                snapshot.mtd_pnl,
                snapshot.ytd_pnl,
                snapshot.inception_pnl,
                snapshot.var_95,
                snapshot.var_99,
                snapshot.expected_shortfall,
                snapshot.current_drawdown,
                snapshot.max_drawdown,
                snapshot.portfolio_delta,
                snapshot.portfolio_gamma,
                snapshot.portfolio_vega,
                snapshot.portfolio_theta,
                snapshot.trigger_reason,
                snapshot.notes,
            ))

            # Delete old positions for this snapshot (in case of update)
            conn.execute("DELETE FROM positions WHERE snapshot_id = ?", (snapshot.snapshot_id,))

            # Insert positions
            for pos in snapshot.positions:
                conn.execute("""
                    INSERT INTO positions (
                        snapshot_id, symbol, quantity, avg_cost, market_price,
                        market_value, unrealized_pnl, realized_pnl,
                        weight_pct, sector, asset_class
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.snapshot_id,
                    pos.symbol,
                    pos.quantity,
                    pos.avg_cost,
                    pos.market_price,
                    pos.market_value,
                    pos.unrealized_pnl,
                    pos.realized_pnl,
                    pos.weight_pct,
                    pos.sector,
                    pos.asset_class,
                ))

        logger.debug(f"Saved snapshot {snapshot.snapshot_id} ({snapshot.snapshot_type.value})")

    def get_snapshot(self, snapshot_id: str) -> PortfolioSnapshot | None:
        """Get a specific snapshot by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?",
                (snapshot_id,)
            ).fetchone()

            if row is None:
                return None

            # Get positions
            positions = conn.execute(
                "SELECT * FROM positions WHERE snapshot_id = ?",
                (snapshot_id,)
            ).fetchall()

            return self._row_to_snapshot(row, positions)

    def get_snapshots(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        snapshot_type: SnapshotType | None = None,
        limit: int = 100,
    ) -> list[PortfolioSnapshot]:
        """Query snapshots with filters."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM snapshots WHERE 1=1"
            params: list = []

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())

            if snapshot_type:
                query += " AND snapshot_type = ?"
                params.append(snapshot_type.value)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()

            snapshots = []
            for row in rows:
                positions = conn.execute(
                    "SELECT * FROM positions WHERE snapshot_id = ?",
                    (row['snapshot_id'],)
                ).fetchall()
                snapshots.append(self._row_to_snapshot(row, positions))

            return snapshots

    def get_latest_snapshot(
        self,
        snapshot_type: SnapshotType | None = None,
    ) -> PortfolioSnapshot | None:
        """Get the most recent snapshot."""
        snapshots = self.get_snapshots(snapshot_type=snapshot_type, limit=1)
        return snapshots[0] if snapshots else None

    def get_eod_snapshots(
        self,
        days: int = 30,
    ) -> list[PortfolioSnapshot]:
        """Get end-of-day snapshots for the last N days."""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        return self.get_snapshots(
            start_date=start_date,
            snapshot_type=SnapshotType.EOD,
            limit=days,
        )

    def _row_to_snapshot(self, row: sqlite3.Row, position_rows: list) -> PortfolioSnapshot:
        """Convert database rows to PortfolioSnapshot."""
        positions = [
            PositionSnapshot(
                symbol=p['symbol'],
                quantity=p['quantity'],
                avg_cost=p['avg_cost'],
                market_price=p['market_price'],
                market_value=p['market_value'],
                unrealized_pnl=p['unrealized_pnl'],
                realized_pnl=p['realized_pnl'],
                weight_pct=p['weight_pct'],
                sector=p['sector'] or "",
                asset_class=p['asset_class'] or "",
            )
            for p in position_rows
        ]

        return PortfolioSnapshot(
            snapshot_id=row['snapshot_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            snapshot_type=SnapshotType(row['snapshot_type']),
            net_liquidation=row['net_liquidation'],
            gross_exposure=row['gross_exposure'],
            net_exposure=row['net_exposure'],
            cash=row['cash'],
            daily_pnl=row['daily_pnl'],
            mtd_pnl=row['mtd_pnl'],
            ytd_pnl=row['ytd_pnl'],
            inception_pnl=row['inception_pnl'],
            var_95=row['var_95'],
            var_99=row['var_99'],
            expected_shortfall=row['expected_shortfall'],
            current_drawdown=row['current_drawdown'],
            max_drawdown=row['max_drawdown'],
            portfolio_delta=row['portfolio_delta'] or 0.0,
            portfolio_gamma=row['portfolio_gamma'] or 0.0,
            portfolio_vega=row['portfolio_vega'] or 0.0,
            portfolio_theta=row['portfolio_theta'] or 0.0,
            positions=positions,
            trigger_reason=row['trigger_reason'] or "",
            notes=row['notes'] or "",
        )

    def delete_old_snapshots(
        self,
        retention_days: int = 365 * 7,  # 7 years for compliance
        keep_eod: bool = True,
    ) -> int:
        """Delete old snapshots while respecting retention policy."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

        with sqlite3.connect(self.db_path) as conn:
            if keep_eod:
                # Only delete intraday snapshots older than cutoff
                result = conn.execute("""
                    DELETE FROM snapshots
                    WHERE timestamp < ? AND snapshot_type = ?
                """, (cutoff.isoformat(), SnapshotType.INTRADAY.value))
            else:
                result = conn.execute(
                    "DELETE FROM snapshots WHERE timestamp < ?",
                    (cutoff.isoformat(),)
                )

            deleted = result.rowcount

            # Clean up orphaned positions
            conn.execute("""
                DELETE FROM positions
                WHERE snapshot_id NOT IN (SELECT snapshot_id FROM snapshots)
            """)

        logger.info(f"Deleted {deleted} old snapshots")
        return deleted

    def export_to_json(
        self,
        filepath: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """Export snapshots to JSON file."""
        snapshots = self.get_snapshots(start_date=start_date, end_date=end_date, limit=10000)

        data = [s.to_dict() for s in snapshots]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported {len(snapshots)} snapshots to {filepath}")
        return len(snapshots)

    def get_statistics(self) -> dict:
        """Get storage statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]

            by_type = {}
            for row in conn.execute("""
                SELECT snapshot_type, COUNT(*)
                FROM snapshots
                GROUP BY snapshot_type
            """):
                by_type[row[0]] = row[1]

            oldest = conn.execute(
                "SELECT MIN(timestamp) FROM snapshots"
            ).fetchone()[0]

            newest = conn.execute(
                "SELECT MAX(timestamp) FROM snapshots"
            ).fetchone()[0]

        return {
            'total_snapshots': total,
            'by_type': by_type,
            'oldest_snapshot': oldest,
            'newest_snapshot': newest,
            'db_path': str(self.db_path),
            'db_size_mb': self.db_path.stat().st_size / 1024 / 1024 if self.db_path.exists() else 0,
        }


class PortfolioSnapshotManager:
    """
    High-level manager for portfolio snapshots (#P17).

    Handles automatic snapshot scheduling and comparison.
    """

    def __init__(
        self,
        store: PortfolioSnapshotStore | None = None,
        intraday_interval_minutes: int = 60,
    ):
        self.store = store or PortfolioSnapshotStore()
        self.intraday_interval = timedelta(minutes=intraday_interval_minutes)
        self._last_intraday_snapshot: datetime | None = None
        self._snapshot_counter = 0

    def _generate_snapshot_id(self, snapshot_type: SnapshotType) -> str:
        """Generate unique snapshot ID."""
        self._snapshot_counter += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{snapshot_type.value}_{timestamp}_{self._snapshot_counter}"

    def should_take_intraday_snapshot(self) -> bool:
        """Check if it's time for an intraday snapshot."""
        if self._last_intraday_snapshot is None:
            return True

        elapsed = datetime.now(timezone.utc) - self._last_intraday_snapshot
        return elapsed >= self.intraday_interval

    def create_snapshot(
        self,
        snapshot_type: SnapshotType,
        risk_state: Any,  # RiskState from risk_agent
        positions: dict[str, Any],  # Symbol -> position data
        trigger_reason: str = "",
        notes: str = "",
    ) -> PortfolioSnapshot:
        """Create a new portfolio snapshot from current state."""
        now = datetime.now(timezone.utc)

        # Build position snapshots
        position_snapshots = []
        total_value = risk_state.net_liquidation or 0

        for symbol, pos in positions.items():
            market_value = getattr(pos, 'market_value', 0) or (
                getattr(pos, 'quantity', 0) * getattr(pos, 'market_price', 0)
            )

            position_snapshots.append(PositionSnapshot(
                symbol=symbol,
                quantity=getattr(pos, 'quantity', 0),
                avg_cost=getattr(pos, 'avg_cost', 0),
                market_price=getattr(pos, 'market_price', 0),
                market_value=market_value,
                unrealized_pnl=getattr(pos, 'unrealized_pnl', 0),
                realized_pnl=getattr(pos, 'realized_pnl', 0),
                weight_pct=(market_value / total_value * 100) if total_value > 0 else 0,
                sector=getattr(pos, 'sector', ""),
                asset_class=getattr(pos, 'asset_class', ""),
            ))

        snapshot = PortfolioSnapshot(
            snapshot_id=self._generate_snapshot_id(snapshot_type),
            timestamp=now,
            snapshot_type=snapshot_type,
            net_liquidation=risk_state.net_liquidation or 0,
            gross_exposure=risk_state.gross_exposure or 0,
            net_exposure=risk_state.net_exposure or 0,
            cash=getattr(risk_state, 'cash', 0),
            daily_pnl=risk_state.daily_pnl or 0,
            mtd_pnl=getattr(risk_state, 'mtd_pnl', 0),
            ytd_pnl=getattr(risk_state, 'ytd_pnl', 0),
            inception_pnl=getattr(risk_state, 'inception_pnl', 0),
            var_95=risk_state.var_95 or 0,
            var_99=risk_state.var_99 or 0,
            expected_shortfall=risk_state.expected_shortfall or 0,
            current_drawdown=risk_state.current_drawdown or 0,
            max_drawdown=risk_state.max_drawdown or 0,
            portfolio_delta=getattr(risk_state, 'portfolio_delta', 0),
            portfolio_gamma=getattr(risk_state, 'portfolio_gamma', 0),
            portfolio_vega=getattr(risk_state, 'portfolio_vega', 0),
            portfolio_theta=getattr(risk_state, 'portfolio_theta', 0),
            positions=position_snapshots,
            trigger_reason=trigger_reason,
            notes=notes,
        )

        # Save to store
        self.store.save_snapshot(snapshot)

        # Update tracking
        if snapshot_type == SnapshotType.INTRADAY:
            self._last_intraday_snapshot = now

        logger.info(
            f"Created {snapshot_type.value} snapshot: "
            f"NLV=${snapshot.net_liquidation:,.0f}, "
            f"positions={len(position_snapshots)}"
        )

        return snapshot

    def compare_snapshots(
        self,
        snapshot1: PortfolioSnapshot,
        snapshot2: PortfolioSnapshot,
    ) -> dict:
        """Compare two snapshots and return differences (#P18 partial)."""
        changes = {
            'time_delta': (snapshot2.timestamp - snapshot1.timestamp).total_seconds() / 3600,  # hours
            'portfolio': {
                'net_liquidation': {
                    'from': snapshot1.net_liquidation,
                    'to': snapshot2.net_liquidation,
                    'change': snapshot2.net_liquidation - snapshot1.net_liquidation,
                    'change_pct': ((snapshot2.net_liquidation / snapshot1.net_liquidation) - 1) * 100
                        if snapshot1.net_liquidation else 0,
                },
                'gross_exposure': {
                    'from': snapshot1.gross_exposure,
                    'to': snapshot2.gross_exposure,
                    'change': snapshot2.gross_exposure - snapshot1.gross_exposure,
                },
                'current_drawdown': {
                    'from': snapshot1.current_drawdown,
                    'to': snapshot2.current_drawdown,
                    'change': snapshot2.current_drawdown - snapshot1.current_drawdown,
                },
            },
            'positions': {
                'added': [],
                'removed': [],
                'changed': [],
            },
        }

        # Compare positions
        symbols1 = {p.symbol for p in snapshot1.positions}
        symbols2 = {p.symbol for p in snapshot2.positions}

        # Added positions
        for symbol in symbols2 - symbols1:
            pos = snapshot2.get_position(symbol)
            if pos:
                changes['positions']['added'].append({
                    'symbol': symbol,
                    'quantity': pos.quantity,
                    'market_value': pos.market_value,
                })

        # Removed positions
        for symbol in symbols1 - symbols2:
            pos = snapshot1.get_position(symbol)
            if pos:
                changes['positions']['removed'].append({
                    'symbol': symbol,
                    'quantity': pos.quantity,
                    'market_value': pos.market_value,
                })

        # Changed positions
        for symbol in symbols1 & symbols2:
            pos1 = snapshot1.get_position(symbol)
            pos2 = snapshot2.get_position(symbol)

            if pos1 and pos2 and pos1.quantity != pos2.quantity:
                changes['positions']['changed'].append({
                    'symbol': symbol,
                    'quantity_from': pos1.quantity,
                    'quantity_to': pos2.quantity,
                    'quantity_change': pos2.quantity - pos1.quantity,
                    'value_change': pos2.market_value - pos1.market_value,
                })

        return changes

    def get_performance_history(self, days: int = 30) -> list[dict]:
        """Get daily performance history."""
        snapshots = self.store.get_eod_snapshots(days)

        history = []
        for snapshot in reversed(snapshots):  # Oldest first
            history.append({
                'date': snapshot.timestamp.date().isoformat(),
                'net_liquidation': snapshot.net_liquidation,
                'daily_pnl': snapshot.daily_pnl,
                'mtd_pnl': snapshot.mtd_pnl,
                'ytd_pnl': snapshot.ytd_pnl,
                'drawdown': snapshot.current_drawdown,
                'var_95': snapshot.var_95,
            })

        return history

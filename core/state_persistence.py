"""
State Persistence
=================

Serialize CIO state to disk for hot restart capability.
Enables system recovery without losing position tracking, strategy weights,
and correlation data.

Features:
- Atomic writes (temp file + rename)
- Backup rotation (5 max)
- JSON serialization with datetime handling
- Automatic loading on startup
- Periodic auto-save
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrackedPositionState:
    """Serializable state for a tracked position."""
    symbol: str
    entry_price: float
    current_price: float
    quantity: float
    side: str  # "long" or "short"
    entry_time: str  # ISO format
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    initial_risk: float = 0.0
    original_conviction: float = 0.5
    current_conviction: float = 0.5


@dataclass
class StrategyPerformanceState:
    """Serializable state for strategy performance."""
    strategy_name: str
    total_signals: int = 0
    profitable_signals: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_r_multiple: float = 0.0


@dataclass
class CIOPersistedState:
    """
    Complete CIO state for persistence.

    This captures everything needed to restore CIO to its previous state
    after a restart.
    """
    # Strategy weights (agent_name -> weight)
    current_weights: Dict[str, float] = field(default_factory=dict)

    # Strategy performance tracking
    strategy_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Signal correlation matrix (serialized as agent pair -> correlation)
    signal_correlation_matrix: Dict[str, float] = field(default_factory=dict)

    # Current market regime
    current_regime: str = "normal"

    # VIX-related state
    last_vix_level: float = 20.0
    vix_regime: str = "normal"

    # Position tracking
    tracked_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Decision history (limited to last N)
    recent_decisions: List[Dict[str, Any]] = field(default_factory=list)

    # Performance metrics
    total_decisions: int = 0
    profitable_decisions: int = 0
    total_pnl: float = 0.0

    # R-multiple stats
    r_multiple_exits_2r: int = 0
    r_multiple_exits_3r: int = 0
    total_r_multiple_closed: float = 0.0
    closed_trade_count: int = 0

    # Metadata
    persisted_at: str = ""
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CIOPersistedState":
        """Create from dictionary."""
        # Handle version migration if needed
        version = data.get("version", "1.0")

        return cls(
            current_weights=data.get("current_weights", {}),
            strategy_performance=data.get("strategy_performance", {}),
            signal_correlation_matrix=data.get("signal_correlation_matrix", {}),
            current_regime=data.get("current_regime", "normal"),
            last_vix_level=data.get("last_vix_level", 20.0),
            vix_regime=data.get("vix_regime", "normal"),
            tracked_positions=data.get("tracked_positions", {}),
            recent_decisions=data.get("recent_decisions", []),
            total_decisions=data.get("total_decisions", 0),
            profitable_decisions=data.get("profitable_decisions", 0),
            total_pnl=data.get("total_pnl", 0.0),
            r_multiple_exits_2r=data.get("r_multiple_exits_2r", 0),
            r_multiple_exits_3r=data.get("r_multiple_exits_3r", 0),
            total_r_multiple_closed=data.get("total_r_multiple_closed", 0.0),
            closed_trade_count=data.get("closed_trade_count", 0),
            persisted_at=data.get("persisted_at", ""),
            version=version,
        )


@dataclass
class StatePersistenceConfig:
    """Configuration for state persistence."""
    state_dir: str = "state"
    cio_state_file: str = "cio_state.json"
    max_backups: int = 5
    auto_save_interval_seconds: float = 60.0
    max_recent_decisions: int = 100


class StatePersistence:
    """
    State Persistence Manager for CIO hot restart.

    Handles:
    - Atomic file writes (temp + rename)
    - Backup rotation
    - Automatic state loading
    - Periodic auto-save
    """

    def __init__(self, config: StatePersistenceConfig | None = None):
        self._config = config or StatePersistenceConfig()
        self._state_dir = Path(self._config.state_dir)
        self._cio_state_path = self._state_dir / self._config.cio_state_file

        # Ensure state directory exists
        self._state_dir.mkdir(parents=True, exist_ok=True)

        # Track last save time
        self._last_save_time: datetime | None = None

        # Thread safety for concurrent save operations
        self._lock = threading.RLock()

        logger.info(f"StatePersistence initialized at {self._state_dir}")

    def save_cio_state(self, state: CIOPersistedState) -> bool:
        """
        Save CIO state to disk with atomic write.

        Uses temp file + rename pattern for crash safety.
        Thread-safe: uses lock to prevent concurrent save/rotate corruption.

        Args:
            state: CIO state to persist

        Returns:
            True if save succeeded
        """
        with self._lock:
            try:
                # Update persistence timestamp
                state.persisted_at = datetime.now(timezone.utc).isoformat()

                # Limit recent decisions to max
                if len(state.recent_decisions) > self._config.max_recent_decisions:
                    state.recent_decisions = state.recent_decisions[-self._config.max_recent_decisions:]

                # Serialize to JSON
                state_dict = state.to_dict()
                json_data = json.dumps(state_dict, indent=2, default=str)

                # Atomic write: temp file + rename
                temp_path = self._cio_state_path.with_suffix(".tmp")

                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(json_data)
                    f.flush()
                    os.fsync(f.fileno())

                # Backup existing file before replacing
                if self._cio_state_path.exists():
                    self._rotate_backups()

                # Atomic rename
                temp_path.replace(self._cio_state_path)

                self._last_save_time = datetime.now(timezone.utc)
                logger.debug(f"CIO state saved to {self._cio_state_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to save CIO state: {e}")
                # Clean up temp file if it exists
                temp_path = self._cio_state_path.with_suffix(".tmp")
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                return False

    def load_cio_state(self) -> CIOPersistedState | None:
        """
        Load CIO state from disk.

        Returns:
            Loaded state or None if not found/invalid
        """
        if not self._cio_state_path.exists():
            logger.info("No persisted CIO state found")
            return None

        try:
            with open(self._cio_state_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            state = CIOPersistedState.from_dict(data)
            logger.info(
                f"Loaded CIO state from {state.persisted_at} "
                f"(v{state.version}, {len(state.tracked_positions)} positions)"
            )
            return state

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in CIO state file: {e}")
            # Try loading from backup
            return self._load_from_backup()

        except Exception as e:
            logger.error(f"Failed to load CIO state: {e}")
            return self._load_from_backup()

    def _load_from_backup(self) -> CIOPersistedState | None:
        """Try to load state from most recent backup."""
        for i in range(1, self._config.max_backups + 1):
            backup_path = self._cio_state_path.with_suffix(f".bak{i}")
            if backup_path.exists():
                try:
                    with open(backup_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    state = CIOPersistedState.from_dict(data)
                    logger.warning(f"Loaded CIO state from backup {backup_path}")
                    return state
                except Exception as e:
                    logger.debug(f"Failed to load backup {backup_path}: {e}")
                    continue
        return None

    def _rotate_backups(self) -> None:
        """Rotate backup files."""
        try:
            # Shift existing backups
            for i in range(self._config.max_backups, 1, -1):
                older = self._cio_state_path.with_suffix(f".bak{i-1}")
                newer = self._cio_state_path.with_suffix(f".bak{i}")
                if older.exists():
                    shutil.move(str(older), str(newer))

            # Move current to .bak1
            if self._cio_state_path.exists():
                shutil.copy2(
                    str(self._cio_state_path),
                    str(self._cio_state_path.with_suffix(".bak1"))
                )

        except Exception as e:
            logger.warning(f"Failed to rotate backups: {e}")

    def should_auto_save(self) -> bool:
        """Check if auto-save is due based on interval."""
        if self._last_save_time is None:
            return True

        elapsed = (datetime.now(timezone.utc) - self._last_save_time).total_seconds()
        return elapsed >= self._config.auto_save_interval_seconds

    def delete_state(self) -> bool:
        """Delete persisted state and backups."""
        try:
            if self._cio_state_path.exists():
                self._cio_state_path.unlink()

            # Delete backups
            for i in range(1, self._config.max_backups + 1):
                backup = self._cio_state_path.with_suffix(f".bak{i}")
                if backup.exists():
                    backup.unlink()

            logger.info("Deleted all persisted CIO state")
            return True

        except Exception as e:
            logger.error(f"Failed to delete state: {e}")
            return False

    def get_state_info(self) -> Dict[str, Any]:
        """Get information about persisted state."""
        result = {
            "state_file": str(self._cio_state_path),
            "exists": self._cio_state_path.exists(),
            "last_save": self._last_save_time.isoformat() if self._last_save_time else None,
            "backups": [],
        }

        if self._cio_state_path.exists():
            stat = self._cio_state_path.stat()
            result["size_bytes"] = stat.st_size
            result["modified"] = datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat()

        # Check backups
        for i in range(1, self._config.max_backups + 1):
            backup = self._cio_state_path.with_suffix(f".bak{i}")
            if backup.exists():
                result["backups"].append({
                    "name": backup.name,
                    "size_bytes": backup.stat().st_size,
                })

        return result


def create_state_persistence(config: Dict[str, Any] | None = None) -> StatePersistence:
    """Factory function to create StatePersistence from config dict."""
    if config is None:
        return StatePersistence()

    persistence_config = StatePersistenceConfig(
        state_dir=config.get("state_dir", "state"),
        cio_state_file=config.get("cio_state_file", "cio_state.json"),
        max_backups=config.get("max_backups", 5),
        auto_save_interval_seconds=config.get("auto_save_interval_seconds", 60.0),
        max_recent_decisions=config.get("max_recent_decisions", 100),
    )
    return StatePersistence(persistence_config)

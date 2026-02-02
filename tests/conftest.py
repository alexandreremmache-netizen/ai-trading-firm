"""
Pytest Configuration
====================

Shared fixtures for testing.
"""

import pytest
import asyncio
from pathlib import Path


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Minimal test configuration."""
    return {
        "firm": {
            "name": "Test Trading Firm",
            "mode": "paper",
        },
        "broker": {
            "host": "127.0.0.1",
            "port": 7497,
            "client_id": 999,
        },
        "event_bus": {
            "max_queue_size": 100,
            "signal_timeout_seconds": 1.0,
            "sync_barrier_timeout_seconds": 2.0,
        },
        "risk": {
            "max_position_size_pct": 5.0,
            "max_daily_loss_pct": 3.0,
            "max_drawdown_pct": 10.0,
        },
        "agents": {
            "macro": {"enabled": True},
            "stat_arb": {"enabled": True},
            "momentum": {"enabled": True},
            "market_making": {"enabled": False},
            "options_vol": {"enabled": False},
        },
    }


@pytest.fixture
def temp_logs_dir(tmp_path):
    """Create temporary logs directory."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    return logs_dir

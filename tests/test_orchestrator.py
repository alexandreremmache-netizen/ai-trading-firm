"""
Tests for Main Orchestrator
===========================

Tests for main.py orchestrator functionality (Issue #22).
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path


class TestTradingFirmOrchestrator:
    """Test TradingFirmOrchestrator functionality."""

    def test_orchestrator_import(self):
        """Test that orchestrator can be imported."""
        from main import TradingFirmOrchestrator
        assert TradingFirmOrchestrator is not None

    def test_orchestrator_creation(self, tmp_path):
        """Test orchestrator can be created with config path."""
        from main import TradingFirmOrchestrator

        # Create a minimal config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
firm:
  name: Test Firm
  mode: paper
broker:
  host: 127.0.0.1
  port: 7497
  client_id: 999
event_bus:
  max_queue_size: 100
  signal_timeout_seconds: 1.0
  sync_barrier_timeout_seconds: 2.0
risk:
  max_position_size_pct: 5.0
  max_daily_loss_pct: 3.0
  max_drawdown_pct: 10.0
agents:
  macro:
    enabled: false
  stat_arb:
    enabled: false
  momentum:
    enabled: false
  market_making:
    enabled: false
  options_vol:
    enabled: false
  surveillance:
    enabled: false
  transaction_reporting:
    enabled: false
logging:
  audit_file: logs/audit.jsonl
  trade_file: logs/trades.jsonl
  decision_file: logs/decisions.jsonl
universe:
  equities: []
  etfs: []
  futures: []
  forex: []
""")

        orchestrator = TradingFirmOrchestrator(config_path=str(config_file))
        assert orchestrator is not None
        assert orchestrator._running is False

    def test_orchestrator_initial_state(self, tmp_path):
        """Test orchestrator starts in correct initial state."""
        from main import TradingFirmOrchestrator

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
firm:
  name: Test
  mode: paper
broker:
  host: 127.0.0.1
  port: 7497
  client_id: 1
""")

        orchestrator = TradingFirmOrchestrator(config_path=str(config_file))

        # Check initial state
        assert orchestrator._running is False
        assert orchestrator._event_bus is None
        assert orchestrator._broker is None
        assert orchestrator._startup_time is None

    def test_orchestrator_get_status_before_init(self, tmp_path):
        """Test get_status works before initialization."""
        from main import TradingFirmOrchestrator

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
firm:
  name: Test
  mode: paper
broker:
  host: 127.0.0.1
  port: 7497
  client_id: 1
""")

        orchestrator = TradingFirmOrchestrator(config_path=str(config_file))
        status = orchestrator.get_status()

        assert status["running"] is False
        assert status["startup_time"] is None
        assert status["broker"]["connected"] is False

    def test_orchestrator_request_shutdown(self, tmp_path):
        """Test request_shutdown sets shutdown event."""
        from main import TradingFirmOrchestrator

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
firm:
  name: Test
  mode: paper
broker:
  host: 127.0.0.1
  port: 7497
  client_id: 1
""")

        orchestrator = TradingFirmOrchestrator(config_path=str(config_file))
        orchestrator.request_shutdown()

        assert orchestrator._shutdown_event.is_set()


class TestOrchestratorConfig:
    """Test orchestrator configuration loading."""

    def test_load_config_missing_file(self, tmp_path):
        """Test error when config file doesn't exist."""
        from main import TradingFirmOrchestrator

        orchestrator = TradingFirmOrchestrator(
            config_path=str(tmp_path / "nonexistent.yaml")
        )

        with pytest.raises(FileNotFoundError):
            orchestrator._load_config()

    def test_load_config_valid_file(self, tmp_path):
        """Test loading valid config file."""
        from main import TradingFirmOrchestrator

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
firm:
  name: Test Firm
  mode: paper
broker:
  host: 127.0.0.1
  port: 7497
  client_id: 1
""")

        orchestrator = TradingFirmOrchestrator(config_path=str(config_file))
        config = orchestrator._load_config()

        assert config["firm"]["name"] == "Test Firm"
        assert config["firm"]["mode"] == "paper"
        assert config["broker"]["port"] == 7497


class TestOrchestratorAgents:
    """Test orchestrator agent management."""

    def test_signal_agents_list_initialized(self, tmp_path):
        """Test signal agents list is initialized empty."""
        from main import TradingFirmOrchestrator

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
firm:
  name: Test
  mode: paper
broker:
  host: 127.0.0.1
  port: 7497
  client_id: 1
""")

        orchestrator = TradingFirmOrchestrator(config_path=str(config_file))

        assert orchestrator._signal_agents == []
        assert orchestrator._cio_agent is None
        assert orchestrator._risk_agent is None
        assert orchestrator._compliance_agent is None
        assert orchestrator._execution_agent is None


class TestSetupSignalHandlers:
    """Test signal handler setup."""

    def test_signal_handlers_import(self):
        """Test signal handlers function can be imported."""
        from main import setup_signal_handlers
        assert setup_signal_handlers is not None

    def test_signal_handlers_setup(self, tmp_path):
        """Test signal handlers can be set up."""
        from main import TradingFirmOrchestrator, setup_signal_handlers

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
firm:
  name: Test
  mode: paper
broker:
  host: 127.0.0.1
  port: 7497
  client_id: 1
""")

        orchestrator = TradingFirmOrchestrator(config_path=str(config_file))

        # Should not raise
        setup_signal_handlers(orchestrator)


class TestMainFunction:
    """Test main entry point."""

    def test_main_import(self):
        """Test main function can be imported."""
        from main import main
        assert main is not None
        assert asyncio.iscoroutinefunction(main)

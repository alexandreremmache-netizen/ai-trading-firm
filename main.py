#!/usr/bin/env python3
"""
AI Trading Firm - Main Orchestrator
====================================

Entry point for the multi-agent trading system.

This orchestrator:
1. Loads configuration
2. Initializes all components (broker, event bus, logger, monitoring)
3. Starts all agents in correct order
4. Manages the event-driven execution loop
5. Handles graceful shutdown and kill-switch

Per the constitution (CLAUDE.md):
- Event-driven, no infinite loops
- Signal agents run in parallel (fan-out)
- CIO decision after synchronization barrier (fan-in)
- Risk, Compliance, Execution run sequentially
- All decisions logged with full audit trail
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import signal
import sys
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

# NumPy availability check for graceful degradation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logging.warning(
        "NumPy not available - some quantitative features will be disabled. "
        "Install with: pip install numpy"
    )

import yaml

from core.event_bus import EventBus, AgentCriticality
from core.broker import IBBroker, BrokerConfig
from core.logger import AuditLogger
from core.agent_base import AgentConfig
from core.events import EventType, RiskAlertEvent, RiskAlertSeverity, FillEvent, MarketDataEvent
from core.monitoring import MonitoringSystem
from core.health_check import (
    HealthChecker,
    HealthCheckServer,
    HealthCheckConfig,
    create_health_check_server,
)
from core.notifications import (
    NotificationManager,
    ComplianceOfficerNotifier,
    RiskLimitBreachNotifier,
    FileNotificationChannel,
    WebhookNotificationChannel,
    AlertSeverity,
    AlertCategory,
)
from core.infrastructure_ops import AuditLogBackupManager
from core.immutable_ledger import ImmutableAuditLedger, create_audit_ledger

# New Infrastructure Components
from core.contract_specs import ContractSpecsManager
from core.correlation_manager import CorrelationManager
from core.futures_roll_manager import FuturesRollManager
from core.var_calculator import VaRCalculator
from core.stress_tester import StressTester
from core.position_sizing import PositionSizer
from core.attribution import PerformanceAttribution
from core.best_execution import BestExecutionAnalyzer
from core.stop_loss_manager import StopLossManager, StopLossConfig

# Signal Agents (parallel - fan-out)
from agents.macro_agent import MacroAgent
from agents.stat_arb_agent import StatArbAgent
from agents.momentum_agent import MomentumAgent
from agents.market_making_agent import MarketMakingAgent

# Decision Agent (single authority)
from agents.cio_agent import CIOAgent

# Validation Agents (sequential after CIO)
from agents.risk_agent import RiskAgent
from agents.compliance_agent import ComplianceAgent

# Execution Agent (only one that sends orders)
from agents.execution_agent import ExecutionAgentImpl

# New Compliance Agents (EU/AMF)
from agents.surveillance_agent import SurveillanceAgent
from agents.transaction_reporting_agent import TransactionReportingAgent
from agents.reconciliation_agent import ReconciliationAgent

from data.market_data import MarketDataManager, SymbolConfig

# Economic Calendar for EventDrivenAgent
from core.economic_calendar import EconomicCalendar, create_economic_calendar

# Historical Data Warmup for instant agent readiness
from core.historical_warmup import HistoricalWarmup, create_historical_warmup

# Dashboard integration for real-time monitoring
from dashboard.server import DashboardServer, create_dashboard_server
from dashboard.broker_sync import BrokerMetricsSync, create_broker_sync
from dashboard_v2.server import DashboardV2Server, create_dashboard_v2_server

# SOLID Refactoring: AgentFactory for improved SRP
from core.agent_factory import AgentFactory, AgentFactoryConfig

# SOLID Refactoring: DI Container (optional usage)
from core.dependency_injection import DIContainer, ServiceProvider


logger = logging.getLogger(__name__)


class TradingFirmOrchestrator:
    """
    Main orchestrator for the AI Trading Firm.

    Manages the lifecycle of all agents and coordinates
    the event-driven execution model per CLAUDE.md.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     MARKET DATA (IB)                        │
    └─────────────────────────┬───────────────────────────────────┘
                              │ MarketDataEvent
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              SIGNAL AGENTS (parallel fan-out)               │
    │  [Macro] [StatArb] [Momentum] [MarketMaking] [MACDv]        │
    └─────────────────────────┬───────────────────────────────────┘
                              │ SignalEvent (barrier sync)
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    CIO AGENT (single)                       │
    │              THE decision-making authority                  │
    └─────────────────────────┬───────────────────────────────────┘
                              │ DecisionEvent
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    RISK AGENT                               │
    │     Kill-switch, VaR, position/leverage limits             │
    └─────────────────────────┬───────────────────────────────────┘
                              │ ValidatedDecisionEvent
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 COMPLIANCE AGENT (EU/AMF)                   │
    │        Blackout, MNPI, restricted instruments              │
    └─────────────────────────┬───────────────────────────────────┘
                              │ ValidatedDecisionEvent
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  EXECUTION AGENT                            │
    │      TWAP/VWAP algorithms - ONLY one sending orders        │
    └─────────────────────────┬───────────────────────────────────┘
                              │ OrderEvent
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  INTERACTIVE BROKERS                        │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config_path: str = "config.yaml"):
        self._config_path = config_path
        self._config: dict[str, Any] = {}

        # Core components
        self._event_bus: EventBus | None = None
        self._broker: IBBroker | None = None
        self._audit_logger: AuditLogger | None = None
        self._market_data: MarketDataManager | None = None

        # Monitoring system
        self._monitoring: MonitoringSystem | None = None

        # Health check server (#S5)
        self._health_checker: HealthChecker | None = None
        self._health_server: HealthCheckServer | None = None

        # Notification system (#C33, #R27)
        self._notification_manager: NotificationManager | None = None
        self._compliance_notifier: ComplianceOfficerNotifier | None = None
        self._risk_notifier: RiskLimitBreachNotifier | None = None

        # Audit log backup manager
        self._backup_manager: AuditLogBackupManager | None = None

        # Immutable audit ledger for MiFID II compliance
        self._audit_ledger: ImmutableAuditLedger | None = None

        # Dashboard server for real-time monitoring
        self._dashboard_server: DashboardServer | None = None
        self._dashboard_v2_server: DashboardV2Server | None = None
        self._broker_sync: BrokerMetricsSync | None = None

        # Infrastructure components
        self._contract_specs: ContractSpecsManager | None = None
        self._correlation_manager: CorrelationManager | None = None
        self._futures_roll_manager: FuturesRollManager | None = None
        self._var_calculator: VaRCalculator | None = None
        self._stress_tester: StressTester | None = None
        self._position_sizer: PositionSizer | None = None
        self._attribution: PerformanceAttribution | None = None
        self._best_execution: BestExecutionAnalyzer | None = None
        self._stop_loss_manager: StopLossManager | None = None
        self._economic_calendar: EconomicCalendar | None = None
        self._historical_warmup: HistoricalWarmup | None = None

        # Agents
        self._signal_agents: list[Any] = []
        self._cio_agent: CIOAgent | None = None
        self._risk_agent: RiskAgent | None = None
        self._compliance_agent: ComplianceAgent | None = None
        self._execution_agent: ExecutionAgentImpl | None = None
        self._surveillance_agent: SurveillanceAgent | None = None
        self._transaction_reporting_agent: TransactionReportingAgent | None = None
        self._reconciliation_agent: ReconciliationAgent | None = None

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._startup_time: datetime | None = None
        self._dry_run: bool = False
        self._shutdown_in_progress: bool = False
        self._graceful_shutdown_timeout: float = 30.0

    def set_dry_run(self, dry_run: bool) -> None:
        """Enable dry-run mode (validate config without starting)."""
        self._dry_run = dry_run

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self._config_path)

        if not config_file.exists():
            logger.error(f"Config file not found: {config_file}")
            logger.error("To create a config file, run one of:")
            logger.error(f"  cp config.simple.yaml {config_file}  # Simple config for beginners")
            logger.error(f"  cp config.yaml {config_file}         # Full config with all options")
            logger.error("Or specify a different config with:")
            logger.error("  python main.py --config path/to/config.yaml")
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_file}")
        return config

    def _print_startup_summary(self) -> None:
        """Print startup validation summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("STARTUP VALIDATION SUMMARY")
        logger.info("=" * 60)

        # Configuration summary
        mode = self._config.get("firm", {}).get("mode", "paper")
        broker_host = self._config.get("broker", {}).get("host", "127.0.0.1")
        broker_port = self._config.get("broker", {}).get("port", 4002)

        logger.info(f"  Mode: {mode.upper()}")
        logger.info(f"  Broker: {broker_host}:{broker_port}")

        # Risk limits summary
        risk = self._config.get("risk", {})
        logger.info(f"  Max Position Size: {risk.get('max_position_size_pct', 5.0)}%")
        logger.info(f"  Max Daily Loss: {risk.get('max_daily_loss_pct', 3.0)}%")
        logger.info(f"  Max Drawdown: {risk.get('max_drawdown_pct', 10.0)}%")
        logger.info(f"  Max Leverage: {risk.get('max_leverage', 2.0)}x")

        # Universe summary
        universe = self._config.get("universe", {})
        equities_count = len(universe.get("equities", []))
        etfs_count = len(universe.get("etfs", []))
        futures_count = len(universe.get("futures", []))
        forex_count = len(universe.get("forex", []))
        total_instruments = equities_count + etfs_count + futures_count + forex_count

        logger.info(f"  Trading Universe: {total_instruments} instruments")
        logger.info(f"    - Equities: {equities_count}")
        logger.info(f"    - ETFs: {etfs_count}")
        logger.info(f"    - Futures: {futures_count}")
        logger.info(f"    - Forex: {forex_count}")

        # Agents summary
        agents = self._config.get("agents", {})
        enabled_agents = []
        for agent_name in ["macro", "stat_arb", "momentum", "market_making", "options_vol"]:
            if agents.get(agent_name, {}).get("enabled", True):
                enabled_agents.append(agent_name)
        logger.info(f"  Enabled Signal Agents: {len(enabled_agents)}")
        for agent in enabled_agents:
            logger.info(f"    - {agent}")

        # Compliance summary
        compliance = self._config.get("compliance", {})
        logger.info(f"  Jurisdiction: {compliance.get('jurisdiction', 'EU')}")
        logger.info(f"  Regulator: {compliance.get('regulator', 'AMF')}")

        # Transaction reporting
        tx_reporting = self._config.get("transaction_reporting", {})
        lei = tx_reporting.get("firm_lei", "")
        if lei:
            logger.info(f"  LEI: {lei[:4]}...{lei[-4:] if len(lei) > 8 else ''}")
        else:
            logger.warning("  LEI: NOT CONFIGURED (required for live trading)")

        logger.info("=" * 60)
        logger.info("")

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("AI TRADING FIRM - INITIALIZING")
        logger.info("=" * 60)

        self._startup_time = datetime.now(timezone.utc)

        # Load configuration
        self._config = self._load_config()

        # Validate configuration at startup (CRITICAL - Expert 20 fix)
        from core.config_validator import validate_config_at_startup
        try:
            validate_config_at_startup(self._config)
            logger.info("Configuration validation passed")
        except ValueError as e:
            logger.error(f"Configuration validation failed:\n{e}")
            raise

        # Validate mode
        mode = self._config.get("firm", {}).get("mode", "paper")
        if mode == "live":
            logger.critical("=" * 60)
            logger.critical("WARNING: LIVE TRADING MODE DETECTED")
            logger.critical("Real money will be used. This is NOT a simulation!")
            logger.critical("=" * 60)
            # CRITICAL: Require explicit confirmation for live trading
            import sys
            if sys.stdin.isatty():
                confirmation = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
                if confirmation != "CONFIRM LIVE TRADING":
                    logger.error("Live trading mode not confirmed. Exiting for safety.")
                    raise ValueError("Live trading mode requires explicit confirmation")
            else:
                # Non-interactive mode - require environment variable
                import os
                if os.environ.get("CONFIRM_LIVE_TRADING") != "YES":
                    logger.error("Live mode requires CONFIRM_LIVE_TRADING=YES environment variable")
                    raise ValueError("Live trading mode not confirmed via environment")
        else:
            logger.info(f"Running in {mode.upper()} mode")

        # Initialize audit logger first (required for compliance)
        logging_config = self._config.get("logging", {})
        self._audit_logger = AuditLogger(
            audit_file=logging_config.get("audit_file", "logs/audit.jsonl"),
            trade_file=logging_config.get("trade_file", "logs/trades.jsonl"),
            decision_file=logging_config.get("decision_file", "logs/decisions.jsonl"),
        )

        self._audit_logger.log_system_event(
            "startup",
            {"mode": mode, "config_path": self._config_path},
        )

        # Start async writer for non-blocking audit I/O
        await self._audit_logger.start_async_writer()

        # Wire data retention tracking for MiFID II compliance
        try:
            from core.data_retention import DataRetentionManager, create_retention_aware_logger
            self._retention_manager = DataRetentionManager(
                db_path="logs/retention.db",
                archive_path="archive",
            )
            create_retention_aware_logger(self._audit_logger, self._retention_manager)
            logger.info("Data retention tracking enabled (MiFID II 7-year)")
        except Exception as e:
            logger.warning(f"Data retention tracking not available: {e}")

        # Initialize monitoring system
        await self._initialize_monitoring()

        # Initialize health check server (#S5)
        await self._initialize_health_check()

        # Initialize event bus
        event_bus_config = self._config.get("event_bus", {})

        # Parse agent criticality from config.yaml
        criticality_map = {
            "critical": AgentCriticality.CRITICAL,
            "high": AgentCriticality.HIGH,
            "normal": AgentCriticality.NORMAL,
        }
        agent_criticality_config = event_bus_config.get("agent_criticality", {})
        agent_criticality = {
            name: criticality_map.get(level.lower(), AgentCriticality.NORMAL)
            for name, level in agent_criticality_config.items()
        } if agent_criticality_config else None

        self._event_bus = EventBus(
            max_queue_size=event_bus_config.get("max_queue_size", 10000),
            signal_timeout=event_bus_config.get("signal_timeout_seconds", 5.0),
            barrier_timeout=event_bus_config.get("sync_barrier_timeout_seconds", 10.0),
            agent_criticality=agent_criticality,
        )

        # CRITICAL: Initialize and connect immutable audit ledger for MiFID II compliance
        # This ensures ALL trading decisions, orders, and kill switch activations are recorded
        audit_ledger_config = self._config.get("audit_ledger", {})
        ledger_path = audit_ledger_config.get("storage_path", "logs/audit_ledger")
        self._audit_ledger = create_audit_ledger(storage_path=ledger_path)
        self._event_bus.set_audit_ledger(self._audit_ledger)
        logger.info(f"Immutable audit ledger connected for MiFID II compliance (path: {ledger_path})")

        # Initialize dashboard server (must be after event bus creation)
        await self._initialize_dashboard()

        # Initialize broker connection
        await self._initialize_broker()

        # Initialize market data manager
        await self._initialize_market_data()

        # Initialize infrastructure components
        await self._initialize_infrastructure()

        # Initialize agents
        await self._initialize_agents()

        # Wire components together
        self._wire_components()

        # Wire event bus to health checker (#S5)
        if self._health_checker and self._event_bus:
            self._health_checker.set_event_bus(self._event_bus)

        # Mark system as ready for health checks
        if self._health_checker:
            self._health_checker.set_ready(True)

        # Print startup validation summary
        self._print_startup_summary()

        # Dry-run mode: stop after validation
        if self._dry_run:
            logger.info("=" * 60)
            logger.info("DRY-RUN MODE: Configuration validated successfully")
            logger.info("No trading will occur. Exiting.")
            logger.info("=" * 60)
            return

        logger.info("Initialization complete")

    async def _initialize_monitoring(self) -> None:
        """Initialize monitoring and observability system."""
        logger.info("Initializing monitoring system...")

        monitoring_config = self._config.get("monitoring", {})

        # Create monitoring system (includes metrics, alerts, anomaly detection)
        self._monitoring = MonitoringSystem(
            log_dir=monitoring_config.get("log_dir", "logs/agents"),
            metrics_retention_hours=monitoring_config.get("metrics_retention_hours", 24),
        )

        logger.info("Monitoring system initialized")

        # Initialize notification system (#C33, #R27)
        await self._initialize_notifications()

        # Initialize audit log backup manager
        await self._initialize_backup_manager()

    async def _initialize_notifications(self) -> None:
        """Initialize notification system (#C33, #R27)."""
        logger.info("Initializing notification system...")

        notifications_config = self._config.get("notifications", {})

        # Create notification channels
        channels = []

        # File channel (always enabled for audit trail)
        file_channel = FileNotificationChannel(
            filepath=notifications_config.get("alerts_file", "logs/alerts.jsonl")
        )
        channels.append(file_channel)

        # Webhook channel (if configured)
        webhook_url = notifications_config.get("webhook_url")
        if webhook_url:
            webhook_channel = WebhookNotificationChannel(
                webhook_url=webhook_url,
                timeout_seconds=notifications_config.get("webhook_timeout", 10.0),
            )
            channels.append(webhook_channel)
            logger.info(f"Webhook notifications enabled: {webhook_url[:50]}...")

        # Create notification manager
        self._notification_manager = NotificationManager(
            channels=channels,
            throttle_minutes=notifications_config.get("throttle_minutes", 5.0),
            escalation_delay_minutes=notifications_config.get("escalation_delay_minutes", 15.0),
        )

        # Create specialized notifiers
        self._compliance_notifier = ComplianceOfficerNotifier(
            notification_manager=self._notification_manager,
            compliance_officer_email=notifications_config.get("compliance_officer_email"),
            compliance_webhook=notifications_config.get("compliance_webhook"),
        )

        self._risk_notifier = RiskLimitBreachNotifier(
            notification_manager=self._notification_manager,
        )

        logger.info(f"Notification system initialized with {len(channels)} channels")

    async def _initialize_backup_manager(self) -> None:
        """Initialize audit log backup manager."""
        logger.info("Initializing audit log backup manager...")

        backup_config = self._config.get("backup", {})

        logging_config = self._config.get("logging", {})
        log_dir = Path(logging_config.get("audit_file", "logs/audit.jsonl")).parent

        self._backup_manager = AuditLogBackupManager(
            log_dir=str(log_dir),
            backup_dir=backup_config.get("backup_dir", "logs/backup"),
            retention_days=backup_config.get("retention_days", 90),
        )

        # Schedule daily backup if enabled
        if backup_config.get("enabled", True):
            backup_hour = backup_config.get("backup_hour", 2)  # Default 2 AM
            backup_minute = backup_config.get("backup_minute", 0)
            self._backup_manager.schedule_daily_backup(hour=backup_hour, minute=backup_minute)
            logger.info(f"Daily audit log backup scheduled at {backup_hour:02d}:{backup_minute:02d}")

        logger.info("Audit log backup manager initialized")

    async def _initialize_health_check(self) -> None:
        """Initialize health check server (#S5)."""
        logger.info("Initializing health check server...")

        health_config = self._config.get("health_check", {})

        # Create health check config
        config = HealthCheckConfig(
            host=health_config.get("host", "0.0.0.0"),
            port=health_config.get("port", 8080),
            max_event_queue_pct=health_config.get("max_event_queue_pct", 90.0),
            max_latency_ms=health_config.get("max_latency_ms", 1000.0),
            min_active_agents=health_config.get("min_active_agents", 3),
            broker_required=health_config.get("broker_required", False),
        )

        # Create health checker and server
        self._health_checker, self._health_server = create_health_check_server(
            get_status_fn=self.get_status,
            monitoring=self._monitoring,
            config=config,
        )

        # Start the health check server
        if health_config.get("enabled", True):
            self._health_server.start()
            logger.info("=" * 50)
            logger.info("MONITORING ENDPOINTS:")
            logger.info(f"  Health:    http://{config.host}:{config.port}/health")
            logger.info(f"  Readiness: http://{config.host}:{config.port}/ready")
            logger.info(f"  Liveness:  http://{config.host}:{config.port}/alive")
            logger.info("=" * 50)
        else:
            logger.info("Health check server disabled in config")

    async def _initialize_dashboard(self) -> None:
        """
        Initialize the real-time dashboard server.

        The dashboard server shares the EventBus with the trading system,
        enabling real-time streaming of events to WebSocket clients.
        """
        logger.info("Initializing dashboard server...")

        dashboard_config = self._config.get("dashboard", {})

        if not dashboard_config.get("enabled", True):
            logger.info("Dashboard server disabled in config")
            return

        # Use a different port than health check to avoid conflicts
        # Health check uses 8080, dashboard uses 8081 by default
        dashboard_port = dashboard_config.get("port", 8081)
        dashboard_host = dashboard_config.get("host", "0.0.0.0")

        # Create dashboard server with shared EventBus
        self._dashboard_server = create_dashboard_server(
            event_bus=self._event_bus,
            host=dashboard_host,
            port=dashboard_port,
        )

        logger.info("=" * 50)
        logger.info("DASHBOARD SERVER:")
        logger.info(f"  Dashboard V1:  http://{dashboard_host}:{dashboard_port}/")
        logger.info(f"  WebSocket V1:  ws://{dashboard_host}:{dashboard_port}/ws")
        logger.info(f"  API V1:        http://{dashboard_host}:{dashboard_port}/api/status")

        # Initialize Dashboard V2 (Bloomberg Terminal)
        v2_config = self._config.get("dashboard_v2", {})
        if v2_config.get("enabled", True):
            v2_port = v2_config.get("port", 8082)
            v2_host = v2_config.get("host", dashboard_host)
            self._dashboard_v2_server = create_dashboard_v2_server(
                event_bus=self._event_bus,
                host=v2_host,
                port=v2_port,
            )
            logger.info(f"  Dashboard V2:  http://{v2_host}:{v2_port}/")
            logger.info(f"  WebSocket V2:  ws://{v2_host}:{v2_port}/ws/v2")

        logger.info("=" * 50)

    async def _start_dashboard(self) -> asyncio.Task | None:
        """Start the dashboard server in background."""
        if not self._dashboard_server:
            return None

        try:
            import uvicorn

            dashboard_config = self._config.get("dashboard", {})
            dashboard_port = dashboard_config.get("port", 8081)
            dashboard_host = dashboard_config.get("host", "0.0.0.0")

            # Create uvicorn config for async server
            config = uvicorn.Config(
                self._dashboard_server.app,
                host=dashboard_host,
                port=dashboard_port,
                log_level="warning",  # Reduce noise in logs
                access_log=False,
            )
            server = uvicorn.Server(config)

            # Start server in background task
            task = asyncio.create_task(server.serve())
            logger.info(f"Dashboard server started on http://{dashboard_host}:{dashboard_port}")
            return task

        except ImportError:
            logger.warning(
                "uvicorn not installed - dashboard server will not start. "
                "Install with: pip install uvicorn"
            )
            return None
        except Exception as e:
            logger.exception(f"Failed to start dashboard server: {e}")
            return None

    async def _start_dashboard_v2(self) -> asyncio.Task | None:
        """Start the V2 Bloomberg dashboard server in background."""
        if not self._dashboard_v2_server:
            return None

        try:
            import uvicorn

            v2_config = self._config.get("dashboard_v2", {})
            v2_port = v2_config.get("port", 8082)
            v2_host = v2_config.get("host", "0.0.0.0")

            config = uvicorn.Config(
                self._dashboard_v2_server.app,
                host=v2_host,
                port=v2_port,
                log_level="warning",
                access_log=False,
            )
            server = uvicorn.Server(config)

            task = asyncio.create_task(server.serve())
            logger.info(f"Dashboard V2 Bloomberg started on http://{v2_host}:{v2_port}")
            return task

        except ImportError:
            return None
        except Exception as e:
            logger.exception(f"Failed to start dashboard V2: {e}")
            return None

    async def _start_broker_sync(self) -> None:
        """
        Start broker metrics sync for real-time P&L in dashboard.

        This bridges IB portfolio data to the dashboard, ensuring actual
        account values, positions, and P&L are displayed correctly.
        """
        if not self._broker or not self._dashboard_server:
            logger.debug("Broker sync skipped: broker or dashboard not available")
            return

        try:
            dashboard_config = self._config.get("dashboard", {})
            sync_config = dashboard_config.get("broker_sync", {})

            # Create broker sync
            self._broker_sync = create_broker_sync(
                broker=self._broker,
                dashboard_server=self._dashboard_server,
                performance_metrics=None,  # Will integrate later if needed
                position_view=None,  # Will integrate later if needed
                portfolio_interval=sync_config.get("portfolio_interval_seconds", 2.0),
                performance_interval=sync_config.get("performance_interval_seconds", 5.0),
            )

            # Start sync
            await self._broker_sync.start()

            logger.info(
                "Broker metrics sync started - dashboard will show real IB P&L and positions"
            )

        except Exception as e:
            logger.exception(f"Failed to start broker sync: {e}")
            self._broker_sync = None

    async def _initialize_broker(self) -> None:
        """Initialize broker connection."""
        broker_config = self._config.get("broker", {})

        self._broker = IBBroker(
            BrokerConfig(
                host=broker_config.get("host", "127.0.0.1"),
                port=broker_config.get("port", 7497),
                client_id=broker_config.get("client_id", 1),
                timeout_seconds=broker_config.get("timeout_seconds", 30),
                readonly=broker_config.get("readonly", False),
            )
        )

        # Connect to broker
        logger.info(f"Connecting to IB at {broker_config.get('host', '127.0.0.1')}:{broker_config.get('port', 7497)}...")
        connected = await self._broker.connect()

        if connected:
            logger.info(f"Connected to IB - Account: {self._broker.account_id}")

            # Request delayed data if no market data subscription
            if broker_config.get("use_delayed_data", True):
                await self._broker.request_market_data_type(3)  # Delayed data
                logger.info("Using delayed market data (free)")
        else:
            logger.warning("Failed to connect to IB - running in SIMULATION mode")
            logger.warning("Start TWS/IB Gateway and enable API to use real data")

        # Register fill callback for monitoring
        self._broker.on_fill(self._on_fill)
        self._broker.on_market_data(self._on_market_data)

    async def _initialize_market_data(self) -> None:
        """Initialize market data manager."""
        universe = self._config.get("universe", {})
        symbols = []

        # Equities
        for equity in universe.get("equities", []):
            symbols.append(SymbolConfig(
                symbol=equity["symbol"],
                exchange=equity.get("exchange", "SMART"),
                currency=equity.get("currency", "USD"),
                sec_type="STK",
            ))

        # ETFs
        for etf in universe.get("etfs", []):
            symbols.append(SymbolConfig(
                symbol=etf["symbol"],
                exchange=etf.get("exchange", "SMART"),
                currency=etf.get("currency", "USD"),
                sec_type="STK",
            ))

        # Futures
        for future in universe.get("futures", []):
            symbols.append(SymbolConfig(
                symbol=future["symbol"],
                exchange=future.get("exchange", "CME"),
                currency=future.get("currency", "USD"),
                sec_type="FUT",
            ))

        # Forex
        for forex in universe.get("forex", []):
            symbols.append(SymbolConfig(
                symbol=forex["symbol"],
                exchange=forex.get("exchange", "IDEALPRO"),
                currency=forex.get("currency", "USD"),
                sec_type="CASH",
            ))

        self._market_data = MarketDataManager(
            broker=self._broker,
            event_bus=self._event_bus,
            symbols=symbols,
        )

        logger.info(f"Market data manager initialized with {len(symbols)} symbols")
        logger.info(f"  Equities: {len(universe.get('equities', []))}")
        logger.info(f"  ETFs: {len(universe.get('etfs', []))}")
        logger.info(f"  Futures: {len(universe.get('futures', []))}")
        logger.info(f"  Forex: {len(universe.get('forex', []))}")

    async def _run_historical_warmup(self) -> None:
        """
        Run historical data warmup to immediately prepare agents.

        Fetches the last 100 bars for each symbol from IB so agents
        can generate signals immediately without waiting for live data.
        """
        warmup_config = self._config.get("warmup", {})

        if not warmup_config.get("enabled", True):
            logger.info("Historical warmup disabled in config")
            return

        # Get all symbols from universe
        universe = self._config.get("universe", {})
        symbols = []

        for equity in universe.get("equities", []):
            symbols.append(equity["symbol"])
        for etf in universe.get("etfs", []):
            symbols.append(etf["symbol"])
        for future in universe.get("futures", []):
            symbols.append(future["symbol"])
        # Skip forex for warmup (different contract type)

        if not symbols:
            logger.info("No symbols to warm up")
            return

        # Create warmup instance
        self._historical_warmup = create_historical_warmup(
            broker=self._broker,
            event_bus=self._event_bus,
            config={
                "bars_to_fetch": warmup_config.get("bars_to_fetch", 100),
                "bar_size": warmup_config.get("bar_size", "1 min"),
                "max_concurrent": warmup_config.get("max_concurrent", 5),
                "timeout_seconds": warmup_config.get("timeout_seconds", 30.0),
            },
        )

        # Run warmup
        logger.info(f"Running historical warmup for {len(symbols)} symbols...")
        results = await self._historical_warmup.warmup_symbols(symbols)

        successful = sum(1 for v in results.values() if v > 0)
        total_bars = sum(results.values())

        if successful > 0:
            logger.info(
                f"Warmup complete: {successful}/{len(symbols)} symbols, "
                f"{total_bars} bars - Agents ready for immediate signals"
            )
        else:
            logger.warning(
                "Warmup failed - agents will need to wait for live data to accumulate"
            )

    async def _init_economic_calendar(self, event_driven_agent) -> None:
        """Initialize economic calendar and register events with EventDrivenAgent."""
        if not self._economic_calendar:
            return

        try:
            await self._economic_calendar.initialize()
            count = self._economic_calendar.register_with_agent(event_driven_agent)

            # Log next upcoming events
            next_event = self._economic_calendar.get_next_event()
            if next_event:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                hours_until = (next_event.timestamp - now).total_seconds() / 3600
                logger.info(
                    f"Economic Calendar: {count} events loaded. "
                    f"Next: {next_event.event_type.value} in {hours_until:.1f}h"
                )
            else:
                logger.info(f"Economic Calendar: {count} events loaded")

            # Start auto-update for fresh API data
            await self._economic_calendar.start_auto_update()

        except Exception as e:
            logger.warning(f"Failed to initialize economic calendar: {e}")

    async def _initialize_infrastructure(self) -> None:
        """Initialize infrastructure components (contract specs, risk calculators, etc.)."""
        logger.info("Initializing infrastructure components...")

        # Contract Specifications Manager
        contract_specs_config = self._config.get("contract_specs", {})
        self._contract_specs = ContractSpecsManager(config=contract_specs_config)
        logger.info(f"Contract specs manager initialized ({self._contract_specs.get_all_symbols().__len__()} contracts)")

        # Correlation Manager
        correlation_config = self._config.get("correlation", {})
        self._correlation_manager = CorrelationManager(config=correlation_config)
        logger.info("Correlation manager initialized")

        # Futures Roll Manager
        futures_roll_config = self._config.get("futures_roll", {})
        if futures_roll_config.get("enabled", True):
            self._futures_roll_manager = FuturesRollManager(config=futures_roll_config)
            logger.info("Futures roll manager initialized")

        # VaR Calculator
        var_config = self._config.get("var", {})
        self._var_calculator = VaRCalculator(config=var_config)
        logger.info(f"VaR calculator initialized (method: {var_config.get('method', 'all')})")

        # Stress Tester
        stress_config = self._config.get("stress_testing", {})
        if stress_config.get("enabled", True):
            self._stress_tester = StressTester(config=stress_config)
            logger.info("Stress tester initialized")

        # Position Sizer (Kelly criterion)
        position_sizing_config = self._config.get("position_sizing", {})
        self._position_sizer = PositionSizer(config=position_sizing_config)
        logger.info(f"Position sizer initialized (method: {position_sizing_config.get('method', 'kelly')})")

        # Performance Attribution
        attribution_config = self._config.get("attribution", {})
        self._attribution = PerformanceAttribution(config=attribution_config)
        logger.info("Performance attribution initialized")

        # Best Execution Analyzer
        best_execution_config = self._config.get("best_execution", {})
        self._best_execution = BestExecutionAnalyzer(config=best_execution_config)
        logger.info(f"Best execution analyzer initialized (benchmark: {best_execution_config.get('benchmark', 'vwap')})")

        # Stop-Loss Manager for automatic position protection
        exit_rules_config = self._config.get("agents", {}).get("exit_rules", {})
        stop_loss_config = StopLossConfig(
            use_atr_stops=exit_rules_config.get("use_atr_stops", True),
            atr_multiplier=exit_rules_config.get("stop_loss_atr_multiplier", 2.5),
            fixed_stop_loss_pct=exit_rules_config.get("default_stop_loss_pct", 5.0),
            trailing_stop_enabled=exit_rules_config.get("trailing_enabled", True),
            trailing_activation_pct=exit_rules_config.get("trailing_activation_pct", 2.0),
            trailing_distance_pct=exit_rules_config.get("trailing_distance_pct", 3.0),
            take_profit_enabled=exit_rules_config.get("enabled", True),
            take_profit_pct=exit_rules_config.get("default_take_profit_pct", 10.0),
            trailing_take_profit=exit_rules_config.get("trailing_take_profit", True),
            max_holding_hours=exit_rules_config.get("time_exit_hours", 0),
            drawdown_close_threshold_pct=self._config.get("risk", {}).get("drawdown", {}).get("close_threshold_pct", 15.0),
            close_worst_n_positions=self._config.get("risk", {}).get("drawdown", {}).get("close_worst_n", 2),
        )
        self._stop_loss_manager = StopLossManager(
            config=stop_loss_config,
            event_bus=self._event_bus,
            audit_logger=self._audit_logger,
            broker=self._broker,
        )
        logger.info(
            f"Stop-loss manager initialized: ATR={stop_loss_config.use_atr_stops} "
            f"(mult={stop_loss_config.atr_multiplier}), trailing={stop_loss_config.trailing_stop_enabled}"
        )

        logger.info("Infrastructure components initialized")

    def _wire_components(self) -> None:
        """Wire infrastructure components to agents."""
        logger.info("Wiring components together...")

        # Wire VaR calculator to Risk Agent
        if self._risk_agent and self._var_calculator:
            self._risk_agent.set_var_calculator(self._var_calculator)
            logger.info("  VaRCalculator -> RiskAgent")

        # Wire Stress Tester to Risk Agent
        if self._risk_agent and self._stress_tester:
            self._risk_agent.set_stress_tester(self._stress_tester)
            logger.info("  StressTester -> RiskAgent")

        # Wire Correlation Manager to Risk Agent
        if self._risk_agent and self._correlation_manager:
            self._risk_agent.set_correlation_manager(self._correlation_manager)
            logger.info("  CorrelationManager -> RiskAgent")

        # Wire Position Sizer to CIO Agent
        if self._cio_agent and self._position_sizer:
            self._cio_agent.set_position_sizer(self._position_sizer)
            logger.info("  PositionSizer -> CIOAgent")

        # Wire Attribution to CIO Agent
        if self._cio_agent and self._attribution:
            self._cio_agent.set_attribution(self._attribution)
            logger.info("  PerformanceAttribution -> CIOAgent")

        # Wire Correlation Manager to CIO Agent
        if self._cio_agent and self._correlation_manager:
            self._cio_agent.set_correlation_manager(self._correlation_manager)
            logger.info("  CorrelationManager -> CIOAgent")

        # Wire Broker to CIO Agent for leverage-aware deleveraging
        if self._cio_agent and self._broker:
            self._cio_agent.set_broker(self._broker)
            logger.info("  Broker -> CIOAgent (leverage monitoring)")

        # Wire Best Execution to Execution Agent
        if self._execution_agent and self._best_execution:
            self._execution_agent.set_best_execution_analyzer(self._best_execution)
            logger.info("  BestExecutionAnalyzer -> ExecutionAgent")

        # Wire Stop-Loss Manager to Execution Agent
        if self._execution_agent and self._stop_loss_manager:
            self._execution_agent.set_stop_loss_manager(self._stop_loss_manager)
            logger.info("  StopLossManager -> ExecutionAgent")

        # Wire Contract Specs to Broker
        if self._broker and self._contract_specs:
            self._broker.set_contract_specs_manager(self._contract_specs)
            logger.info("  ContractSpecsManager -> Broker")

        # Wire Risk Notifier to Risk Agent (#R27)
        if self._risk_agent and self._risk_notifier:
            self._risk_agent.set_risk_notifier(self._risk_notifier)
            logger.info("  RiskLimitBreachNotifier -> RiskAgent")

        # Wire Compliance Notifier to Compliance Agent (#C33)
        if self._compliance_agent and self._compliance_notifier:
            self._compliance_agent.set_compliance_notifier(self._compliance_notifier)
            logger.info("  ComplianceOfficerNotifier -> ComplianceAgent")

        # Wire Compliance Notifier to Surveillance Agent
        if self._surveillance_agent and self._compliance_notifier:
            self._surveillance_agent.set_compliance_notifier(self._compliance_notifier)
            logger.info("  ComplianceOfficerNotifier -> SurveillanceAgent")

        # Wire Dashboard Server to orchestrator for real-time data access
        if self._dashboard_server:
            self._dashboard_server.set_orchestrator(self)
            logger.info("  Orchestrator -> DashboardServer V1 (real-time data)")

        # Wire Dashboard V2 Server to orchestrator
        if self._dashboard_v2_server:
            self._dashboard_v2_server.set_orchestrator(self)
            logger.info("  Orchestrator -> DashboardServer V2 Bloomberg (real-time data)")

        logger.info("Component wiring complete")

    async def _initialize_agents(self) -> None:
        """Initialize all trading agents."""
        agents_config = self._config.get("agents", {})
        risk_config = self._config.get("risk", {})
        compliance_config = self._config.get("compliance", {})

        # ========== SIGNAL AGENTS (parallel execution) ==========
        logger.info("Initializing signal agents...")

        if agents_config.get("macro", {}).get("enabled", True):
            macro_agent = MacroAgent(
                config=AgentConfig(
                    name="MacroAgent",
                    enabled=True,
                    parameters=agents_config.get("macro", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            self._signal_agents.append(macro_agent)
            self._event_bus.register_signal_agent("MacroAgent")

        if agents_config.get("stat_arb", {}).get("enabled", True):
            stat_arb_agent = StatArbAgent(
                config=AgentConfig(
                    name="StatArbAgent",
                    enabled=True,
                    parameters=agents_config.get("stat_arb", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            self._signal_agents.append(stat_arb_agent)
            self._event_bus.register_signal_agent("StatArbAgent")

        if agents_config.get("momentum", {}).get("enabled", True):
            momentum_agent = MomentumAgent(
                config=AgentConfig(
                    name="MomentumAgent",
                    enabled=True,
                    parameters=agents_config.get("momentum", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            self._signal_agents.append(momentum_agent)
            self._event_bus.register_signal_agent("MomentumAgent")

        if agents_config.get("market_making", {}).get("enabled", True):
            mm_agent = MarketMakingAgent(
                config=AgentConfig(
                    name="MarketMakingAgent",
                    enabled=True,
                    parameters=agents_config.get("market_making", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            self._signal_agents.append(mm_agent)
            self._event_bus.register_signal_agent("MarketMakingAgent")

        # Sentiment Agent (LLM-powered news analysis)
        if agents_config.get("sentiment", {}).get("enabled", True):
            from agents.sentiment_agent import SentimentAgent
            from core.llm_client import LLMClient

            sentiment_config = agents_config.get("sentiment", {})
            llm_client = LLMClient(config=sentiment_config)

            sentiment_agent = SentimentAgent(
                config=AgentConfig(
                    name="SentimentAgent",
                    enabled=True,
                    parameters=sentiment_config,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
                llm_client=llm_client,
            )
            self._signal_agents.append(sentiment_agent)
            self._event_bus.register_signal_agent("SentimentAgent")

        # Chart Analysis Agent (Claude Vision pattern recognition)
        if agents_config.get("chart_analysis", {}).get("enabled", True):
            from agents.chart_analysis_agent import ChartAnalysisAgent

            chart_config = agents_config.get("chart_analysis", {})
            chart_agent = ChartAnalysisAgent(
                config=AgentConfig(
                    name="ChartAnalysisAgent",
                    enabled=True,
                    parameters=chart_config,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            self._signal_agents.append(chart_agent)
            self._event_bus.register_signal_agent("ChartAnalysisAgent")

        # Forecasting Agent (LLM-powered price prediction)
        if agents_config.get("forecasting", {}).get("enabled", True):
            from agents.forecasting_agent import ForecastingAgent
            from core.llm_client import LLMClient

            forecasting_config = agents_config.get("forecasting", {})
            forecasting_llm_client = LLMClient(config=forecasting_config.get("llm", {}))

            forecasting_agent = ForecastingAgent(
                config=AgentConfig(
                    name="ForecastingAgent",
                    enabled=True,
                    parameters=forecasting_config,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
                llm_client=forecasting_llm_client,
            )
            self._signal_agents.append(forecasting_agent)
            self._event_bus.register_signal_agent("ForecastingAgent")

        # ========== NEW PHASE 6 AGENTS ==========
        # Session Agent (Opening Range Breakout)
        if agents_config.get("session", {}).get("enabled", True):
            from agents.session_agent import SessionAgent
            session_agent = SessionAgent(
                config=AgentConfig(
                    name="SessionAgent",
                    enabled=True,
                    parameters=agents_config.get("session", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            self._signal_agents.append(session_agent)
            self._event_bus.register_signal_agent("SessionAgent")

        # Index Spread Agent (MES/MNQ spread trading)
        if agents_config.get("index_spread", {}).get("enabled", True):
            from agents.index_spread_agent import IndexSpreadAgent
            index_spread_agent = IndexSpreadAgent(
                config=AgentConfig(
                    name="IndexSpreadAgent",
                    enabled=True,
                    parameters=agents_config.get("index_spread", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            self._signal_agents.append(index_spread_agent)
            self._event_bus.register_signal_agent("IndexSpreadAgent")

        # TTM Squeeze Agent (BB inside KC volatility breakout)
        if agents_config.get("ttm_squeeze", {}).get("enabled", True):
            from agents.ttm_squeeze_agent import TTMSqueezeAgent
            ttm_squeeze_agent = TTMSqueezeAgent(
                config=AgentConfig(
                    name="TTMSqueezeAgent",
                    enabled=True,
                    parameters=agents_config.get("ttm_squeeze", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            self._signal_agents.append(ttm_squeeze_agent)
            self._event_bus.register_signal_agent("TTMSqueezeAgent")

        # Event-Driven Agent (FOMC, NFP, CPI)
        if agents_config.get("event_driven", {}).get("enabled", True):
            from agents.event_driven_agent import EventDrivenAgent
            event_driven_agent = EventDrivenAgent(
                config=AgentConfig(
                    name="EventDrivenAgent",
                    enabled=True,
                    parameters=agents_config.get("event_driven", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            self._signal_agents.append(event_driven_agent)
            self._event_bus.register_signal_agent("EventDrivenAgent")

            # Initialize Economic Calendar and register events
            calendar_config = agents_config.get("event_driven", {}).get("calendar", {})
            self._economic_calendar = create_economic_calendar(calendar_config)
            # FIX-38: Await calendar init (was fire-and-forget causing race condition)
            await self._init_economic_calendar(event_driven_agent)

        # Mean Reversion Agent (RSI extremes, BB touches, z-score)
        if agents_config.get("mean_reversion", {}).get("enabled", True):
            from agents.mean_reversion_agent import MeanReversionAgent
            mean_reversion_agent = MeanReversionAgent(
                config=AgentConfig(
                    name="MeanReversionAgent",
                    enabled=True,
                    parameters=agents_config.get("mean_reversion", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            self._signal_agents.append(mean_reversion_agent)
            self._event_bus.register_signal_agent("MeanReversionAgent")

        # MACD-v Agent (Volatility-Normalized MACD - Charles H. Dow Award 2022)
        if agents_config.get("macdv", {}).get("enabled", True):
            from agents.macdv_agent import MACDvAgent
            macdv_agent = MACDvAgent(
                config=AgentConfig(
                    name="MACDvAgent",
                    enabled=True,
                    parameters=agents_config.get("macdv", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            self._signal_agents.append(macdv_agent)
            self._event_bus.register_signal_agent("MACDvAgent")

        logger.info(f"Initialized {len(self._signal_agents)} signal agents")

        # ========== CIO AGENT (THE decision maker) ==========
        logger.info("Initializing CIO agent...")
        cio_params = dict(agents_config.get("cio", {}))
        # Pass universe info for market hours filtering
        universe = self._config.get("universe", {})
        cio_params["_us_equity_symbols"] = [
            s["symbol"] for s in universe.get("equities", [])
        ]
        cio_params["_us_etf_symbols"] = [
            s["symbol"] for s in universe.get("etfs", [])
        ]
        cio_params["_forex_symbols"] = [
            s["symbol"] for s in universe.get("forex", [])
        ]
        self._cio_agent = CIOAgent(
            config=AgentConfig(
                name="CIOAgent",
                enabled=True,
                parameters=cio_params,
            ),
            event_bus=self._event_bus,
            audit_logger=self._audit_logger,
        )

        # ========== RISK AGENT (separate from compliance) ==========
        logger.info("Initializing Risk agent...")
        drawdown_config = risk_config.get("drawdown", {})
        max_drawdown_pct = risk_config.get("max_drawdown_pct", 10.0)
        risk_params = {
            "limits": {
                "max_position_size_pct": risk_config.get("max_position_size_pct", 5.0),
                "max_sector_exposure_pct": risk_config.get("max_sector_exposure_pct", 20.0),
                "max_leverage": risk_config.get("max_leverage", 2.0),
                "max_portfolio_var_pct": risk_config.get("max_portfolio_var_pct", 2.0),
                "max_daily_loss_pct": risk_config.get("max_daily_loss_pct", 3.0),
                "max_drawdown_pct": max_drawdown_pct,
            },
            "rate_limits": {
                "max_orders_per_minute": risk_config.get("max_orders_per_minute", 10),
                "min_order_interval_ms": risk_config.get("min_order_interval_ms", 100),
            },
            "sector_map": self._config.get("sector_map", {}),
            # Tiered drawdown thresholds
            "drawdown": {
                "warning_pct": drawdown_config.get("warning_pct", 5.0),
                "reduce_pct": drawdown_config.get("reduce_pct", 7.5),
                "halt_pct": drawdown_config.get("halt_pct", max_drawdown_pct),
                "position_reduction_factor": drawdown_config.get("position_reduction_factor", 0.5),
            },
        }

        self._risk_agent = RiskAgent(
            config=AgentConfig(
                name="RiskAgent",
                enabled=True,
                parameters=risk_params,
            ),
            event_bus=self._event_bus,
            audit_logger=self._audit_logger,
            broker=self._broker,
        )

        # ========== COMPLIANCE AGENT (EU/AMF) ==========
        logger.info("Initializing Compliance agent...")
        compliance_params = {
            "jurisdiction": compliance_config.get("jurisdiction", "EU"),
            "restricted_instruments": compliance_config.get("banned_instruments", []),
            "allowed_asset_classes": compliance_config.get("allowed_asset_classes", ["equity", "etf"]),
            "entity_lei": compliance_config.get("firm_lei", ""),
        }

        self._compliance_agent = ComplianceAgent(
            config=AgentConfig(
                name="ComplianceAgent",
                enabled=True,
                parameters=compliance_params,
            ),
            event_bus=self._event_bus,
            audit_logger=self._audit_logger,
        )

        # ========== EXECUTION AGENT (ONLY agent that sends orders) ==========
        logger.info("Initializing Execution agent...")
        self._execution_agent = ExecutionAgentImpl(
            config=AgentConfig(
                name="ExecutionAgent",
                enabled=True,
                parameters=agents_config.get("execution", {}),
            ),
            event_bus=self._event_bus,
            audit_logger=self._audit_logger,
            broker=self._broker,
        )

        # ========== SURVEILLANCE AGENT (MAR 2014/596/EU) ==========
        surveillance_config = self._config.get("surveillance", {})
        if agents_config.get("surveillance", {}).get("enabled", True):
            logger.info("Initializing Surveillance agent...")
            surveillance_params = {
                "wash_trading_detection": surveillance_config.get("wash_trading_detection", True),
                "spoofing_detection": surveillance_config.get("spoofing_detection", True),
                "quote_stuffing_detection": surveillance_config.get("quote_stuffing_detection", True),
                "layering_detection": surveillance_config.get("layering_detection", True),
                "wash_trading_window_seconds": surveillance_config.get("wash_trading_window_seconds", 60),
                "spoofing_cancel_threshold": surveillance_config.get("spoofing_cancel_threshold", 0.8),
                "quote_stuffing_rate_per_second": surveillance_config.get("quote_stuffing_rate_per_second", 10),
            }
            self._surveillance_agent = SurveillanceAgent(
                config=AgentConfig(
                    name="SurveillanceAgent",
                    enabled=True,
                    parameters=surveillance_params,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )

        # ========== TRANSACTION REPORTING AGENT (ESMA RTS 22/23) ==========
        transaction_reporting_config = self._config.get("transaction_reporting", {})
        if agents_config.get("transaction_reporting", {}).get("enabled", True):
            logger.info("Initializing Transaction Reporting agent...")
            reporting_params = {
                "enabled": transaction_reporting_config.get("enabled", True),
                "reporting_deadline_minutes": transaction_reporting_config.get("reporting_deadline_minutes", 15),
                "firm_lei": transaction_reporting_config.get("firm_lei", ""),
                "firm_country": transaction_reporting_config.get("firm_country", "FR"),
                "default_venue": transaction_reporting_config.get("default_venue", "XPAR"),
            }
            self._transaction_reporting_agent = TransactionReportingAgent(
                config=AgentConfig(
                    name="TransactionReportingAgent",
                    enabled=True,
                    parameters=reporting_params,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
                reporting_config=reporting_params,
            )

        # ========== RECONCILIATION AGENT (Position Safety) ==========
        reconciliation_config = self._config.get("reconciliation", {})
        if reconciliation_config.get("enabled", True):
            logger.info("Initializing Reconciliation agent...")
            recon_params = {
                "reconcile_interval_seconds": reconciliation_config.get("interval_seconds", 60.0),
                "auto_correct_enabled": reconciliation_config.get("auto_correct", False),
                "discrepancy_threshold_pct": reconciliation_config.get("discrepancy_threshold_pct", 1.0),
            }
            self._reconciliation_agent = ReconciliationAgent(
                config=AgentConfig(
                    name="ReconciliationAgent",
                    enabled=True,
                    parameters=recon_params,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
                broker=self._broker,
                get_theoretical_positions_fn=(
                    self._cio_agent.get_tracked_positions if self._cio_agent else None
                ),
            )

        logger.info("All agents initialized")
        logger.info("  Signal agents: " + ", ".join(a.name for a in self._signal_agents))
        logger.info("  Decision: CIOAgent")
        logger.info("  Validation: RiskAgent -> ComplianceAgent")
        logger.info("  Execution: ExecutionAgent")
        if self._surveillance_agent:
            logger.info("  Surveillance: SurveillanceAgent (MAR 2014/596/EU)")
        if self._transaction_reporting_agent:
            logger.info("  Reporting: TransactionReportingAgent (ESMA RTS 22/23)")
        if self._reconciliation_agent:
            logger.info("  Reconciliation: ReconciliationAgent (Position Safety)")

    async def _initialize_agents_with_factory(self) -> None:
        """
        Initialize agents using AgentFactory (SOLID refactoring alternative).

        This method demonstrates the use of AgentFactory for cleaner SRP.
        Can be used instead of _initialize_agents() for improved testability.
        """
        factory_config = AgentFactoryConfig(
            agents_config=self._config.get("agents", {}),
            risk_config=self._config.get("risk", {}),
            compliance_config=self._config.get("compliance", {}),
            surveillance_config=self._config.get("surveillance", {}),
            transaction_reporting_config=self._config.get("transaction_reporting", {}),
            sector_map=self._config.get("sector_map", {}),
        )

        factory = AgentFactory(
            event_bus=self._event_bus,
            audit_logger=self._audit_logger,
            broker=self._broker,
            config=factory_config,
        )

        agents = factory.create_all_agents()

        # Assign to orchestrator
        self._signal_agents = agents.signal_agents
        self._cio_agent = agents.cio_agent
        self._risk_agent = agents.risk_agent
        self._compliance_agent = agents.compliance_agent
        self._execution_agent = agents.execution_agent
        self._surveillance_agent = agents.surveillance_agent
        self._transaction_reporting_agent = agents.transaction_reporting_agent

        logger.info("Agents initialized via AgentFactory")

    async def start(self) -> None:
        """Start the trading system."""
        logger.info("=" * 60)
        logger.info("AI TRADING FIRM - STARTING")
        logger.info("=" * 60)

        self._running = True

        # Subscribe to events for monitoring and shutdown
        self._event_bus.subscribe(EventType.RISK_ALERT, self._handle_risk_alert)

        # Start event bus
        event_bus_task = asyncio.create_task(self._event_bus.start())

        # Start all agents
        agent_tasks = []

        # Signal agents start in parallel
        for agent in self._signal_agents:
            await agent.start()

        # Sequential agents in order
        await self._cio_agent.start()
        await self._risk_agent.start()
        await self._compliance_agent.start()
        await self._execution_agent.start()

        # Compliance/surveillance agents
        if self._surveillance_agent:
            await self._surveillance_agent.start()
        if self._transaction_reporting_agent:
            await self._transaction_reporting_agent.start()
        if self._reconciliation_agent:
            await self._reconciliation_agent.start()
            await self._reconciliation_agent.start_reconciliation_loop()

        # Start stop-loss manager
        if self._stop_loss_manager:
            await self._stop_loss_manager.start()

        # Historical warmup - fetch past bars to immediately warm up agents
        if self._broker.is_connected:
            await self._run_historical_warmup()

        # Start market data
        await self._market_data.start()

        # If broker not connected, generate simulated data
        if not self._broker.is_connected:
            logger.info("Starting simulated market data (IB not connected)")
            sim_task = asyncio.create_task(self._market_data.generate_simulated_data())
            agent_tasks.append(sim_task)

        agent_tasks.append(event_bus_task)

        # Start monitoring
        if self._monitoring:
            monitoring_task = asyncio.create_task(self._run_monitoring())
            agent_tasks.append(monitoring_task)

        # Start dashboard server (after broker connection for event streaming)
        dashboard_task = await self._start_dashboard()
        if dashboard_task:
            agent_tasks.append(dashboard_task)

        # Start Dashboard V2 Bloomberg Terminal
        dashboard_v2_task = await self._start_dashboard_v2()
        if dashboard_v2_task:
            agent_tasks.append(dashboard_v2_task)

        # Start broker metrics sync for real P&L in dashboard
        if self._broker and self._dashboard_server:
            await self._start_broker_sync()

        self._audit_logger.log_system_event(
            "started",
            {
                "signal_agents": [a.name for a in self._signal_agents],
                "broker_connected": self._broker.is_connected,
                "account_id": self._broker.account_id if self._broker.is_connected else None,
                "dashboard_enabled": self._dashboard_server is not None,
            },
        )

        dashboard_config = self._config.get("dashboard", {})
        dashboard_port = dashboard_config.get("port", 8081)

        logger.info("=" * 60)
        logger.info("TRADING SYSTEM STARTED")
        logger.info(f"  Broker: {'CONNECTED' if self._broker.is_connected else 'SIMULATED'}")
        logger.info(f"  Mode: {self._config.get('firm', {}).get('mode', 'paper').upper()}")
        if self._dashboard_server:
            logger.info(f"  Dashboard V1: http://localhost:{dashboard_port}/")
        if self._dashboard_v2_server:
            v2_port = self._config.get("dashboard_v2", {}).get("port", 8082)
            logger.info(f"  Dashboard V2: http://localhost:{v2_port}/  (Bloomberg)")
        logger.info("  Waiting for market data events...")
        logger.info("=" * 60)

        # Run until shutdown
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass

        # Cleanup
        await self.stop()

    async def _run_monitoring(self) -> None:
        """Run periodic monitoring tasks."""
        # MON-007: Track monitoring loop health
        _last_heartbeat: datetime = datetime.now(timezone.utc)

        while self._running:
            try:
                # MON-007: Emit heartbeat metric at each iteration
                _last_heartbeat = datetime.now(timezone.utc)
                self._monitoring.metrics.record_metric(
                    "monitoring_loop_heartbeat",
                    _last_heartbeat.timestamp(),
                )

                # Collect portfolio metrics
                if self._broker.is_connected:
                    portfolio = await self._broker.get_portfolio_state()

                    self._monitoring.metrics.record_metric(
                        "portfolio.net_liquidation",
                        portfolio.net_liquidation,
                    )
                    self._monitoring.metrics.record_metric(
                        "portfolio.daily_pnl",
                        portfolio.daily_pnl,
                    )

                    # Check for alerts
                    if portfolio.net_liquidation > 0:
                        daily_pnl_pct = portfolio.daily_pnl / portfolio.net_liquidation * 100
                        self._monitoring.alerts.check_metric("daily_pnl_pct", daily_pnl_pct)

                # Collect agent metrics
                if self._risk_agent:
                    risk_status = self._risk_agent.get_status()
                    self._monitoring.metrics.record_metric(
                        "risk.var_95",
                        risk_status.get("risk_state", {}).get("var_95", 0)
                    )
                    self._monitoring.metrics.record_metric(
                        "risk.leverage",
                        risk_status.get("risk_state", {}).get("leverage", 0)
                    )

                # RT-P1-5: Add jitter to prevent thundering herd
                # Base 30s interval with +/- 5s random jitter
                sleep_duration = 30 + random.uniform(-5, 5)
                await asyncio.sleep(sleep_duration)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                # RT-P1-5: Also add jitter to error recovery sleep
                await asyncio.sleep(5 + random.uniform(-1, 1))

    async def stop(self) -> None:
        """Stop the trading system gracefully with timeout protection."""
        # Prevent double shutdown
        if self._shutdown_in_progress:
            logger.warning("Shutdown already in progress, ignoring duplicate request")
            return

        self._shutdown_in_progress = True

        logger.info("=" * 60)
        logger.info("AI TRADING FIRM - GRACEFUL SHUTDOWN")
        logger.info("=" * 60)

        self._running = False
        shutdown_start = datetime.now(timezone.utc)

        # Helper for shutdown with timeout
        async def _stop_with_timeout(coro, name: str, timeout: float = 5.0) -> None:
            """Stop a component with timeout protection."""
            try:
                await asyncio.wait_for(coro, timeout=timeout)
                logger.info(f"  [OK] {name} stopped")
            except asyncio.TimeoutError:
                logger.warning(f"  [TIMEOUT] {name} did not stop within {timeout}s")
            except Exception as e:
                logger.error(f"  [ERROR] {name} stop failed: {e}")

        logger.info("Stopping components (timeout: {:.0f}s per component)...".format(
            self._graceful_shutdown_timeout / 6  # Approximate per-component timeout
        ))

        # Mark system as not ready (#S5)
        if self._health_checker:
            self._health_checker.set_ready(False)
            logger.info("  [OK] Health check marked not ready")

        # Stop health check server (#S5)
        if self._health_server:
            self._health_server.stop()
            logger.info("  [OK] Health check server stopped")

        # Stop broker sync first (before dashboard)
        if self._broker_sync:
            await _stop_with_timeout(self._broker_sync.stop(), "Broker metrics sync")

        # Stop dashboard server
        # The uvicorn server task will be cancelled when agent_tasks are cancelled
        if self._dashboard_server:
            logger.info("  [OK] Dashboard server stopping (task will be cancelled)")

        # Stop economic calendar auto-update
        if self._economic_calendar:
            await _stop_with_timeout(self._economic_calendar.stop_auto_update(), "EconomicCalendar")

        # Stop market data first
        if self._market_data:
            await _stop_with_timeout(self._market_data.stop(), "MarketDataManager")

        # Stop compliance/surveillance agents
        if self._transaction_reporting_agent:
            await _stop_with_timeout(
                self._transaction_reporting_agent.stop(), "TransactionReportingAgent"
            )
        if self._surveillance_agent:
            await _stop_with_timeout(self._surveillance_agent.stop(), "SurveillanceAgent")
        if self._reconciliation_agent:
            await self._reconciliation_agent.stop_reconciliation_loop()
            await _stop_with_timeout(self._reconciliation_agent.stop(), "ReconciliationAgent")

        # Stop stop-loss manager before execution agent
        if self._stop_loss_manager:
            await _stop_with_timeout(self._stop_loss_manager.stop(), "StopLossManager")

        # Stop agents in reverse order (execution -> compliance -> risk -> cio -> signals)
        if self._execution_agent:
            await _stop_with_timeout(self._execution_agent.stop(), "ExecutionAgent")

        if self._compliance_agent:
            await _stop_with_timeout(self._compliance_agent.stop(), "ComplianceAgent")

        if self._risk_agent:
            await _stop_with_timeout(self._risk_agent.stop(), "RiskAgent")

        if self._cio_agent:
            await _stop_with_timeout(self._cio_agent.stop(), "CIOAgent")

        for agent in self._signal_agents:
            await _stop_with_timeout(agent.stop(), agent.name)

        # CRITICAL: Flush audit ledger before stopping event bus to ensure all entries are persisted
        if self._audit_ledger:
            try:
                entries_flushed = self._audit_ledger.flush_to_disk()
                logger.info(f"  [OK] Flushed {entries_flushed} audit ledger entries to disk (MiFID II compliance)")
            except Exception as e:
                logger.error(f"  [ERROR] Failed to flush audit ledger: {e}")

        # Stop event bus
        if self._event_bus:
            await _stop_with_timeout(self._event_bus.stop(), "EventBus")

        # Disconnect broker (longer timeout as it may have pending orders)
        if self._broker:
            await _stop_with_timeout(self._broker.disconnect(), "Broker", timeout=10.0)

        shutdown_duration = (datetime.now(timezone.utc) - shutdown_start).total_seconds()

        self._audit_logger.log_system_event("stopped", {
            "uptime_seconds": (datetime.now(timezone.utc) - self._startup_time).total_seconds() if self._startup_time else 0,
            "shutdown_duration_seconds": shutdown_duration,
            "graceful": True,
        })

        logger.info("=" * 60)
        logger.info(f"SHUTDOWN COMPLETE (took {shutdown_duration:.1f}s)")
        logger.info("=" * 60)

    async def _handle_risk_alert(self, event: RiskAlertEvent) -> None:
        """
        Handle risk alerts - AUTONOMOUS MODE: Never triggers shutdown.

        The system manages risk autonomously through:
        - Progressive position reduction
        - Defensive mode (exits only)
        - Position closure based on drawdown levels

        The system NEVER shuts down automatically.
        """
        if event.halt_trading:
            # Log the alert but DO NOT shutdown - autonomous risk management instead
            logger.critical("=" * 60)
            logger.critical("CRITICAL RISK ALERT - AUTONOMOUS MODE ACTIVE")
            logger.critical(f"Alert: {event.message}")
            logger.critical("System continues running with autonomous risk management")
            logger.critical("=" * 60)
            # NOTE: We do NOT call self._shutdown_event.set()
            # The system stays running and manages risk autonomously
        else:
            logger.warning(f"Risk Alert: {event.message}")

    def _on_fill(self, fill: FillEvent) -> None:
        """Handle fill notifications for monitoring and dashboard."""
        if self._monitoring:
            self._monitoring.metrics.record_metric(
                "execution.fill",
                fill.filled_quantity * fill.fill_price,
            )
            logger.info(f"Fill received: {fill.side.value} {fill.filled_quantity} {fill.symbol} @ {fill.fill_price}")

        # CRITICAL: Publish FillEvent to EventBus so dashboard receives it
        # Without this, dashboard won't see fills and positions won't update
        if self._event_bus:
            # Use fire-and-forget publish (sync callback)
            asyncio.create_task(self._event_bus.publish(fill))

    def _on_market_data(self, data: MarketDataEvent) -> None:
        """Handle market data for monitoring."""
        if data.last > 0:
            if self._monitoring:
                self._monitoring.metrics.record_metric(
                    f"market.{data.symbol}.last",
                    data.last,
                )
            # Wire live prices to ExecutionAgent for stop orders and slippage calc
            if self._execution_agent:
                self._execution_agent.update_price(data.symbol, data.last)
            # Wire live prices to CIO for conviction-based position sizing
            if self._cio_agent:
                self._cio_agent.update_price(data.symbol, data.last)
            # Wire live prices to RiskAgent for accurate notional/leverage checks
            if self._risk_agent:
                self._risk_agent.update_price(data.symbol, data.last)

    def request_shutdown(self, reason: str = "user request") -> None:
        """Request graceful shutdown."""
        logger.info(f"Shutdown requested: {reason}")
        if self._audit_logger:
            self._audit_logger.log_system_event("shutdown_requested", {"reason": reason})
        self._shutdown_event.set()

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "running": self._running,
            "startup_time": self._startup_time.isoformat() if self._startup_time else None,
            "uptime_seconds": (datetime.now(timezone.utc) - self._startup_time).total_seconds() if self._startup_time else 0,
            "mode": self._config.get("firm", {}).get("mode", "paper"),
            "broker": {
                "connected": self._broker.is_connected if self._broker else False,
                "account_id": self._broker.account_id if self._broker and self._broker.is_connected else None,
            },
            "event_bus_queue_size": self._event_bus.queue_size if self._event_bus else 0,
            "agents": {
                "signal": [a.get_status() for a in self._signal_agents],
                "cio": self._cio_agent.get_status() if self._cio_agent else None,
                "risk": self._risk_agent.get_status() if self._risk_agent else None,
                "compliance": self._compliance_agent.get_status() if self._compliance_agent else None,
                "execution": self._execution_agent.get_status() if self._execution_agent else None,
                "surveillance": self._surveillance_agent.get_status() if self._surveillance_agent else None,
                "transaction_reporting": self._transaction_reporting_agent.get_status() if self._transaction_reporting_agent else None,
            },
            "infrastructure": {
                "contract_specs": self._contract_specs is not None,
                "correlation_manager": self._correlation_manager is not None,
                "futures_roll_manager": self._futures_roll_manager is not None,
                "var_calculator": self._var_calculator is not None,
                "stress_tester": self._stress_tester is not None,
                "position_sizer": self._position_sizer is not None,
                "attribution": self._attribution is not None,
                "best_execution": self._best_execution is not None,
            },
        }


def setup_signal_handlers(orchestrator: TradingFirmOrchestrator) -> None:
    """Set up signal handlers for graceful shutdown."""
    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}")
        orchestrator.request_shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    # SIGTERM is not available on Windows
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, handle_signal)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Trading Firm - Multi-Agent Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with default config.yaml
  python main.py --config config.simple.yaml  # Use simple config
  python main.py --dry-run                # Validate config without starting
  python main.py --dry-run --config my.yaml   # Validate specific config
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Validate configuration without starting the system",
    )
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_arguments()

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    Path("logs/agents").mkdir(exist_ok=True)

    # Configure logging
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/system.log"),
        ],
    )

    print()
    print("=" * 60)
    print("     AI TRADING FIRM - Multi-Agent Trading System")
    print("=" * 60)
    if args.dry_run:
        print("              [DRY-RUN MODE - Validation Only]")
        print("=" * 60)
    print()

    # Initialize orchestrator with config path
    orchestrator = TradingFirmOrchestrator(config_path=args.config)

    # Enable dry-run mode if requested
    if args.dry_run:
        orchestrator.set_dry_run(True)

    # Set up signal handlers
    setup_signal_handlers(orchestrator)

    try:
        # Initialize
        await orchestrator.initialize()

        # In dry-run mode, stop after initialization
        if args.dry_run:
            logger.info("Dry-run complete. Exiting.")
            return

        # Start
        await orchestrator.start()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except ConnectionRefusedError:
        logger.error("=" * 60)
        logger.error("CONNECTION ERROR: Could not connect to Interactive Brokers")
        logger.error("=" * 60)
        logger.error("Checklist:")
        logger.error("  1. Is IB Gateway or TWS running?")
        logger.error("  2. Is API enabled in IB Gateway/TWS settings?")
        logger.error("  3. Check config port (default: 4002 for paper, 4001 for live)")
        logger.error("  4. Check firewall settings")
    except FileNotFoundError as e:
        if "config" in str(e).lower():
            logger.error("Config file not found. Run: cp config.simple.yaml config.yaml")
        else:
            logger.error(f"File not found: {e}")
    except ValueError as e:
        if "validation" in str(e).lower() or "configuration" in str(e).lower():
            logger.error("Fix configuration errors above and try again")
        elif "live trading" in str(e).lower():
            logger.error("Live trading mode requires explicit confirmation")
        else:
            logger.error(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        if not args.dry_run:
            await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Integrated Trading System with Dashboard
=========================================

This script properly starts the trading system and dashboard together,
sharing a single EventBus for real-time data flow.

Architecture:
    main.py (TradingFirmOrchestrator)
            |
            v
    [Shared EventBus] -----> Dashboard Server
            |                      |
            v                      v
    Signal/Decision/Fill   WebSocket clients
    Events                 (real-time updates)

Usage:
    python dashboard/run_with_trading.py
    python dashboard/run_with_trading.py --config config.simple.yaml
    python dashboard/run_with_trading.py --dashboard-port 8090

Per CLAUDE.md:
- Event-driven, no polling loops
- Observable and auditable system
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn

from core.event_bus import EventBus
from core.events import EventType
from dashboard.server import create_dashboard_server, DashboardServer


logger = logging.getLogger(__name__)


class IntegratedTradingSystem:
    """
    Integrated system that runs both trading and dashboard together.

    Shares a single EventBus between the TradingFirmOrchestrator and
    DashboardServer for real-time event propagation.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        dashboard_host: str = "0.0.0.0",
        dashboard_port: int = 8080,
        dry_run: bool = False,
    ):
        self._config_path = config_path
        self._dashboard_host = dashboard_host
        self._dashboard_port = dashboard_port
        self._dry_run = dry_run

        # Shared components
        self._event_bus: EventBus | None = None
        self._orchestrator: Any = None  # TradingFirmOrchestrator
        self._dashboard: DashboardServer | None = None

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._startup_time: datetime | None = None

    async def initialize(self) -> None:
        """Initialize all components with shared EventBus."""
        logger.info("=" * 60)
        logger.info("INTEGRATED TRADING SYSTEM - INITIALIZING")
        logger.info("=" * 60)

        self._startup_time = datetime.now(timezone.utc)

        # Create shared EventBus
        self._event_bus = EventBus(
            max_queue_size=10000,
            signal_timeout=5.0,
            barrier_timeout=10.0,
        )
        logger.info("Shared EventBus created")

        # Create dashboard server with shared EventBus
        self._dashboard = create_dashboard_server(
            event_bus=self._event_bus,
            host=self._dashboard_host,
            port=self._dashboard_port,
        )
        logger.info(f"Dashboard server created on {self._dashboard_host}:{self._dashboard_port}")

        # Import and create orchestrator (late import to avoid circular deps)
        from main import TradingFirmOrchestrator

        # Create orchestrator but inject our shared EventBus
        self._orchestrator = TradingFirmOrchestrator(config_path=self._config_path)

        if self._dry_run:
            self._orchestrator.set_dry_run(True)

        # Initialize orchestrator (this loads config, creates agents, etc.)
        await self._orchestrator.initialize()

        # CRITICAL: Replace the orchestrator's EventBus with our shared one
        # This allows dashboard to receive all events
        await self._inject_shared_event_bus()

        logger.info("Initialization complete")

    async def _inject_shared_event_bus(self) -> None:
        """
        Inject the shared EventBus into all orchestrator components.

        This is the key integration point - it ensures the dashboard
        receives all events from the trading system.
        """
        logger.info("Injecting shared EventBus into trading system components...")

        # Get the orchestrator's event bus subscriptions
        original_bus = self._orchestrator._event_bus

        if original_bus is None:
            logger.warning("Orchestrator has no EventBus, using shared bus directly")
            self._orchestrator._event_bus = self._event_bus
            return

        # Transfer signal agent registrations
        for agent_name in original_bus._signal_agents:
            self._event_bus.register_signal_agent(agent_name)

        # Transfer subscriptions from original bus to shared bus
        for event_type, handlers in original_bus._subscribers.items():
            for handler in handlers:
                self._event_bus.subscribe(event_type, handler)

        # Replace the event bus reference in orchestrator
        self._orchestrator._event_bus = self._event_bus

        # Update event bus reference in all agents
        agents_to_update = [
            self._orchestrator._cio_agent,
            self._orchestrator._risk_agent,
            self._orchestrator._compliance_agent,
            self._orchestrator._execution_agent,
            self._orchestrator._surveillance_agent,
            self._orchestrator._transaction_reporting_agent,
        ] + self._orchestrator._signal_agents

        for agent in agents_to_update:
            if agent and hasattr(agent, '_event_bus'):
                agent._event_bus = self._event_bus

        # Update market data manager
        if self._orchestrator._market_data and hasattr(self._orchestrator._market_data, '_event_bus'):
            self._orchestrator._market_data._event_bus = self._event_bus

        # Wire dashboard state to receive events
        if self._dashboard:
            await self._dashboard._state.initialize()

        logger.info("Shared EventBus injection complete")

    async def start(self) -> None:
        """Start both trading system and dashboard."""
        if self._dry_run:
            logger.info("Dry-run mode - not starting services")
            return

        logger.info("=" * 60)
        logger.info("INTEGRATED TRADING SYSTEM - STARTING")
        logger.info("=" * 60)

        self._running = True

        # Start uvicorn server for dashboard in background
        config = uvicorn.Config(
            self._dashboard.app,
            host=self._dashboard_host,
            port=self._dashboard_port,
            log_level="warning",
        )
        server = uvicorn.Server(config)

        # Create tasks
        dashboard_task = asyncio.create_task(server.serve())
        trading_task = asyncio.create_task(self._run_trading())

        logger.info("=" * 60)
        logger.info("SYSTEM STARTED")
        logger.info(f"  Dashboard: http://{self._dashboard_host}:{self._dashboard_port}")
        logger.info(f"  WebSocket: ws://{self._dashboard_host}:{self._dashboard_port}/ws")
        logger.info(f"  API Status: http://{self._dashboard_host}:{self._dashboard_port}/api/status")
        logger.info("=" * 60)

        # Wait for shutdown signal
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass

        # Stop services
        self._running = False
        server.should_exit = True

        await self.stop()

        # Cancel tasks
        for task in [dashboard_task, trading_task]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _run_trading(self) -> None:
        """Run the trading system (delegates to orchestrator.start)."""
        try:
            # Subscribe to events for kill-switch handling
            self._event_bus.subscribe(EventType.KILL_SWITCH, self._handle_kill_switch)

            # Start event bus
            event_bus_task = asyncio.create_task(self._event_bus.start())

            # Start all agents via orchestrator
            # Note: We don't call orchestrator.start() because it has its own event loop
            # Instead, we start agents manually

            # Start signal agents
            for agent in self._orchestrator._signal_agents:
                await agent.start()

            # Start sequential agents
            if self._orchestrator._cio_agent:
                await self._orchestrator._cio_agent.start()
            if self._orchestrator._risk_agent:
                await self._orchestrator._risk_agent.start()
            if self._orchestrator._compliance_agent:
                await self._orchestrator._compliance_agent.start()
            if self._orchestrator._execution_agent:
                await self._orchestrator._execution_agent.start()

            # Start compliance agents
            if self._orchestrator._surveillance_agent:
                await self._orchestrator._surveillance_agent.start()
            if self._orchestrator._transaction_reporting_agent:
                await self._orchestrator._transaction_reporting_agent.start()

            # Start market data
            if self._orchestrator._market_data:
                await self._orchestrator._market_data.start()

                # If broker not connected, generate simulated data
                if self._orchestrator._broker and not self._orchestrator._broker.is_connected:
                    logger.info("Starting simulated market data")
                    asyncio.create_task(
                        self._orchestrator._market_data.generate_simulated_data()
                    )

            logger.info("Trading system components started")

            # Wait for shutdown
            while self._running:
                await asyncio.sleep(1.0)

            # Stop event bus
            await self._event_bus.stop()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Trading system error: {e}")

    async def _handle_kill_switch(self, event: Any) -> None:
        """Handle kill switch event."""
        logger.critical("KILL SWITCH ACTIVATED - Initiating shutdown")
        self._shutdown_event.set()

    async def stop(self) -> None:
        """Stop all components gracefully."""
        logger.info("=" * 60)
        logger.info("INTEGRATED TRADING SYSTEM - STOPPING")
        logger.info("=" * 60)

        self._running = False

        # Stop market data
        if self._orchestrator and self._orchestrator._market_data:
            try:
                await asyncio.wait_for(
                    self._orchestrator._market_data.stop(),
                    timeout=5.0
                )
            except Exception as e:
                logger.warning(f"Market data stop error: {e}")

        # Stop agents in reverse order
        if self._orchestrator:
            agents_to_stop = [
                self._orchestrator._transaction_reporting_agent,
                self._orchestrator._surveillance_agent,
                self._orchestrator._execution_agent,
                self._orchestrator._compliance_agent,
                self._orchestrator._risk_agent,
                self._orchestrator._cio_agent,
            ] + list(reversed(self._orchestrator._signal_agents))

            for agent in agents_to_stop:
                if agent:
                    try:
                        await asyncio.wait_for(agent.stop(), timeout=3.0)
                    except Exception as e:
                        logger.warning(f"Agent stop error: {e}")

        # Disconnect broker
        if self._orchestrator and self._orchestrator._broker:
            try:
                await asyncio.wait_for(
                    self._orchestrator._broker.disconnect(),
                    timeout=5.0
                )
            except Exception as e:
                logger.warning(f"Broker disconnect error: {e}")

        shutdown_duration = (
            (datetime.now(timezone.utc) - self._startup_time).total_seconds()
            if self._startup_time else 0
        )

        logger.info("=" * 60)
        logger.info(f"SHUTDOWN COMPLETE (runtime: {shutdown_duration:.1f}s)")
        logger.info("=" * 60)

    def request_shutdown(self, reason: str = "user request") -> None:
        """Request graceful shutdown."""
        logger.info(f"Shutdown requested: {reason}")
        self._shutdown_event.set()

    def get_status(self) -> dict[str, Any]:
        """Get system status."""
        return {
            "running": self._running,
            "startup_time": self._startup_time.isoformat() if self._startup_time else None,
            "dashboard": {
                "host": self._dashboard_host,
                "port": self._dashboard_port,
                "connections": (
                    self._dashboard.connection_manager.connection_count
                    if self._dashboard else 0
                ),
            },
            "event_bus": {
                "queue_size": self._event_bus.queue_size if self._event_bus else 0,
                "running": self._event_bus.is_running if self._event_bus else False,
            },
            "orchestrator": (
                self._orchestrator.get_status() if self._orchestrator else None
            ),
        }


def setup_signal_handlers(system: IntegratedTradingSystem) -> None:
    """Set up signal handlers for graceful shutdown."""
    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}")
        system.request_shutdown("signal received")

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, handle_signal)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Trading Firm - Integrated System with Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dashboard/run_with_trading.py
  python dashboard/run_with_trading.py --config config.simple.yaml
  python dashboard/run_with_trading.py --dashboard-port 8090 --dry-run
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--dashboard-host",
        default="0.0.0.0",
        help="Dashboard server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--dashboard-port", "-p",
        type=int,
        default=8080,
        help="Dashboard server port (default: 8080)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Validate configuration without starting services",
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
            logging.FileHandler("logs/integrated_system.log"),
        ],
    )

    print()
    print("=" * 60)
    print("     AI TRADING FIRM - Integrated System + Dashboard")
    print("=" * 60)
    if args.dry_run:
        print("              [DRY-RUN MODE - Validation Only]")
    print("=" * 60)
    print()

    # Create integrated system
    system = IntegratedTradingSystem(
        config_path=args.config,
        dashboard_host=args.dashboard_host,
        dashboard_port=args.dashboard_port,
        dry_run=args.dry_run,
    )

    # Set up signal handlers
    setup_signal_handlers(system)

    try:
        await system.initialize()

        if args.dry_run:
            logger.info("Dry-run complete. Configuration valid.")
            return

        await system.start()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
    finally:
        if not args.dry_run:
            await system.stop()


if __name__ == "__main__":
    asyncio.run(main())

"""
Base Agent
==========

Abstract base class for all agents in the trading system.
Enforces single-responsibility principle and stateless design.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from core.events import Event, EventType

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    enabled: bool = True
    timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 10.0  # Max time for graceful shutdown
    parameters: dict[str, Any] = field(default_factory=dict)


class ShutdownState:
    """Tracks agent shutdown state for graceful termination."""
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class BaseAgent(ABC):
    """
    Abstract base class for all trading system agents.

    Design principles:
    - Single responsibility per agent
    - Stateless where possible (state in events)
    - All actions logged for audit
    - Timeout and fault tolerance built-in
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        self._config = config
        self._event_bus = event_bus
        self._audit_logger = audit_logger
        self._running = False
        self._started_at: datetime | None = None
        self._last_heartbeat: datetime | None = None
        self._event_count = 0
        self._error_count = 0

        # Graceful shutdown state (#S3)
        self._shutdown_state = ShutdownState.STOPPED
        self._pending_tasks: set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        self._cleanup_handlers: list[Callable[[], Coroutine[Any, Any, None]]] = []

    @property
    def name(self) -> str:
        """Agent name."""
        return self._config.name

    @property
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._running

    @property
    def is_enabled(self) -> bool:
        """Check if agent is enabled."""
        return self._config.enabled

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the agent.

        Called once before start. Use for setup that requires async.
        """
        pass

    @abstractmethod
    async def process_event(self, event: Event) -> None:
        """
        Process an incoming event.

        This is the main event handler for the agent.
        Must be implemented by all agents.
        """
        pass

    @abstractmethod
    def get_subscribed_events(self) -> list[EventType]:
        """
        Return list of event types this agent subscribes to.

        Used by orchestrator to set up event routing.
        """
        pass

    async def start(self) -> None:
        """Start the agent."""
        if not self._config.enabled:
            logger.info(f"Agent {self.name} is disabled, not starting")
            return

        self._running = True
        self._shutdown_state = ShutdownState.RUNNING
        self._shutdown_event.clear()
        self._started_at = datetime.now(timezone.utc)
        self._pending_tasks.clear()

        # Subscribe to events
        for event_type in self.get_subscribed_events():
            self._event_bus.subscribe(event_type, self._handle_event)

        await self.initialize()

        self._audit_logger.log_agent_event(
            agent_name=self.name,
            event_type="started",
            details={"subscribed_events": [e.value for e in self.get_subscribed_events()]},
        )

        logger.info(f"Agent {self.name} started")

    async def stop(self, timeout: float | None = None) -> bool:
        """
        Stop the agent gracefully with timeout (#S3).

        Graceful shutdown sequence:
        1. Stop accepting new events
        2. Wait for pending tasks to complete (with timeout)
        3. Run cleanup handlers
        4. Unsubscribe from events
        5. Log shutdown

        Args:
            timeout: Override shutdown timeout (uses config default if None)

        Returns:
            True if shutdown completed gracefully, False if forced/timed out
        """
        if self._shutdown_state != ShutdownState.RUNNING:
            logger.warning(f"Agent {self.name} already shutting down or stopped")
            return True

        shutdown_timeout = timeout or self._config.shutdown_timeout_seconds
        self._shutdown_state = ShutdownState.SHUTTING_DOWN
        self._running = False

        logger.info(f"Agent {self.name} starting graceful shutdown (timeout={shutdown_timeout}s)")

        graceful = True

        # Wait for pending tasks to complete
        if self._pending_tasks:
            pending_count = len(self._pending_tasks)
            logger.info(f"Agent {self.name} waiting for {pending_count} pending tasks")

            try:
                # Wait for pending tasks with timeout
                done, pending = await asyncio.wait(
                    self._pending_tasks,
                    timeout=shutdown_timeout * 0.7,  # Use 70% of timeout for tasks
                    return_when=asyncio.ALL_COMPLETED
                )

                if pending:
                    logger.warning(
                        f"Agent {self.name}: {len(pending)} tasks did not complete, cancelling"
                    )
                    for task in pending:
                        task.cancel()
                    graceful = False

            except Exception as e:
                logger.error(f"Error waiting for pending tasks: {e}")
                graceful = False

        # Run cleanup handlers
        cleanup_timeout = shutdown_timeout * 0.2  # Use 20% of timeout for cleanup
        for handler in self._cleanup_handlers:
            try:
                await asyncio.wait_for(handler, timeout=cleanup_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Agent {self.name} cleanup handler timed out")
                graceful = False
            except Exception as e:
                logger.error(f"Agent {self.name} cleanup handler error: {e}")
                graceful = False

        # Subclass-specific cleanup
        try:
            await self._on_shutdown()
        except Exception as e:
            logger.error(f"Agent {self.name} _on_shutdown error: {e}")
            graceful = False

        # Unsubscribe from events
        for event_type in self.get_subscribed_events():
            self._event_bus.unsubscribe(event_type, self._handle_event)

        self._shutdown_state = ShutdownState.STOPPED
        self._shutdown_event.set()

        self._audit_logger.log_agent_event(
            agent_name=self.name,
            event_type="stopped",
            details={
                "events_processed": self._event_count,
                "errors": self._error_count,
                "uptime_seconds": self._get_uptime_seconds(),
                "graceful": graceful,
            },
        )

        logger.info(f"Agent {self.name} stopped (graceful={graceful})")
        return graceful

    async def _on_shutdown(self) -> None:
        """
        Subclass hook for custom shutdown logic.

        Override this method to add cleanup specific to the agent type.
        Called after pending tasks complete but before unsubscribing from events.
        """
        pass

    def add_cleanup_handler(self, handler: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """
        Add a cleanup handler to run during shutdown.

        Args:
            handler: Async function to call during shutdown
        """
        self._cleanup_handlers.append(handler)

    def track_task(self, task: asyncio.Task) -> None:
        """
        Track an async task for graceful shutdown.

        Call this for any background tasks started by the agent.
        """
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    @property
    def is_shutting_down(self) -> bool:
        """Check if agent is in shutdown state."""
        return self._shutdown_state == ShutdownState.SHUTTING_DOWN

    async def wait_for_shutdown(self) -> None:
        """Wait until agent has fully stopped."""
        await self._shutdown_event.wait()

    async def _handle_event(self, event: Event) -> None:
        """
        Internal event handler with error handling and metrics.

        Respects shutdown state by not processing new events during shutdown.
        """
        if not self._running or self._shutdown_state != ShutdownState.RUNNING:
            return

        try:
            # Create task and track it for graceful shutdown
            task = asyncio.create_task(
                asyncio.wait_for(
                    self.process_event(event),
                    timeout=self._config.timeout_seconds,
                )
            )
            self.track_task(task)
            await task

            self._event_count += 1
            self._last_heartbeat = datetime.now(timezone.utc)

        except asyncio.TimeoutError:
            self._error_count += 1
            logger.error(f"Agent {self.name} timed out processing event {event.event_id}")
            self._audit_logger.log_agent_event(
                agent_name=self.name,
                event_type="timeout",
                details={"event_id": event.event_id},
            )

        except asyncio.CancelledError:
            # Expected during shutdown
            logger.debug(f"Agent {self.name} event processing cancelled (shutdown)")

        except Exception as e:
            self._error_count += 1
            logger.error(f"Agent {self.name} error: {e}")
            self._audit_logger.log_agent_event(
                agent_name=self.name,
                event_type="error",
                details={"event_id": event.event_id, "error": str(e)},
            )

    def _get_uptime_seconds(self) -> float:
        """Get agent uptime in seconds."""
        if self._started_at is None:
            return 0.0
        return (datetime.now(timezone.utc) - self._started_at).total_seconds()

    def get_status(self) -> dict[str, Any]:
        """Get agent status for monitoring."""
        return {
            "name": self.name,
            "enabled": self._config.enabled,
            "running": self._running,
            "shutdown_state": self._shutdown_state,
            "pending_tasks": len(self._pending_tasks),
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
            "events_processed": self._event_count,
            "errors": self._error_count,
            "uptime_seconds": self._get_uptime_seconds(),
        }


class SignalAgent(BaseAgent):
    """
    Base class for signal-generating agents.

    Signal agents:
    - Subscribe to market data
    - Generate SignalEvents (advisory only)
    - Run in parallel (fan-out)
    - NEVER send orders directly
    """

    async def start(self) -> None:
        """Start and register as signal agent."""
        self._event_bus.register_signal_agent(self.name)
        await super().start()

    def get_subscribed_events(self) -> list[EventType]:
        """Signal agents subscribe to market data by default."""
        return [EventType.MARKET_DATA]


class DecisionAgent(BaseAgent):
    """
    Base class for decision-making agents.

    In this system, there is exactly ONE decision agent: the CIO.
    """

    def get_subscribed_events(self) -> list[EventType]:
        """Decision agent subscribes to signals."""
        return [EventType.SIGNAL]


class ValidationAgent(BaseAgent):
    """
    Base class for validation agents (Risk, Compliance).
    """

    def get_subscribed_events(self) -> list[EventType]:
        """Validation agents subscribe to decisions."""
        return [EventType.DECISION]


class ExecutionAgent(BaseAgent):
    """
    Base class for execution agent.

    ONLY this agent can send orders to the broker.
    """

    def get_subscribed_events(self) -> list[EventType]:
        """Execution agent subscribes to validated decisions."""
        return [EventType.VALIDATED_DECISION]

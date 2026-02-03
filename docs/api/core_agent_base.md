# agent_base

**Path**: `C:\Users\Alexa\ai-trading-firm\core\agent_base.py`

## Overview

Base Agent
==========

Abstract base class for all agents in the trading system.
Enforces single-responsibility principle and stateless design.

## Classes

### AgentConfig

Configuration for an agent.

### ShutdownState

Tracks agent shutdown state for graceful termination.

### BaseAgent

**Inherits from**: ABC

Abstract base class for all trading system agents.

Design principles:
- Single responsibility per agent
- Stateless where possible (state in events)
- All actions logged for audit
- Timeout and fault tolerance built-in

#### Methods

##### `def __init__(self, config: AgentConfig, event_bus: EventBus, audit_logger: AuditLogger)`

##### `def name(self) -> str`

Agent name.

##### `def is_running(self) -> bool`

Check if agent is running.

##### `def is_enabled(self) -> bool`

Check if agent is enabled.

##### `async def initialize(self) -> None`

Initialize the agent.

Called once before start. Use for setup that requires async.

##### `async def process_event(self, event: Event) -> None`

Process an incoming event.

This is the main event handler for the agent.
Must be implemented by all agents.

##### `def get_subscribed_events(self) -> list[EventType]`

Return list of event types this agent subscribes to.

Used by orchestrator to set up event routing.

##### `async def start(self) -> None`

Start the agent.

##### `async def stop(self, timeout: ) -> bool`

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

##### `def add_cleanup_handler(self, handler: Callable[, Coroutine[Any, Any, None]]) -> None`

Add a cleanup handler to run during shutdown.

Args:
    handler: Async function to call during shutdown

##### `def track_task(self, task: asyncio.Task) -> None`

Track an async task for graceful shutdown.

Call this for any background tasks started by the agent.

##### `def is_shutting_down(self) -> bool`

Check if agent is in shutdown state.

##### `async def wait_for_shutdown(self) -> None`

Wait until agent has fully stopped.

##### `def get_status(self) -> dict[str, Any]`

Get agent status for monitoring.

### SignalAgent

**Inherits from**: BaseAgent

Base class for signal-generating agents.

Signal agents:
- Subscribe to market data
- Generate SignalEvents (advisory only)
- Run in parallel (fan-out)
- NEVER send orders directly

#### Methods

##### `async def start(self) -> None`

Start and register as signal agent.

##### `def get_subscribed_events(self) -> list[EventType]`

Signal agents subscribe to market data by default.

### DecisionAgent

**Inherits from**: BaseAgent

Base class for decision-making agents.

In this system, there is exactly ONE decision agent: the CIO.

#### Methods

##### `def get_subscribed_events(self) -> list[EventType]`

Decision agent subscribes to signals.

### ValidationAgent

**Inherits from**: BaseAgent

Base class for validation agents (Risk, Compliance).

#### Methods

##### `def get_subscribed_events(self) -> list[EventType]`

Validation agents subscribe to decisions.

### ExecutionAgent

**Inherits from**: BaseAgent

Base class for execution agent.

ONLY this agent can send orders to the broker.

#### Methods

##### `def get_subscribed_events(self) -> list[EventType]`

Execution agent subscribes to validated decisions.

# agent_factory

**Path**: `C:\Users\Alexa\ai-trading-firm\core\agent_factory.py`

## Overview

Agent Factory
=============

Factory for creating and configuring trading agents.

This module extracts agent creation logic from TradingFirmOrchestrator
following the Single Responsibility Principle (SRP).

The factory:
- Creates signal agents (parallel execution)
- Creates decision agent (CIO - single authority)
- Creates validation agents (Risk, Compliance)
- Creates execution agent
- Creates compliance/surveillance agents (EU/AMF)

## Classes

### AgentFactoryConfig

Configuration for agent factory.

### CreatedAgents

Container for all created agents.

#### Methods

##### `def get_all_agents(self) -> list[Any]`

Get all non-None agents as a list.

### AgentFactory

Factory for creating trading system agents.

Centralizes agent creation logic, improving testability
and following Single Responsibility Principle.

Usage:
    factory = AgentFactory(event_bus, audit_logger, broker, config)
    agents = factory.create_all_agents()

#### Methods

##### `def __init__(self, event_bus: EventBus, audit_logger: AuditLogger, broker: IBBroker | None, config: )`

##### `def create_all_agents(self) -> CreatedAgents`

Create all trading agents.

Returns:
    CreatedAgents container with all initialized agents

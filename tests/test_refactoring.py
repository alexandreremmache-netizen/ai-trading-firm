"""
Tests for SOLID Refactoring Components
======================================

Tests for:
1. AgentFactory
2. EventBus Health Check
3. DI Container
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch


class TestAgentFactory:
    """Test AgentFactory functionality."""

    def test_agent_factory_import(self):
        """Test that AgentFactory can be imported."""
        from core.agent_factory import AgentFactory, AgentFactoryConfig, CreatedAgents
        assert AgentFactory is not None
        assert AgentFactoryConfig is not None
        assert CreatedAgents is not None

    def test_agent_factory_config_defaults(self):
        """Test AgentFactoryConfig has sensible defaults."""
        from core.agent_factory import AgentFactoryConfig

        config = AgentFactoryConfig()
        assert config.agents_config == {}
        assert config.risk_config == {}
        assert config.compliance_config == {}

    def test_created_agents_container(self):
        """Test CreatedAgents container."""
        from core.agent_factory import CreatedAgents

        agents = CreatedAgents()
        assert agents.signal_agents == []
        assert agents.cio_agent is None
        assert agents.risk_agent is None

    def test_created_agents_get_all(self):
        """Test CreatedAgents.get_all_agents()."""
        from core.agent_factory import CreatedAgents

        mock_agent = Mock()
        mock_agent.name = "TestAgent"

        agents = CreatedAgents()
        agents.signal_agents = [mock_agent]
        agents.cio_agent = mock_agent

        all_agents = agents.get_all_agents()
        assert len(all_agents) == 2


class TestDIContainer:
    """Test Dependency Injection Container."""

    def test_di_container_import(self):
        """Test that DIContainer can be imported."""
        from core.dependency_injection import (
            DIContainer,
            DependencyLifecycle,
            DependencyResolutionError,
            CircularDependencyError,
        )
        assert DIContainer is not None
        assert DependencyLifecycle is not None

    def test_di_container_creation(self):
        """Test DIContainer can be created."""
        from core.dependency_injection import DIContainer

        container = DIContainer()
        assert container is not None

    def test_register_singleton(self):
        """Test singleton registration."""
        from core.dependency_injection import DIContainer

        container = DIContainer()
        container.register_singleton("test", lambda: "value")

        assert container.is_registered("test")

    def test_resolve_singleton(self):
        """Test singleton resolution returns same instance."""
        from core.dependency_injection import DIContainer

        container = DIContainer()
        container.register_singleton("test", lambda: {"value": 42})

        instance1 = container.resolve("test")
        instance2 = container.resolve("test")

        assert instance1 is instance2
        assert instance1["value"] == 42

    def test_register_factory(self):
        """Test factory registration returns new instances."""
        from core.dependency_injection import DIContainer

        counter = [0]

        def factory():
            counter[0] += 1
            return {"count": counter[0]}

        container = DIContainer()
        container.register_factory("test", factory)

        instance1 = container.resolve("test")
        instance2 = container.resolve("test")

        assert instance1 is not instance2
        assert instance1["count"] == 1
        assert instance2["count"] == 2

    def test_register_instance(self):
        """Test registering an existing instance."""
        from core.dependency_injection import DIContainer

        instance = {"key": "value"}
        container = DIContainer()
        container.register_instance("test", instance)

        resolved = container.resolve("test")
        assert resolved is instance

    def test_dependency_resolution_error(self):
        """Test error when resolving unknown dependency."""
        from core.dependency_injection import DIContainer, DependencyResolutionError

        container = DIContainer()

        with pytest.raises(DependencyResolutionError):
            container.resolve("nonexistent")

    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        from core.dependency_injection import DIContainer, CircularDependencyError

        container = DIContainer()

        # Create circular dependency: A -> B -> A
        container.register_singleton(
            "A",
            lambda B: {"A": B},
            dependencies=["B"]
        )
        container.register_singleton(
            "B",
            lambda A: {"B": A},
            dependencies=["A"]
        )

        with pytest.raises(CircularDependencyError):
            container.resolve("A")

    def test_dependency_chain(self):
        """Test dependency chain resolution."""
        from core.dependency_injection import DIContainer

        container = DIContainer()

        container.register_singleton("config", lambda: {"value": 42})
        container.register_singleton(
            "service",
            lambda config: {"config": config},
            dependencies=["config"]
        )

        service = container.resolve("service")
        assert service["config"]["value"] == 42

    def test_unregister(self):
        """Test unregistering a dependency."""
        from core.dependency_injection import DIContainer

        container = DIContainer()
        container.register_singleton("test", lambda: "value")

        assert container.is_registered("test")

        result = container.unregister("test")
        assert result is True
        assert not container.is_registered("test")

    def test_clear(self):
        """Test clearing all registrations."""
        from core.dependency_injection import DIContainer

        container = DIContainer()
        container.register_singleton("test1", lambda: "value1")
        container.register_singleton("test2", lambda: "value2")

        container.clear()

        assert not container.is_registered("test1")
        assert not container.is_registered("test2")

    def test_get_dependency_graph(self):
        """Test getting dependency graph."""
        from core.dependency_injection import DIContainer

        container = DIContainer()
        container.register_singleton("A", lambda: "A")
        container.register_singleton("B", lambda A: f"B({A})", dependencies=["A"])
        container.register_singleton("C", lambda A, B: f"C({A},{B})", dependencies=["A", "B"])

        graph = container.get_dependency_graph()

        assert graph["A"] == []
        assert graph["B"] == ["A"]
        assert graph["C"] == ["A", "B"]

    def test_get_status(self):
        """Test getting container status."""
        from core.dependency_injection import DIContainer

        container = DIContainer()
        container.register_singleton("test", lambda: "value")
        container.resolve("test")  # Instantiate

        status = container.get_status()

        assert status["registrations"] == 1
        assert status["singletons_instantiated"] == 1

    @pytest.mark.asyncio
    async def test_resolve_async(self):
        """Test async resolution."""
        from core.dependency_injection import DIContainer

        async def async_factory():
            await asyncio.sleep(0.01)
            return "async_value"

        container = DIContainer()
        container.register_singleton("async_test", async_factory, is_async=True)

        result = await container.resolve_async("async_test")
        assert result == "async_value"

    @pytest.mark.asyncio
    async def test_initialize_all(self):
        """Test initializing all singletons."""
        from core.dependency_injection import DIContainer

        container = DIContainer()
        container.register_singleton("s1", lambda: "value1")
        container.register_singleton("s2", lambda: "value2")
        container.register_factory("f1", lambda: "factory")  # Should not be initialized

        instances = await container.initialize_all()

        assert len(instances) == 2
        assert "s1" in instances
        assert "s2" in instances
        assert "f1" not in instances

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test shutdown with handlers."""
        from core.dependency_injection import DIContainer

        container = DIContainer()
        shutdown_called = [False]

        def shutdown_handler():
            shutdown_called[0] = True

        container.add_shutdown_handler(shutdown_handler)
        await container.shutdown()

        assert shutdown_called[0] is True


class TestServiceProvider:
    """Test ServiceProvider static interface."""

    def test_service_provider_import(self):
        """Test ServiceProvider can be imported."""
        from core.dependency_injection import ServiceProvider
        assert ServiceProvider is not None

    def test_service_provider_not_initialized(self):
        """Test error when not initialized."""
        from core.dependency_injection import ServiceProvider

        ServiceProvider._container = None  # Reset

        with pytest.raises(RuntimeError):
            ServiceProvider.get("test")

    def test_service_provider_set_container(self):
        """Test setting container."""
        from core.dependency_injection import ServiceProvider, DIContainer

        container = DIContainer()
        container.register_instance("test", "value")

        ServiceProvider.set_container(container)

        assert ServiceProvider.get("test") == "value"

        # Cleanup
        ServiceProvider._container = None


class TestEventBusHealthCheck:
    """Test EventBus health check functionality."""

    def test_health_check_imports(self):
        """Test health check classes can be imported."""
        from core.event_bus import (
            EventBusHealthStatus,
            HealthCheckConfig,
            HealthCheckResult,
        )
        assert EventBusHealthStatus is not None
        assert HealthCheckConfig is not None
        assert HealthCheckResult is not None

    def test_health_status_values(self):
        """Test health status enum values."""
        from core.event_bus import EventBusHealthStatus

        assert EventBusHealthStatus.HEALTHY.value == "healthy"
        assert EventBusHealthStatus.DEGRADED.value == "degraded"
        assert EventBusHealthStatus.UNHEALTHY.value == "unhealthy"
        assert EventBusHealthStatus.RECOVERING.value == "recovering"

    def test_health_check_config_defaults(self):
        """Test HealthCheckConfig has sensible defaults."""
        from core.event_bus import HealthCheckConfig

        config = HealthCheckConfig()
        assert config.enabled is True
        assert config.check_interval_seconds == 10.0
        assert config.max_processing_latency_ms == 1000.0
        assert config.max_consecutive_errors == 5

    def test_health_check_result_to_dict(self):
        """Test HealthCheckResult.to_dict()."""
        from core.event_bus import HealthCheckResult, EventBusHealthStatus
        from datetime import datetime, timezone

        result = HealthCheckResult(
            status=EventBusHealthStatus.HEALTHY,
            timestamp=datetime.now(timezone.utc),
            latency_ms=50.5,
            queue_size=10,
            last_event_processed=datetime.now(timezone.utc),
            consecutive_errors=0,
            message="OK",
        )

        d = result.to_dict()
        assert d["status"] == "healthy"
        assert d["latency_ms"] == 50.5
        assert d["queue_size"] == 10
        assert d["message"] == "OK"

    def test_event_bus_with_health_check(self):
        """Test EventBus accepts health check config."""
        from core.event_bus import EventBus, HealthCheckConfig

        config = HealthCheckConfig(
            enabled=True,
            check_interval_seconds=5.0,
        )

        bus = EventBus(
            max_queue_size=100,
            health_check_config=config,
        )

        assert bus._health_config.enabled is True
        assert bus._health_config.check_interval_seconds == 5.0

    def test_event_bus_health_status_property(self):
        """Test health_status property."""
        from core.event_bus import EventBus, EventBusHealthStatus

        bus = EventBus(max_queue_size=100)
        assert bus.health_status == EventBusHealthStatus.HEALTHY

    def test_event_bus_is_healthy_property(self):
        """Test is_healthy property."""
        from core.event_bus import EventBus

        bus = EventBus(max_queue_size=100)
        assert bus.is_healthy is True

    @pytest.mark.asyncio
    async def test_check_health_returns_result(self):
        """Test check_health returns HealthCheckResult."""
        from core.event_bus import EventBus, HealthCheckResult, EventBusHealthStatus

        bus = EventBus(max_queue_size=100)
        result = await bus.check_health()

        assert isinstance(result, HealthCheckResult)
        assert result.status == EventBusHealthStatus.HEALTHY

    def test_event_bus_status_includes_health(self):
        """Test get_status includes health info."""
        from core.event_bus import EventBus

        bus = EventBus(max_queue_size=100)
        status = bus.get_status()

        assert "health" in status
        assert status["health"]["status"] == "healthy"
        assert status["health"]["enabled"] is True

    def test_on_health_change_callback(self):
        """Test registering health change callback."""
        from core.event_bus import EventBus

        bus = EventBus(max_queue_size=100)
        callback_called = [False]

        def callback(result):
            callback_called[0] = True

        bus.on_health_change(callback)

        assert len(bus._health_callbacks) == 1

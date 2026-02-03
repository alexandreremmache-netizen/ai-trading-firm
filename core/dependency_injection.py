"""
Dependency Injection Container
==============================

Simple DI container for centralizing dependency creation and management.

This module provides:
- Centralized dependency registration and resolution
- Singleton and factory patterns
- Lifecycle management (startup/shutdown)
- Type-safe dependency injection

Benefits:
- Improved testability (easy mocking)
- Reduced coupling between components
- Centralized configuration
- Explicit dependency graph

Usage:
    container = DIContainer()
    container.register_singleton("event_bus", EventBus)
    container.register_factory("agent", lambda: create_agent())

    # Resolve dependencies
    event_bus = container.resolve("event_bus")

    # In tests
    container.register_singleton("event_bus", MockEventBus)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar, Generic, TYPE_CHECKING

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)

T = TypeVar("T")


class DependencyLifecycle(Enum):
    """Lifecycle of a dependency."""
    SINGLETON = "singleton"  # Single instance, created once
    FACTORY = "factory"      # New instance on each resolve
    SCOPED = "scoped"        # One instance per scope


@dataclass
class DependencyRegistration:
    """Registration info for a dependency."""
    name: str
    lifecycle: DependencyLifecycle
    factory: Callable[..., Any]
    dependencies: list[str] = field(default_factory=list)
    instance: Any = None
    is_async: bool = False


class DependencyResolutionError(Exception):
    """Raised when a dependency cannot be resolved."""
    pass


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""
    pass


class DIContainer:
    """
    Simple Dependency Injection Container.

    Features:
    - Singleton and factory registration
    - Dependency chain resolution
    - Async initialization support
    - Lifecycle management

    Thread Safety:
    - Not thread-safe by design (async single-threaded model)
    - Use asyncio.Lock if concurrent access needed
    """

    def __init__(self):
        self._registrations: dict[str, DependencyRegistration] = {}
        self._resolving: set[str] = set()  # For circular dependency detection
        self._initialized = False
        self._shutdown_handlers: list[Callable[[], Any]] = []

    def register_singleton(
        self,
        name: str,
        factory: Callable[..., T],
        dependencies: list[str] | None = None,
        is_async: bool = False,
    ) -> "DIContainer":
        """
        Register a singleton dependency.

        The factory is called once, and the same instance is returned
        on subsequent resolves.

        Args:
            name: Unique name for the dependency
            factory: Factory function to create the instance
            dependencies: Names of dependencies to inject
            is_async: If True, factory is async and returns awaitable

        Returns:
            Self for chaining
        """
        self._registrations[name] = DependencyRegistration(
            name=name,
            lifecycle=DependencyLifecycle.SINGLETON,
            factory=factory,
            dependencies=dependencies or [],
            is_async=is_async,
        )
        logger.debug(f"Registered singleton: {name}")
        return self

    def register_factory(
        self,
        name: str,
        factory: Callable[..., T],
        dependencies: list[str] | None = None,
        is_async: bool = False,
    ) -> "DIContainer":
        """
        Register a factory dependency.

        The factory is called each time the dependency is resolved,
        creating a new instance.

        Args:
            name: Unique name for the dependency
            factory: Factory function to create instances
            dependencies: Names of dependencies to inject
            is_async: If True, factory is async

        Returns:
            Self for chaining
        """
        self._registrations[name] = DependencyRegistration(
            name=name,
            lifecycle=DependencyLifecycle.FACTORY,
            factory=factory,
            dependencies=dependencies or [],
            is_async=is_async,
        )
        logger.debug(f"Registered factory: {name}")
        return self

    def register_instance(self, name: str, instance: Any) -> "DIContainer":
        """
        Register an existing instance as a singleton.

        Args:
            name: Unique name for the dependency
            instance: The instance to register

        Returns:
            Self for chaining
        """
        self._registrations[name] = DependencyRegistration(
            name=name,
            lifecycle=DependencyLifecycle.SINGLETON,
            factory=lambda: instance,
            instance=instance,
        )
        logger.debug(f"Registered instance: {name}")
        return self

    def resolve(self, name: str) -> Any:
        """
        Resolve a dependency by name (sync).

        Args:
            name: Name of the dependency to resolve

        Returns:
            The resolved dependency instance

        Raises:
            DependencyResolutionError: If dependency not found
            CircularDependencyError: If circular dependency detected
        """
        if name not in self._registrations:
            raise DependencyResolutionError(f"Dependency not found: {name}")

        registration = self._registrations[name]

        # Return existing singleton instance
        if registration.lifecycle == DependencyLifecycle.SINGLETON and registration.instance is not None:
            return registration.instance

        # Check for circular dependency
        if name in self._resolving:
            raise CircularDependencyError(
                f"Circular dependency detected while resolving: {name}. "
                f"Resolution stack: {self._resolving}"
            )

        self._resolving.add(name)

        try:
            # Resolve dependencies first
            deps = {}
            for dep_name in registration.dependencies:
                deps[dep_name] = self.resolve(dep_name)

            # Create instance
            if registration.is_async:
                raise DependencyResolutionError(
                    f"Cannot resolve async dependency '{name}' synchronously. "
                    f"Use resolve_async() instead."
                )

            instance = registration.factory(**deps) if deps else registration.factory()

            # Store singleton instance
            if registration.lifecycle == DependencyLifecycle.SINGLETON:
                registration.instance = instance

            return instance

        finally:
            self._resolving.discard(name)

    async def resolve_async(self, name: str) -> Any:
        """
        Resolve a dependency by name (async).

        Args:
            name: Name of the dependency to resolve

        Returns:
            The resolved dependency instance

        Raises:
            DependencyResolutionError: If dependency not found
            CircularDependencyError: If circular dependency detected
        """
        if name not in self._registrations:
            raise DependencyResolutionError(f"Dependency not found: {name}")

        registration = self._registrations[name]

        # Return existing singleton instance
        if registration.lifecycle == DependencyLifecycle.SINGLETON and registration.instance is not None:
            return registration.instance

        # Check for circular dependency
        if name in self._resolving:
            raise CircularDependencyError(
                f"Circular dependency detected while resolving: {name}. "
                f"Resolution stack: {self._resolving}"
            )

        self._resolving.add(name)

        try:
            # Resolve dependencies first
            deps = {}
            for dep_name in registration.dependencies:
                deps[dep_name] = await self.resolve_async(dep_name)

            # Create instance
            if registration.is_async:
                instance = await registration.factory(**deps) if deps else await registration.factory()
            else:
                instance = registration.factory(**deps) if deps else registration.factory()

            # Store singleton instance
            if registration.lifecycle == DependencyLifecycle.SINGLETON:
                registration.instance = instance

            return instance

        finally:
            self._resolving.discard(name)

    def is_registered(self, name: str) -> bool:
        """Check if a dependency is registered."""
        return name in self._registrations

    def get_registration(self, name: str) -> DependencyRegistration | None:
        """Get registration info for a dependency."""
        return self._registrations.get(name)

    def unregister(self, name: str) -> bool:
        """
        Unregister a dependency.

        Args:
            name: Name of the dependency to remove

        Returns:
            True if removed, False if not found
        """
        if name in self._registrations:
            del self._registrations[name]
            logger.debug(f"Unregistered: {name}")
            return True
        return False

    def clear(self) -> None:
        """Clear all registrations."""
        self._registrations.clear()
        self._resolving.clear()
        logger.debug("Container cleared")

    def add_shutdown_handler(self, handler: Callable[[], Any]) -> None:
        """
        Add a handler to call during shutdown.

        Args:
            handler: Function to call (can be async)
        """
        self._shutdown_handlers.append(handler)

    async def initialize_all(self) -> dict[str, Any]:
        """
        Initialize all singleton dependencies.

        Returns:
            Dict of name -> instance for all singletons
        """
        instances = {}
        for name, reg in self._registrations.items():
            if reg.lifecycle == DependencyLifecycle.SINGLETON:
                instances[name] = await self.resolve_async(name)
        self._initialized = True
        logger.info(f"Initialized {len(instances)} singleton dependencies")
        return instances

    async def shutdown(self) -> None:
        """
        Shutdown container and call all shutdown handlers.

        Calls handlers in reverse order of registration.
        """
        logger.info("Shutting down DI container")

        for handler in reversed(self._shutdown_handlers):
            try:
                result = handler()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Shutdown handler error: {e}")

        self._shutdown_handlers.clear()
        self._initialized = False

    def get_dependency_graph(self) -> dict[str, list[str]]:
        """
        Get the dependency graph.

        Returns:
            Dict mapping each dependency to its dependencies
        """
        return {
            name: reg.dependencies
            for name, reg in self._registrations.items()
        }

    def get_status(self) -> dict[str, Any]:
        """Get container status for debugging."""
        return {
            "initialized": self._initialized,
            "registrations": len(self._registrations),
            "singletons_instantiated": sum(
                1 for reg in self._registrations.values()
                if reg.lifecycle == DependencyLifecycle.SINGLETON and reg.instance is not None
            ),
            "dependencies": {
                name: {
                    "lifecycle": reg.lifecycle.value,
                    "has_instance": reg.instance is not None,
                    "dependencies": reg.dependencies,
                }
                for name, reg in self._registrations.items()
            },
        }


# ============================================================================
# Pre-configured Container for Trading System
# ============================================================================

def create_trading_container() -> DIContainer:
    """
    Create a pre-configured DI container for the trading system.

    This sets up standard dependencies without instantiating them.
    Call initialize_all() to create instances.

    Returns:
        Configured DIContainer
    """
    container = DIContainer()

    # Note: Actual registration happens in main.py after config is loaded
    # This function provides a template/factory

    logger.info("Created trading system DI container")
    return container


class ServiceProvider:
    """
    Service provider pattern for accessing the container.

    This class provides a static interface to the DI container,
    useful for legacy code that cannot use constructor injection.

    Usage:
        ServiceProvider.set_container(container)
        event_bus = ServiceProvider.get("event_bus")
    """

    _container: DIContainer | None = None

    @classmethod
    def set_container(cls, container: DIContainer) -> None:
        """Set the global container."""
        cls._container = container

    @classmethod
    def get(cls, name: str) -> Any:
        """Get a dependency from the container."""
        if cls._container is None:
            raise RuntimeError("ServiceProvider not initialized. Call set_container() first.")
        return cls._container.resolve(name)

    @classmethod
    async def get_async(cls, name: str) -> Any:
        """Get a dependency from the container (async)."""
        if cls._container is None:
            raise RuntimeError("ServiceProvider not initialized. Call set_container() first.")
        return await cls._container.resolve_async(name)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a dependency is registered."""
        if cls._container is None:
            return False
        return cls._container.is_registered(name)

    @classmethod
    def get_container(cls) -> DIContainer | None:
        """Get the current container."""
        return cls._container

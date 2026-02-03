# dependency_injection

**Path**: `C:\Users\Alexa\ai-trading-firm\core\dependency_injection.py`

## Overview

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

## Classes

### DependencyLifecycle

**Inherits from**: Enum

Lifecycle of a dependency.

### DependencyRegistration

Registration info for a dependency.

### DependencyResolutionError

**Inherits from**: Exception

Raised when a dependency cannot be resolved.

### CircularDependencyError

**Inherits from**: Exception

Raised when circular dependencies are detected.

### DIContainer

Simple Dependency Injection Container.

Features:
- Singleton and factory registration
- Dependency chain resolution
- Async initialization support
- Lifecycle management

Thread Safety:
- Not thread-safe by design (async single-threaded model)
- Use asyncio.Lock if concurrent access needed

#### Methods

##### `def __init__(self)`

##### `def register_singleton(self, name: str, factory: Callable[Ellipsis, T], dependencies: , is_async: bool) -> DIContainer`

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

##### `def register_factory(self, name: str, factory: Callable[Ellipsis, T], dependencies: , is_async: bool) -> DIContainer`

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

##### `def register_instance(self, name: str, instance: Any) -> DIContainer`

Register an existing instance as a singleton.

Args:
    name: Unique name for the dependency
    instance: The instance to register

Returns:
    Self for chaining

##### `def resolve(self, name: str) -> Any`

Resolve a dependency by name (sync).

Args:
    name: Name of the dependency to resolve

Returns:
    The resolved dependency instance

Raises:
    DependencyResolutionError: If dependency not found
    CircularDependencyError: If circular dependency detected

##### `async def resolve_async(self, name: str) -> Any`

Resolve a dependency by name (async).

Args:
    name: Name of the dependency to resolve

Returns:
    The resolved dependency instance

Raises:
    DependencyResolutionError: If dependency not found
    CircularDependencyError: If circular dependency detected

##### `def is_registered(self, name: str) -> bool`

Check if a dependency is registered.

##### `def get_registration(self, name: str)`

Get registration info for a dependency.

##### `def unregister(self, name: str) -> bool`

Unregister a dependency.

Args:
    name: Name of the dependency to remove

Returns:
    True if removed, False if not found

##### `def clear(self) -> None`

Clear all registrations.

##### `def add_shutdown_handler(self, handler: Callable[, Any]) -> None`

Add a handler to call during shutdown.

Args:
    handler: Function to call (can be async)

##### `async def initialize_all(self) -> dict[str, Any]`

Initialize all singleton dependencies.

Returns:
    Dict of name -> instance for all singletons

##### `async def shutdown(self) -> None`

Shutdown container and call all shutdown handlers.

Calls handlers in reverse order of registration.

##### `def get_dependency_graph(self) -> dict[str, list[str]]`

Get the dependency graph.

Returns:
    Dict mapping each dependency to its dependencies

##### `def get_status(self) -> dict[str, Any]`

Get container status for debugging.

### ServiceProvider

Service provider pattern for accessing the container.

This class provides a static interface to the DI container,
useful for legacy code that cannot use constructor injection.

Usage:
    ServiceProvider.set_container(container)
    event_bus = ServiceProvider.get("event_bus")

#### Methods

##### `def set_container(cls, container: DIContainer) -> None`

Set the global container.

##### `def get(cls, name: str) -> Any`

Get a dependency from the container.

##### `async def get_async(cls, name: str) -> Any`

Get a dependency from the container (async).

##### `def is_registered(cls, name: str) -> bool`

Check if a dependency is registered.

##### `def get_container(cls)`

Get the current container.

## Functions

### `def create_trading_container() -> DIContainer`

Create a pre-configured DI container for the trading system.

This sets up standard dependencies without instantiating them.
Call initialize_all() to create instances.

Returns:
    Configured DIContainer

## Constants

- `T`

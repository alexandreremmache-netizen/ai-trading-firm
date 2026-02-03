# event_persistence

**Path**: `C:\Users\Alexa\ai-trading-firm\core\event_persistence.py`

## Overview

Event Persistence Layer
=======================

Persists unprocessed events to disk to prevent data loss on restart.
Addresses issue #S4: Event bus doesn't persist unprocessed events.

Features:
- SQLite-based persistence for durability
- Write-ahead logging for crash safety
- Automatic cleanup of processed events
- Recovery of unprocessed events on startup
- Support for event priority ordering

## Classes

### EventStatus

**Inherits from**: Enum

Status of persisted event.

### PersistedEvent

Wrapper for persisted event with metadata.

### PersistenceConfig

Configuration for event persistence.

### EventPersistence

Event persistence layer using SQLite.

Provides durability for events to prevent loss on system restart.
Uses WAL mode for better concurrent performance.

#### Methods

##### `def __init__(self, config: )`

##### `def initialize(self) -> None`

Initialize the database and create tables if needed.

##### `def persist_event(self, event: Event, priority: bool) -> bool`

Persist an event to the database.

Args:
    event: Event to persist
    priority: If True, mark as high priority

Returns:
    True if persisted successfully, False otherwise

##### `async def persist_event_async(self, event: Event, priority: bool) -> bool`

Async version of persist_event using run_in_executor.

##### `def mark_processing(self, event_id: str) -> bool`

Mark an event as being processed.

##### `def mark_completed(self, event_id: str) -> bool`

Mark an event as completed (processed successfully).

##### `async def mark_completed_async(self, event_id: str) -> bool`

Async version of mark_completed.

##### `def mark_failed(self, event_id: str, error_message: str) -> bool`

Mark an event as failed with error message.

##### `def get_pending_events(self, limit: int, include_failed: bool) -> list[PersistedEvent]`

Get pending events for processing.

Args:
    limit: Maximum number of events to return
    include_failed: Include failed events that can be retried

Returns:
    List of persisted events ordered by priority and creation time

##### `def reconstruct_event(self, persisted: PersistedEvent)`

Reconstruct an Event object from persisted data.

Args:
    persisted: Persisted event data

Returns:
    Reconstructed Event or None if reconstruction fails

##### `def cleanup_completed_events(self, hours: ) -> int`

Clean up completed events older than the specified hours.

Args:
    hours: Hours to keep completed events (uses config default if None)

Returns:
    Number of events deleted

##### `def get_statistics(self) -> dict[str, Any]`

Get persistence statistics.

##### `def reset_stale_processing(self, timeout_minutes: int) -> int`

Reset events stuck in 'processing' status back to 'pending'.

This handles cases where a process crashed while processing an event.

Args:
    timeout_minutes: Minutes after which processing is considered stale

Returns:
    Number of events reset

##### `def close(self) -> None`

Close the database connection.

## Functions

### `def get_event_persistence(config: ) -> EventPersistence`

Get or create the global event persistence instance.

### `def reset_event_persistence() -> None`

Reset the global event persistence instance (for testing).

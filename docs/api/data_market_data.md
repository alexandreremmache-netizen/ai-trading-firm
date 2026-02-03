# market_data

**Path**: `C:\Users\Alexa\ai-trading-firm\data\market_data.py`

## Overview

Market Data Manager
===================

Centralizes market data handling from Interactive Brokers.
Distributes data to all agents via the event bus.

## Classes

### SymbolConfig

Configuration for a symbol subscription.

### MarketDataManager

Market Data Manager.

Responsibilities:
- Subscribe to market data from Interactive Brokers
- Transform raw data into MarketDataEvents
- Publish events to the event bus
- Handle reconnection and data quality

This is the ONLY source of market data in the system.

#### Methods

##### `def __init__(self, broker: IBBroker, event_bus: EventBus, symbols: list[SymbolConfig])`

##### `async def start(self) -> None`

Start market data subscriptions.

##### `async def stop(self) -> None`

Stop market data subscriptions.

##### `async def generate_simulated_data(self) -> None`

Generate simulated market data for testing.

Used when IB connection is not available (development/testing).

##### `def get_last_price(self, symbol: str)`

Get last known price for a symbol.

##### `def get_all_prices(self) -> dict[str, float]`

Get all last known prices.

##### `def update_count(self) -> int`

Total market data updates processed.

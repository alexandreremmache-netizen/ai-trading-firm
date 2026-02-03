# scenario_analysis

**Path**: `C:\Users\Alexa\ai-trading-firm\core\scenario_analysis.py`

## Overview

Scenario Analysis Module
========================

Worst-case scenario reporting (Issue #R17).
Historical stress event playback (Issue #R18).
What-if analysis support (Issue #P16).

Features:
- Historical crisis replay
- Custom scenario definition
- What-if position changes
- Worst-case scenario identification

## Classes

### HistoricalEvent

**Inherits from**: str, Enum

Notable historical market events for stress testing.

### AssetShock

Price/factor shock for an asset class.

### ScenarioDefinition

Definition of a market scenario.

#### Methods

##### `def to_dict(self) -> dict`

### PositionImpact

Impact of scenario on a single position.

#### Methods

##### `def to_dict(self) -> dict`

### ScenarioResult

Result of running a scenario.

#### Methods

##### `def to_dict(self) -> dict`

### WhatIfResult

Result of what-if analysis (#P16).

#### Methods

##### `def to_dict(self) -> dict`

### HistoricalEventLibrary

Library of historical market events for stress testing (#R18).

Contains calibrated shocks based on actual market moves.

#### Methods

##### `def get_event(cls, event: HistoricalEvent) -> ScenarioDefinition`

Get scenario definition for a historical event.

##### `def get_all_events(cls) -> list[ScenarioDefinition]`

Get all historical event scenarios.

##### `def get_events_by_severity(cls, min_equity_drop: float) -> list[ScenarioDefinition]`

Get events with equity drop worse than threshold.

### ScenarioEngine

Runs scenario analysis on portfolios (#R17, #R18).

Supports both historical replay and custom scenarios.

#### Methods

##### `def __init__(self, margin_requirement_pct: float)`

##### `def update_position(self, symbol: str, quantity: int, price: float, asset_class: str) -> None`

Update position for scenario analysis.

##### `def add_custom_scenario(self, scenario: ScenarioDefinition) -> None`

Add a custom scenario.

##### `def run_scenario(self, scenario: ScenarioDefinition, account_equity: ) -> ScenarioResult`

Run a scenario on the current portfolio.

Args:
    scenario: Scenario to run
    account_equity: Account equity for margin calculations

Returns:
    ScenarioResult with detailed impact

##### `def run_historical_event(self, event: HistoricalEvent, account_equity: ) -> ScenarioResult`

Run a historical event scenario.

##### `def run_all_historical_events(self, account_equity: ) -> list[ScenarioResult]`

Run all historical event scenarios.

##### `def find_worst_case(self, account_equity: ) -> ScenarioResult`

Find the worst-case scenario from all available (#R17).

Runs all scenarios and returns the one with worst P&L.

##### `def generate_worst_case_report(self, account_equity: float, include_top_n: int) -> dict`

Generate comprehensive worst-case report (#R17).

Returns top N worst scenarios with detailed analysis.

### WhatIfAnalyzer

What-if analysis for position changes (#P16).

Allows testing hypothetical portfolio changes.

#### Methods

##### `def __init__(self, scenario_engine: ScenarioEngine, margin_calc: , var_calc: )`

##### `def analyze_position_change(self, symbol: str, quantity_change: int, price: , asset_class: str) -> WhatIfResult`

Analyze impact of changing a single position.

Args:
    symbol: Symbol to change
    quantity_change: Change in quantity (positive = buy, negative = sell)
    price: Current price (uses existing if not provided)
    asset_class: Asset class for new positions

Returns:
    WhatIfResult with before/after comparison

##### `def analyze_multiple_changes(self, changes: dict[str, dict]) -> WhatIfResult`

Analyze impact of multiple position changes.

Args:
    changes: Dict of symbol -> {quantity_change, price, asset_class}

Returns:
    WhatIfResult with aggregate impact

##### `def find_optimal_hedge(self, target_symbol: str, hedge_candidates: list[str], target_var_reduction_pct: float) -> list[WhatIfResult]`

Find optimal hedge for a position.

Tests various hedge ratios and returns best options.

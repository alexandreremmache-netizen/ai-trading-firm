# stress_tester

**Path**: `C:\Users\Alexa\ai-trading-firm\core\stress_tester.py`

## Overview

Stress Testing Module
=====================

Comprehensive stress testing for portfolio risk assessment.
Implements predefined and custom scenarios for regulatory
compliance and risk management.

## Classes

### ScenarioType

**Inherits from**: Enum

Type of stress scenario.

### StressScenario

Definition of a stress scenario.

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary.

### StressTestResult

Result of a stress test.

#### Methods

##### `def to_dict(self) -> dict[str, Any]`

Convert to dictionary.

### StressTester

Portfolio stress testing system.

Features:
- Predefined crisis scenarios
- Custom scenario creation
- Portfolio impact analysis
- Limit breach detection
- Regulatory reporting support

#### Methods

##### `def __init__(self, config: , max_scenario_loss_pct: float, margin_buffer_pct: float)`

Initialize stress tester.

Args:
    config: Configuration with:
        - max_scenario_loss_pct: Maximum acceptable loss (default: 25%)
        - margin_buffer_pct: Margin buffer requirement (default: 20%)
        - run_on_startup: Run tests at startup (default: False)
        - use_predefined: Load predefined scenarios (default: True)
        - custom_scenarios: List of custom scenario definitions (#R5)
    max_scenario_loss_pct: Fallback max loss if not in config
    margin_buffer_pct: Fallback margin buffer if not in config

##### `def load_scenarios_from_file(self, filepath: str) -> int`

Load additional scenarios from a YAML or JSON file (#R5).

Args:
    filepath: Path to scenarios file

Returns:
    Number of scenarios loaded

##### `def register_callback(self, callback: Callable[, None]) -> None`

Register callback for stress test results.

##### `def add_custom_scenario(self, scenario: StressScenario) -> None`

Add a custom stress scenario.

##### `def run_scenario(self, scenario_id: str, positions: dict[str, float], portfolio_value: float, prices: dict[str, float], contract_specs: ) -> StressTestResult`

Run a single stress scenario.

Args:
    scenario_id: ID of scenario to run
    positions: Current positions (symbol -> value or contracts)
    portfolio_value: Total portfolio value
    prices: Current prices by symbol
    contract_specs: Optional contract specifications

Returns:
    StressTestResult with impact analysis

##### `def run_all_scenarios(self, positions: dict[str, float], portfolio_value: float, prices: dict[str, float], contract_specs: , severity_filter: ) -> list[StressTestResult]`

Run all stress scenarios.

Args:
    positions: Current positions
    portfolio_value: Total portfolio value
    prices: Current prices
    contract_specs: Optional contract specifications
    severity_filter: Only run scenarios with this severity or higher

Returns:
    List of StressTestResult

##### `def get_worst_case_scenario(self, positions: dict[str, float], portfolio_value: float, prices: dict[str, float])`

Run all scenarios and return the worst case.

Args:
    positions: Current positions
    portfolio_value: Portfolio value
    prices: Current prices

Returns:
    StressTestResult with worst impact

##### `def get_scenario(self, scenario_id: str)`

Get scenario by ID.

##### `def get_all_scenarios(self) -> list[StressScenario]`

Get all available scenarios.

##### `def get_scenarios_by_type(self, scenario_type: ScenarioType) -> list[StressScenario]`

Get scenarios of a specific type.

##### `def get_recent_results(self, scenario_id: , limit: int) -> list[StressTestResult]`

Get recent stress test results.

Args:
    scenario_id: Filter by scenario ID
    limit: Maximum results to return

Returns:
    List of recent results

##### `def get_failed_scenarios(self, positions: dict[str, float], portfolio_value: float, prices: dict[str, float]) -> list[StressTestResult]`

Run all scenarios and return those that fail limits.

Args:
    positions: Current positions
    portfolio_value: Portfolio value
    prices: Current prices

Returns:
    List of failed StressTestResult

##### `def generate_report(self, results: list[StressTestResult]) -> dict[str, Any]`

Generate stress test report.

Args:
    results: List of stress test results

Returns:
    Report dictionary

##### `def get_status(self) -> dict[str, Any]`

Get tester status for monitoring.

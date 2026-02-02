"""
Stress Testing Module
=====================

Comprehensive stress testing for portfolio risk assessment.
Implements predefined and custom scenarios for regulatory
compliance and risk management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np


logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Type of stress scenario."""
    MARKET_CRASH = "market_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    FLASH_CRASH = "flash_crash"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    CURRENCY_CRISIS = "currency_crisis"
    SECTOR_CRASH = "sector_crash"
    CUSTOM = "custom"


@dataclass
class StressScenario:
    """Definition of a stress scenario."""
    scenario_id: str
    name: str
    scenario_type: ScenarioType
    description: str

    # Market shocks (by asset class or symbol)
    price_shocks: dict[str, float] = field(default_factory=dict)  # Symbol/class -> % change

    # Volatility adjustments
    volatility_multiplier: float = 1.0

    # Correlation adjustments
    correlation_override: float | None = None  # Override all correlations

    # Liquidity adjustments
    liquidity_haircut: float = 0.0  # % reduction in liquidity

    # Time horizon
    horizon_days: int = 1

    # Severity (1-5)
    severity: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "scenario_type": self.scenario_type.value,
            "description": self.description,
            "price_shocks": self.price_shocks,
            "volatility_multiplier": self.volatility_multiplier,
            "correlation_override": self.correlation_override,
            "liquidity_haircut": self.liquidity_haircut,
            "horizon_days": self.horizon_days,
            "severity": self.severity,
        }


@dataclass
class StressTestResult:
    """Result of a stress test."""
    scenario: StressScenario
    timestamp: datetime
    portfolio_value_before: float
    portfolio_value_after: float
    pnl_impact: float
    pnl_impact_pct: float
    positions_impacted: dict[str, float]  # Symbol -> P&L impact
    margin_impact: float
    liquidity_impact: float
    passes_limit: bool
    limit_breaches: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_id": self.scenario.scenario_id,
            "scenario_name": self.scenario.name,
            "timestamp": self.timestamp.isoformat(),
            "portfolio_value_before": self.portfolio_value_before,
            "portfolio_value_after": self.portfolio_value_after,
            "pnl_impact": self.pnl_impact,
            "pnl_impact_pct": self.pnl_impact_pct,
            "positions_impacted": self.positions_impacted,
            "margin_impact": self.margin_impact,
            "liquidity_impact": self.liquidity_impact,
            "passes_limit": self.passes_limit,
            "limit_breaches": self.limit_breaches,
            "details": self.details,
        }


# =============================================================================
# PREDEFINED SCENARIOS
# =============================================================================

PREDEFINED_SCENARIOS: dict[str, StressScenario] = {
    "market_crash_15": StressScenario(
        scenario_id="STRESS-001",
        name="Market Crash (-15%)",
        scenario_type=ScenarioType.MARKET_CRASH,
        description="Broad market decline of 15% across equities and equity futures",
        price_shocks={
            # Equities
            "AAPL": -0.15, "MSFT": -0.15, "GOOGL": -0.15, "AMZN": -0.15,
            "META": -0.15, "NVDA": -0.18, "TSLA": -0.20,
            # ETFs
            "SPY": -0.15, "QQQ": -0.17, "IWM": -0.18,
            # Index Futures
            "ES": -0.15, "NQ": -0.17, "YM": -0.15, "RTY": -0.18,
            # Safe havens rally
            "GC": 0.05, "ZB": 0.03, "ZN": 0.02,
        },
        volatility_multiplier=2.5,
        correlation_override=0.85,
        liquidity_haircut=0.30,
        severity=5,
    ),

    "market_crash_10": StressScenario(
        scenario_id="STRESS-002",
        name="Market Correction (-10%)",
        scenario_type=ScenarioType.MARKET_CRASH,
        description="Moderate market correction of 10%",
        price_shocks={
            "AAPL": -0.10, "MSFT": -0.10, "GOOGL": -0.10, "AMZN": -0.10,
            "META": -0.10, "NVDA": -0.12, "TSLA": -0.15,
            "SPY": -0.10, "QQQ": -0.12, "IWM": -0.12,
            "ES": -0.10, "NQ": -0.12, "YM": -0.10, "RTY": -0.12,
            "GC": 0.03, "ZB": 0.02,
        },
        volatility_multiplier=1.8,
        correlation_override=0.70,
        liquidity_haircut=0.15,
        severity=4,
    ),

    "vix_spike_40": StressScenario(
        scenario_id="STRESS-003",
        name="Volatility Spike (VIX 40+)",
        scenario_type=ScenarioType.VOLATILITY_SPIKE,
        description="Sharp increase in implied volatility, VIX reaches 40+",
        price_shocks={
            # Moderate declines
            "SPY": -0.05, "QQQ": -0.06,
            "ES": -0.05, "NQ": -0.06,
            # Vol-sensitive
            "NVDA": -0.10, "TSLA": -0.12,
        },
        volatility_multiplier=3.0,
        correlation_override=0.75,
        liquidity_haircut=0.25,
        severity=4,
    ),

    "flash_crash": StressScenario(
        scenario_id="STRESS-004",
        name="Flash Crash",
        scenario_type=ScenarioType.FLASH_CRASH,
        description="Rapid intraday decline followed by partial recovery, max drawdown -8%",
        price_shocks={
            # Sharp but partial recovery
            "SPY": -0.05, "QQQ": -0.06, "IWM": -0.08,
            "ES": -0.05, "NQ": -0.06, "RTY": -0.08,
            "AAPL": -0.06, "MSFT": -0.05, "NVDA": -0.10,
        },
        volatility_multiplier=4.0,
        correlation_override=0.90,
        liquidity_haircut=0.50,  # Severe liquidity impact
        horizon_days=1,
        severity=5,
    ),

    "correlation_breakdown": StressScenario(
        scenario_id="STRESS-005",
        name="Correlation Breakdown",
        scenario_type=ScenarioType.CORRELATION_BREAKDOWN,
        description="Historical correlations break down, paired trades diverge",
        price_shocks={
            # Pairs diverge
            "AAPL": 0.05, "MSFT": -0.05,
            "GOOGL": 0.03, "META": -0.07,
            "ES": 0.02, "NQ": -0.03,
            "GC": -0.05, "SI": 0.08,
            # Spread trades fail
            "CL": -0.10, "RB": 0.05,
        },
        volatility_multiplier=1.5,
        correlation_override=None,  # Let correlations vary
        liquidity_haircut=0.10,
        severity=3,
    ),

    "liquidity_crisis": StressScenario(
        scenario_id="STRESS-006",
        name="Liquidity Crisis",
        scenario_type=ScenarioType.LIQUIDITY_CRISIS,
        description="Severe reduction in market liquidity, wide spreads",
        price_shocks={
            # Moderate declines
            "SPY": -0.03, "QQQ": -0.04,
            # Small caps hit harder
            "IWM": -0.08, "RTY": -0.08,
            # Commodities
            "CL": -0.05, "NG": -0.08,
        },
        volatility_multiplier=2.0,
        correlation_override=0.60,
        liquidity_haircut=0.60,  # Severe liquidity reduction
        severity=4,
    ),

    "rates_shock_up": StressScenario(
        scenario_id="STRESS-007",
        name="Interest Rate Shock (+100bp)",
        scenario_type=ScenarioType.INTEREST_RATE_SHOCK,
        description="Sudden 100bp increase in interest rates",
        price_shocks={
            # Bonds decline
            "ZB": -0.08, "ZN": -0.05, "ZF": -0.03,
            # Growth stocks decline
            "NVDA": -0.08, "TSLA": -0.10,
            "QQQ": -0.05, "NQ": -0.05,
            # Banks benefit
            "SPY": -0.02,
            # Gold declines
            "GC": -0.03,
        },
        volatility_multiplier=1.5,
        liquidity_haircut=0.15,
        severity=3,
    ),

    "energy_crisis": StressScenario(
        scenario_id="STRESS-008",
        name="Energy Crisis",
        scenario_type=ScenarioType.SECTOR_CRASH,
        description="Sharp spike in energy prices",
        price_shocks={
            # Energy spikes
            "CL": 0.25, "NG": 0.40, "RB": 0.30, "HO": 0.30,
            "USO": 0.20,
            # Broader market impact
            "SPY": -0.03, "ES": -0.03,
            # Airlines/transport (hypothetical impact)
            "IWM": -0.05, "RTY": -0.05,
        },
        volatility_multiplier=2.0,
        liquidity_haircut=0.20,
        severity=4,
    ),

    "tech_crash": StressScenario(
        scenario_id="STRESS-009",
        name="Tech Sector Crash",
        scenario_type=ScenarioType.SECTOR_CRASH,
        description="Technology sector specific decline of 20%",
        price_shocks={
            "AAPL": -0.18, "MSFT": -0.18, "GOOGL": -0.20, "AMZN": -0.18,
            "META": -0.22, "NVDA": -0.25, "TSLA": -0.25,
            "QQQ": -0.20, "NQ": -0.20,
            "SPY": -0.08, "ES": -0.08,
            "IWM": -0.05,
        },
        volatility_multiplier=2.5,
        correlation_override=0.80,
        liquidity_haircut=0.25,
        severity=5,
    ),

    "commodity_collapse": StressScenario(
        scenario_id="STRESS-010",
        name="Commodity Collapse",
        scenario_type=ScenarioType.SECTOR_CRASH,
        description="Broad commodity price decline",
        price_shocks={
            # Energy
            "CL": -0.20, "NG": -0.25, "RB": -0.20, "HO": -0.20,
            # Metals
            "GC": -0.10, "SI": -0.15, "HG": -0.20, "PL": -0.12,
            # Agriculture
            "ZC": -0.12, "ZW": -0.12, "ZS": -0.15, "ZM": -0.10, "ZL": -0.12,
        },
        volatility_multiplier=2.0,
        liquidity_haircut=0.30,
        severity=4,
    ),
}


class StressTester:
    """
    Portfolio stress testing system.

    Features:
    - Predefined crisis scenarios
    - Custom scenario creation
    - Portfolio impact analysis
    - Limit breach detection
    - Regulatory reporting support
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        max_scenario_loss_pct: float = 25.0,
        margin_buffer_pct: float = 20.0,
    ):
        """
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
        """
        self._config = config or {}
        self._max_scenario_loss_pct = self._config.get("max_scenario_loss_pct", max_scenario_loss_pct)
        self._margin_buffer_pct = self._config.get("margin_buffer_pct", margin_buffer_pct)

        # Scenarios - optionally load predefined (#R5)
        self._scenarios: dict[str, StressScenario] = {}
        if self._config.get("use_predefined", True):
            self._scenarios = dict(PREDEFINED_SCENARIOS)
            logger.info(f"Loaded {len(PREDEFINED_SCENARIOS)} predefined scenarios")

        # Load custom scenarios from config (#R5)
        custom_scenarios = self._config.get("custom_scenarios", [])
        for scenario_def in custom_scenarios:
            try:
                scenario = self._parse_scenario_config(scenario_def)
                self._scenarios[scenario.scenario_id] = scenario
                logger.info(f"Loaded custom scenario from config: {scenario.name}")
            except Exception as e:
                logger.error(f"Failed to load custom scenario: {e}")

        # Results history
        self._results_history: list[StressTestResult] = []
        self._last_full_run: datetime | None = None

        # Callbacks
        self._result_callbacks: list[Callable[[StressTestResult], None]] = []

        logger.info(f"StressTester initialized with {len(self._scenarios)} scenarios")

    def _parse_scenario_config(self, config: dict[str, Any]) -> StressScenario:
        """
        Parse a scenario from config dictionary (#R5).

        Args:
            config: Scenario configuration dict with:
                - id: Unique scenario ID
                - name: Display name
                - type: Scenario type (market_crash, volatility_spike, etc.)
                - description: Description
                - price_shocks: Dict of symbol -> percentage shock
                - volatility_multiplier: Vol multiplier (default: 1.0)
                - correlation_override: Optional correlation override
                - liquidity_haircut: Liquidity reduction % (default: 0)
                - horizon_days: Time horizon (default: 1)
                - severity: Severity 1-5 (default: 3)

        Returns:
            StressScenario instance
        """
        # Map string type to enum
        type_str = config.get("type", "custom").lower()
        scenario_type = {
            "market_crash": ScenarioType.MARKET_CRASH,
            "volatility_spike": ScenarioType.VOLATILITY_SPIKE,
            "flash_crash": ScenarioType.FLASH_CRASH,
            "correlation_breakdown": ScenarioType.CORRELATION_BREAKDOWN,
            "liquidity_crisis": ScenarioType.LIQUIDITY_CRISIS,
            "interest_rate_shock": ScenarioType.INTEREST_RATE_SHOCK,
            "currency_crisis": ScenarioType.CURRENCY_CRISIS,
            "sector_crash": ScenarioType.SECTOR_CRASH,
            "custom": ScenarioType.CUSTOM,
        }.get(type_str, ScenarioType.CUSTOM)

        return StressScenario(
            scenario_id=config.get("id", f"CUSTOM-{len(self._scenarios)+1:03d}"),
            name=config.get("name", "Custom Scenario"),
            scenario_type=scenario_type,
            description=config.get("description", ""),
            price_shocks=config.get("price_shocks", {}),
            volatility_multiplier=config.get("volatility_multiplier", 1.0),
            correlation_override=config.get("correlation_override"),
            liquidity_haircut=config.get("liquidity_haircut", 0.0),
            horizon_days=config.get("horizon_days", 1),
            severity=config.get("severity", 3),
        )

    def load_scenarios_from_file(self, filepath: str) -> int:
        """
        Load additional scenarios from a YAML or JSON file (#R5).

        Args:
            filepath: Path to scenarios file

        Returns:
            Number of scenarios loaded
        """
        import json
        import yaml
        from pathlib import Path

        path = Path(filepath)
        if not path.exists():
            logger.error(f"Scenarios file not found: {filepath}")
            return 0

        try:
            with open(path, 'r') as f:
                if path.suffix in ('.yaml', '.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            scenarios = data.get("scenarios", data if isinstance(data, list) else [])
            loaded = 0

            for scenario_def in scenarios:
                try:
                    scenario = self._parse_scenario_config(scenario_def)
                    self._scenarios[scenario.scenario_id] = scenario
                    loaded += 1
                    logger.info(f"Loaded scenario from file: {scenario.name}")
                except Exception as e:
                    logger.error(f"Failed to parse scenario: {e}")

            logger.info(f"Loaded {loaded} scenarios from {filepath}")
            return loaded

        except Exception as e:
            logger.error(f"Failed to load scenarios from {filepath}: {e}")
            return 0

    def register_callback(self, callback: Callable[[StressTestResult], None]) -> None:
        """Register callback for stress test results."""
        self._result_callbacks.append(callback)

    def add_custom_scenario(self, scenario: StressScenario) -> None:
        """Add a custom stress scenario."""
        self._scenarios[scenario.scenario_id] = scenario
        logger.info(f"Added custom scenario: {scenario.name}")

    def run_scenario(
        self,
        scenario_id: str,
        positions: dict[str, float],
        portfolio_value: float,
        prices: dict[str, float],
        contract_specs: dict[str, Any] | None = None
    ) -> StressTestResult:
        """
        Run a single stress scenario.

        Args:
            scenario_id: ID of scenario to run
            positions: Current positions (symbol -> value or contracts)
            portfolio_value: Total portfolio value
            prices: Current prices by symbol
            contract_specs: Optional contract specifications

        Returns:
            StressTestResult with impact analysis
        """
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Unknown scenario: {scenario_id}")

        return self._execute_scenario(scenario, positions, portfolio_value, prices, contract_specs)

    def run_all_scenarios(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        prices: dict[str, float],
        contract_specs: dict[str, Any] | None = None,
        severity_filter: int | None = None
    ) -> list[StressTestResult]:
        """
        Run all stress scenarios.

        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            prices: Current prices
            contract_specs: Optional contract specifications
            severity_filter: Only run scenarios with this severity or higher

        Returns:
            List of StressTestResult
        """
        results = []

        for scenario_id, scenario in self._scenarios.items():
            if severity_filter and scenario.severity < severity_filter:
                continue

            try:
                result = self._execute_scenario(
                    scenario, positions, portfolio_value, prices, contract_specs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error running scenario {scenario_id}: {e}")

        self._last_full_run = datetime.now(timezone.utc)

        return results

    def _execute_scenario(
        self,
        scenario: StressScenario,
        positions: dict[str, float],
        portfolio_value: float,
        prices: dict[str, float],
        contract_specs: dict[str, Any] | None
    ) -> StressTestResult:
        """Execute a stress scenario and calculate impact."""
        timestamp = datetime.now(timezone.utc)

        # Calculate P&L impact for each position
        position_impacts = {}
        total_pnl_impact = 0.0

        for symbol, position_value in positions.items():
            if position_value == 0:
                continue

            # Get price shock for this symbol
            shock = scenario.price_shocks.get(symbol, 0.0)

            # Also check for asset class shocks
            if shock == 0 and contract_specs:
                spec = contract_specs.get(symbol)
                if spec:
                    asset_class = getattr(spec, "asset_class", None)
                    if asset_class:
                        shock = scenario.price_shocks.get(asset_class.value, 0.0)

            # Calculate P&L impact
            pnl_impact = position_value * shock
            position_impacts[symbol] = pnl_impact
            total_pnl_impact += pnl_impact

        # Calculate portfolio value after stress
        portfolio_value_after = portfolio_value + total_pnl_impact
        pnl_impact_pct = (total_pnl_impact / portfolio_value * 100) if portfolio_value > 0 else 0

        # Calculate margin impact (simplified)
        margin_impact = abs(total_pnl_impact) * (1 + scenario.liquidity_haircut)

        # Calculate liquidity impact
        liquidity_impact = portfolio_value * scenario.liquidity_haircut

        # Check limits
        limit_breaches = []
        if abs(pnl_impact_pct) > self._max_scenario_loss_pct:
            limit_breaches.append(f"max_loss_exceeded: {pnl_impact_pct:.1f}% > {self._max_scenario_loss_pct}%")

        if margin_impact > portfolio_value * (self._margin_buffer_pct / 100):
            limit_breaches.append(f"margin_buffer_breached: impact {margin_impact:.0f}")

        passes_limit = len(limit_breaches) == 0

        result = StressTestResult(
            scenario=scenario,
            timestamp=timestamp,
            portfolio_value_before=portfolio_value,
            portfolio_value_after=portfolio_value_after,
            pnl_impact=total_pnl_impact,
            pnl_impact_pct=pnl_impact_pct,
            positions_impacted=position_impacts,
            margin_impact=margin_impact,
            liquidity_impact=liquidity_impact,
            passes_limit=passes_limit,
            limit_breaches=limit_breaches,
            details={
                "volatility_multiplier": scenario.volatility_multiplier,
                "correlation_override": scenario.correlation_override,
                "worst_position": min(position_impacts.items(), key=lambda x: x[1])[0] if position_impacts else None,
                "worst_position_impact": min(position_impacts.values()) if position_impacts else 0,
            }
        )

        # Store in history
        self._results_history.append(result)
        if len(self._results_history) > 1000:
            self._results_history = self._results_history[-1000:]

        # Dispatch callbacks
        for callback in self._result_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Stress test callback error: {e}")

        log_level = logging.WARNING if not passes_limit else logging.INFO
        logger.log(
            log_level,
            f"Stress test '{scenario.name}': P&L={pnl_impact_pct:+.1f}%, "
            f"passes={passes_limit}, breaches={limit_breaches}"
        )

        return result

    def get_worst_case_scenario(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        prices: dict[str, float]
    ) -> StressTestResult | None:
        """
        Run all scenarios and return the worst case.

        Args:
            positions: Current positions
            portfolio_value: Portfolio value
            prices: Current prices

        Returns:
            StressTestResult with worst impact
        """
        results = self.run_all_scenarios(positions, portfolio_value, prices)

        if not results:
            return None

        # Find worst case by P&L impact
        worst = min(results, key=lambda r: r.pnl_impact)
        return worst

    def get_scenario(self, scenario_id: str) -> StressScenario | None:
        """Get scenario by ID."""
        return self._scenarios.get(scenario_id)

    def get_all_scenarios(self) -> list[StressScenario]:
        """Get all available scenarios."""
        return list(self._scenarios.values())

    def get_scenarios_by_type(self, scenario_type: ScenarioType) -> list[StressScenario]:
        """Get scenarios of a specific type."""
        return [s for s in self._scenarios.values() if s.scenario_type == scenario_type]

    def get_recent_results(
        self,
        scenario_id: str | None = None,
        limit: int = 100
    ) -> list[StressTestResult]:
        """
        Get recent stress test results.

        Args:
            scenario_id: Filter by scenario ID
            limit: Maximum results to return

        Returns:
            List of recent results
        """
        results = self._results_history

        if scenario_id:
            results = [r for r in results if r.scenario.scenario_id == scenario_id]

        return results[-limit:]

    def get_failed_scenarios(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        prices: dict[str, float]
    ) -> list[StressTestResult]:
        """
        Run all scenarios and return those that fail limits.

        Args:
            positions: Current positions
            portfolio_value: Portfolio value
            prices: Current prices

        Returns:
            List of failed StressTestResult
        """
        results = self.run_all_scenarios(positions, portfolio_value, prices)
        return [r for r in results if not r.passes_limit]

    def generate_report(
        self,
        results: list[StressTestResult]
    ) -> dict[str, Any]:
        """
        Generate stress test report.

        Args:
            results: List of stress test results

        Returns:
            Report dictionary
        """
        if not results:
            return {"error": "No results to report"}

        # Summary statistics
        total_scenarios = len(results)
        passed = sum(1 for r in results if r.passes_limit)
        failed = total_scenarios - passed

        # Worst case
        worst = min(results, key=lambda r: r.pnl_impact)

        # By scenario type
        by_type = {}
        for r in results:
            type_name = r.scenario.scenario_type.value
            if type_name not in by_type:
                by_type[type_name] = {"count": 0, "avg_impact": 0, "passed": 0}
            by_type[type_name]["count"] += 1
            by_type[type_name]["avg_impact"] += r.pnl_impact_pct
            if r.passes_limit:
                by_type[type_name]["passed"] += 1

        for type_name in by_type:
            by_type[type_name]["avg_impact"] /= by_type[type_name]["count"]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_scenarios": total_scenarios,
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / total_scenarios if total_scenarios > 0 else 0,
            },
            "worst_case": {
                "scenario": worst.scenario.name,
                "pnl_impact": worst.pnl_impact,
                "pnl_impact_pct": worst.pnl_impact_pct,
            },
            "by_type": by_type,
            "all_breaches": [
                {"scenario": r.scenario.name, "breaches": r.limit_breaches}
                for r in results if not r.passes_limit
            ],
        }

    def get_status(self) -> dict[str, Any]:
        """Get tester status for monitoring."""
        return {
            "scenarios_available": len(self._scenarios),
            "predefined_scenarios": len(PREDEFINED_SCENARIOS),
            "custom_scenarios": len(self._scenarios) - len(PREDEFINED_SCENARIOS),
            "max_scenario_loss_pct": self._max_scenario_loss_pct,
            "margin_buffer_pct": self._margin_buffer_pct,
            "results_in_history": len(self._results_history),
            "last_full_run": self._last_full_run.isoformat() if self._last_full_run else None,
        }

"""
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
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class HistoricalEvent(str, Enum):
    """Notable historical market events for stress testing."""
    BLACK_MONDAY_1987 = "black_monday_1987"
    ASIAN_CRISIS_1997 = "asian_crisis_1997"
    LTCM_1998 = "ltcm_1998"
    DOT_COM_2000 = "dot_com_2000"
    SEPT_11_2001 = "sept_11_2001"
    GFC_2008 = "gfc_2008"
    FLASH_CRASH_2010 = "flash_crash_2010"
    EURO_CRISIS_2011 = "euro_crisis_2011"
    CHINA_DEVAL_2015 = "china_deval_2015"
    VOLMAGEDDON_2018 = "volmageddon_2018"
    COVID_2020 = "covid_2020"
    MEME_STOCKS_2021 = "meme_stocks_2021"
    RATE_SHOCK_2022 = "rate_shock_2022"


@dataclass
class AssetShock:
    """Price/factor shock for an asset class."""
    asset_class: str
    price_change_pct: float
    volatility_multiplier: float = 1.0
    correlation_shift: float = 0.0
    duration_days: int = 1


@dataclass
class ScenarioDefinition:
    """Definition of a market scenario."""
    name: str
    description: str
    shocks: list[AssetShock]
    event_date: date | None = None
    duration_days: int = 1
    is_historical: bool = False

    # Factor changes
    interest_rate_change_bps: float = 0.0
    vix_level: float | None = None
    credit_spread_change_bps: float = 0.0

    # P2 Enhancement: Probability weighting
    probability: float = 1.0  # Scenario probability (for weighting)
    severity_rank: int = 0  # Relative severity (1 = worst)
    category: str = "custom"  # Category: 'historical', 'hypothetical', 'custom'

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'shocks': [
                {
                    'asset_class': s.asset_class,
                    'price_change_pct': s.price_change_pct,
                    'volatility_multiplier': s.volatility_multiplier,
                    'correlation_shift': s.correlation_shift,
                }
                for s in self.shocks
            ],
            'event_date': self.event_date.isoformat() if self.event_date else None,
            'duration_days': self.duration_days,
            'is_historical': self.is_historical,
            'interest_rate_change_bps': self.interest_rate_change_bps,
            'vix_level': self.vix_level,
            'credit_spread_change_bps': self.credit_spread_change_bps,
            'probability': self.probability,
            'severity_rank': self.severity_rank,
            'category': self.category,
        }


@dataclass
class PositionImpact:
    """Impact of scenario on a single position."""
    symbol: str
    quantity: int
    current_value: float
    stressed_value: float
    pnl: float
    pnl_pct: float
    contribution_to_total: float

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'current_value': self.current_value,
            'stressed_value': self.stressed_value,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'contribution_to_total': self.contribution_to_total,
        }


@dataclass
class ScenarioResult:
    """Result of running a scenario."""
    scenario_name: str
    run_timestamp: datetime

    # Portfolio impact
    portfolio_value_before: float
    portfolio_value_after: float
    total_pnl: float
    total_pnl_pct: float

    # Position details
    position_impacts: list[PositionImpact]

    # Risk metrics under scenario
    var_under_scenario: float | None = None
    margin_call_triggered: bool = False
    margin_shortfall: float = 0.0

    # Worst affected
    worst_position: str = ""
    worst_position_pnl_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            'scenario_name': self.scenario_name,
            'run_timestamp': self.run_timestamp.isoformat(),
            'portfolio_value_before': self.portfolio_value_before,
            'portfolio_value_after': self.portfolio_value_after,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'position_impacts': [p.to_dict() for p in self.position_impacts],
            'var_under_scenario': self.var_under_scenario,
            'margin_call_triggered': self.margin_call_triggered,
            'margin_shortfall': self.margin_shortfall,
            'worst_position': self.worst_position,
            'worst_position_pnl_pct': self.worst_position_pnl_pct,
        }


@dataclass
class WhatIfResult:
    """Result of what-if analysis (#P16)."""
    description: str
    position_changes: dict[str, int]  # symbol -> quantity change

    # Before/after comparison
    metrics_before: dict
    metrics_after: dict

    # Impact
    pnl_impact: float
    var_impact: float
    margin_impact: float
    exposure_change: dict

    recommendation: str

    def to_dict(self) -> dict:
        return {
            'description': self.description,
            'position_changes': self.position_changes,
            'metrics_before': self.metrics_before,
            'metrics_after': self.metrics_after,
            'pnl_impact': self.pnl_impact,
            'var_impact': self.var_impact,
            'margin_impact': self.margin_impact,
            'exposure_change': self.exposure_change,
            'recommendation': self.recommendation,
        }


class HistoricalEventLibrary:
    """
    Library of historical market events for stress testing (#R18).

    Contains calibrated shocks based on actual market moves.
    """

    EVENTS = {
        HistoricalEvent.BLACK_MONDAY_1987: ScenarioDefinition(
            name="Black Monday 1987",
            description="October 19, 1987 - Largest one-day percentage decline in DJIA history (-22.6%)",
            event_date=date(1987, 10, 19),
            duration_days=1,
            is_historical=True,
            vix_level=150.0,
            shocks=[
                AssetShock("us_equity", -22.6, 5.0, 0.3),
                AssetShock("international_equity", -15.0, 4.0, 0.4),
                AssetShock("treasury", 2.0, 1.5),
                AssetShock("gold", 3.0, 2.0),
                AssetShock("usd", 2.0, 1.5),
            ],
        ),

        HistoricalEvent.LTCM_1998: ScenarioDefinition(
            name="LTCM Crisis 1998",
            description="August-September 1998 - Russian default and LTCM collapse",
            event_date=date(1998, 8, 17),
            duration_days=30,
            is_historical=True,
            vix_level=45.0,
            credit_spread_change_bps=200,
            shocks=[
                AssetShock("us_equity", -15.0, 2.5, 0.2),
                AssetShock("emerging_market", -35.0, 4.0, 0.5),
                AssetShock("treasury", 5.0, 1.2),
                AssetShock("high_yield", -12.0, 3.0),
                AssetShock("commodity", -10.0, 2.0),
            ],
        ),

        HistoricalEvent.GFC_2008: ScenarioDefinition(
            name="Global Financial Crisis 2008",
            description="September-October 2008 - Lehman collapse and credit freeze",
            event_date=date(2008, 9, 15),
            duration_days=60,
            is_historical=True,
            vix_level=80.0,
            credit_spread_change_bps=500,
            interest_rate_change_bps=-100,
            shocks=[
                AssetShock("us_equity", -40.0, 4.0, 0.4),
                AssetShock("international_equity", -45.0, 4.5, 0.5),
                AssetShock("emerging_market", -55.0, 5.0, 0.6),
                AssetShock("real_estate", -35.0, 3.0),
                AssetShock("high_yield", -25.0, 5.0),
                AssetShock("treasury", 10.0, 2.0),
                AssetShock("gold", 5.0, 2.5),
                AssetShock("commodity", -50.0, 4.0),
            ],
        ),

        HistoricalEvent.FLASH_CRASH_2010: ScenarioDefinition(
            name="Flash Crash 2010",
            description="May 6, 2010 - Intraday crash and recovery",
            event_date=date(2010, 5, 6),
            duration_days=1,
            is_historical=True,
            vix_level=40.0,
            shocks=[
                AssetShock("us_equity", -9.0, 8.0, 0.5),
                AssetShock("etf", -15.0, 10.0, 0.6),
                AssetShock("treasury", 1.5, 2.0),
            ],
        ),

        HistoricalEvent.COVID_2020: ScenarioDefinition(
            name="COVID-19 Crash 2020",
            description="February-March 2020 - Pandemic market crash",
            event_date=date(2020, 2, 20),
            duration_days=30,
            is_historical=True,
            vix_level=82.0,
            credit_spread_change_bps=350,
            interest_rate_change_bps=-150,
            shocks=[
                AssetShock("us_equity", -35.0, 5.0, 0.5),
                AssetShock("international_equity", -35.0, 5.0, 0.5),
                AssetShock("emerging_market", -40.0, 6.0, 0.6),
                AssetShock("energy", -65.0, 8.0, 0.3),
                AssetShock("high_yield", -20.0, 6.0),
                AssetShock("treasury", 8.0, 3.0),
                AssetShock("gold", 10.0, 3.0),
                AssetShock("real_estate", -25.0, 3.0),
            ],
        ),

        HistoricalEvent.VOLMAGEDDON_2018: ScenarioDefinition(
            name="Volmageddon 2018",
            description="February 5, 2018 - XIV collapse and vol spike",
            event_date=date(2018, 2, 5),
            duration_days=1,
            is_historical=True,
            vix_level=50.0,
            shocks=[
                AssetShock("us_equity", -4.0, 4.0, 0.3),
                AssetShock("volatility", 100.0, 5.0, -0.9),  # VIX doubled
                AssetShock("treasury", 0.5, 1.5),
            ],
        ),

        HistoricalEvent.RATE_SHOCK_2022: ScenarioDefinition(
            name="Rate Shock 2022",
            description="2022 - Aggressive Fed tightening",
            event_date=date(2022, 1, 1),
            duration_days=365,
            is_historical=True,
            vix_level=35.0,
            interest_rate_change_bps=425,
            shocks=[
                AssetShock("us_equity", -25.0, 1.8, 0.2),
                AssetShock("growth_equity", -35.0, 2.5, 0.3),
                AssetShock("treasury", -15.0, 2.0),
                AssetShock("investment_grade", -18.0, 2.0),
                AssetShock("high_yield", -15.0, 2.5),
                AssetShock("real_estate", -30.0, 2.0),
                AssetShock("crypto", -65.0, 4.0),
            ],
        ),
    }

    @classmethod
    def get_event(cls, event: HistoricalEvent) -> ScenarioDefinition:
        """Get scenario definition for a historical event."""
        return cls.EVENTS.get(event)

    @classmethod
    def get_all_events(cls) -> list[ScenarioDefinition]:
        """Get all historical event scenarios."""
        return list(cls.EVENTS.values())

    @classmethod
    def get_events_by_severity(cls, min_equity_drop: float = -10.0) -> list[ScenarioDefinition]:
        """Get events with equity drop worse than threshold."""
        results = []
        for scenario in cls.EVENTS.values():
            for shock in scenario.shocks:
                if shock.asset_class == "us_equity" and shock.price_change_pct <= min_equity_drop:
                    results.append(scenario)
                    break
        return sorted(results, key=lambda s: next(
            (sh.price_change_pct for sh in s.shocks if sh.asset_class == "us_equity"), 0
        ))


class ScenarioEngine:
    """
    Runs scenario analysis on portfolios (#R17, #R18).

    Supports both historical replay and custom scenarios.
    """

    def __init__(
        self,
        margin_requirement_pct: float = 25.0,  # Portfolio margin
    ):
        self.margin_requirement_pct = margin_requirement_pct

        # Current positions
        self._positions: dict[str, dict] = {}  # symbol -> {quantity, price, asset_class}

        # Asset class mappings
        self._symbol_to_asset_class: dict[str, str] = {}

        # Custom scenarios
        self._custom_scenarios: dict[str, ScenarioDefinition] = {}

        # Results history
        self._results_history: list[ScenarioResult] = []

    def update_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        asset_class: str = "us_equity",
    ) -> None:
        """Update position for scenario analysis."""
        self._positions[symbol] = {
            'quantity': quantity,
            'price': price,
            'asset_class': asset_class,
        }
        self._symbol_to_asset_class[symbol] = asset_class

    def add_custom_scenario(self, scenario: ScenarioDefinition) -> None:
        """Add a custom scenario."""
        self._custom_scenarios[scenario.name] = scenario

    def run_scenario(
        self,
        scenario: ScenarioDefinition,
        account_equity: float | None = None,
    ) -> ScenarioResult:
        """
        Run a scenario on the current portfolio.

        Args:
            scenario: Scenario to run
            account_equity: Account equity for margin calculations

        Returns:
            ScenarioResult with detailed impact
        """
        # Create shock lookup
        shock_by_class = {s.asset_class: s for s in scenario.shocks}

        position_impacts = []
        total_value_before = 0.0
        total_value_after = 0.0

        for symbol, pos in self._positions.items():
            asset_class = pos['asset_class']
            current_value = pos['quantity'] * pos['price']
            total_value_before += abs(current_value)

            # Find applicable shock
            shock = shock_by_class.get(asset_class)

            if shock:
                price_change = shock.price_change_pct / 100
                # Direction matters for sign
                if pos['quantity'] > 0:
                    stressed_value = current_value * (1 + price_change)
                else:
                    # Short positions benefit from price drops
                    stressed_value = current_value * (1 - price_change)
            else:
                # No shock for this asset class
                stressed_value = current_value

            pnl = stressed_value - current_value
            pnl_pct = (pnl / abs(current_value) * 100) if current_value != 0 else 0

            total_value_after += stressed_value if pos['quantity'] > 0 else abs(stressed_value)

            position_impacts.append(PositionImpact(
                symbol=symbol,
                quantity=pos['quantity'],
                current_value=current_value,
                stressed_value=stressed_value,
                pnl=pnl,
                pnl_pct=pnl_pct,
                contribution_to_total=0.0,  # Calculated below
            ))

        # Calculate total P&L
        total_pnl = sum(p.pnl for p in position_impacts)
        total_pnl_pct = (total_pnl / total_value_before * 100) if total_value_before > 0 else 0

        # Update contributions
        for impact in position_impacts:
            impact.contribution_to_total = (impact.pnl / total_pnl * 100) if total_pnl != 0 else 0

        # Find worst position
        worst = min(position_impacts, key=lambda p: p.pnl_pct) if position_impacts else None

        # Check margin
        margin_call = False
        margin_shortfall = 0.0
        if account_equity is not None:
            equity_after = account_equity + total_pnl
            margin_required = total_value_after * self.margin_requirement_pct / 100
            if equity_after < margin_required:
                margin_call = True
                margin_shortfall = margin_required - equity_after

        result = ScenarioResult(
            scenario_name=scenario.name,
            run_timestamp=datetime.now(timezone.utc),
            portfolio_value_before=total_value_before,
            portfolio_value_after=total_value_after,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            position_impacts=position_impacts,
            margin_call_triggered=margin_call,
            margin_shortfall=margin_shortfall,
            worst_position=worst.symbol if worst else "",
            worst_position_pnl_pct=worst.pnl_pct if worst else 0.0,
        )

        self._results_history.append(result)
        return result

    def run_historical_event(
        self,
        event: HistoricalEvent,
        account_equity: float | None = None,
    ) -> ScenarioResult:
        """Run a historical event scenario."""
        scenario = HistoricalEventLibrary.get_event(event)
        if scenario is None:
            raise ValueError(f"Unknown historical event: {event}")
        return self.run_scenario(scenario, account_equity)

    def run_all_historical_events(
        self,
        account_equity: float | None = None,
    ) -> list[ScenarioResult]:
        """Run all historical event scenarios."""
        results = []
        for event in HistoricalEvent:
            try:
                result = self.run_historical_event(event, account_equity)
                results.append(result)
            except ValueError:
                continue
        return results

    def find_worst_case(
        self,
        account_equity: float | None = None,
    ) -> ScenarioResult:
        """
        Find the worst-case scenario from all available (#R17).

        Runs all scenarios and returns the one with worst P&L.
        """
        all_results = self.run_all_historical_events(account_equity)

        # Also run custom scenarios
        for scenario in self._custom_scenarios.values():
            result = self.run_scenario(scenario, account_equity)
            all_results.append(result)

        if not all_results:
            raise ValueError("No scenarios available")

        # Find worst
        return min(all_results, key=lambda r: r.total_pnl)

    def generate_worst_case_report(
        self,
        account_equity: float,
        include_top_n: int = 5,
    ) -> dict:
        """
        Generate comprehensive worst-case report (#R17).

        Returns top N worst scenarios with detailed analysis.
        """
        all_results = self.run_all_historical_events(account_equity)

        for scenario in self._custom_scenarios.values():
            result = self.run_scenario(scenario, account_equity)
            all_results.append(result)

        # Sort by P&L (worst first)
        all_results.sort(key=lambda r: r.total_pnl)

        top_worst = all_results[:include_top_n]

        # Summary statistics
        avg_loss = sum(r.total_pnl for r in all_results) / len(all_results) if all_results else 0
        margin_calls = sum(1 for r in all_results if r.margin_call_triggered)

        return {
            'report_timestamp': datetime.now(timezone.utc).isoformat(),
            'account_equity': account_equity,
            'num_scenarios_tested': len(all_results),
            'worst_case': top_worst[0].to_dict() if top_worst else None,
            'worst_case_loss_pct': top_worst[0].total_pnl_pct if top_worst else 0,
            'average_scenario_loss': avg_loss,
            'scenarios_triggering_margin_call': margin_calls,
            'top_worst_scenarios': [r.to_dict() for r in top_worst],
            'positions_most_at_risk': self._identify_vulnerable_positions(top_worst),
        }

    def _identify_vulnerable_positions(
        self,
        worst_results: list[ScenarioResult],
    ) -> list[dict]:
        """Identify positions that appear frequently in worst scenarios."""
        position_losses: dict[str, list[float]] = {}

        for result in worst_results:
            for impact in result.position_impacts:
                if impact.symbol not in position_losses:
                    position_losses[impact.symbol] = []
                position_losses[impact.symbol].append(impact.pnl_pct)

        # Calculate average loss per position
        vulnerable = []
        for symbol, losses in position_losses.items():
            avg_loss = sum(losses) / len(losses)
            vulnerable.append({
                'symbol': symbol,
                'avg_loss_pct': avg_loss,
                'worst_loss_pct': min(losses),
                'scenarios_affected': len(losses),
            })

        # Sort by average loss
        vulnerable.sort(key=lambda x: x['avg_loss_pct'])
        return vulnerable[:10]


class WhatIfAnalyzer:
    """
    What-if analysis for position changes (#P16).

    Allows testing hypothetical portfolio changes.
    """

    def __init__(
        self,
        scenario_engine: ScenarioEngine,
        margin_calc: Callable[[dict], float] | None = None,
        var_calc: Callable[[dict], float] | None = None,
    ):
        self.scenario_engine = scenario_engine
        self._margin_calc = margin_calc or self._default_margin_calc
        self._var_calc = var_calc or self._default_var_calc

    def _default_margin_calc(self, positions: dict) -> float:
        """Default margin calculation."""
        total_value = sum(
            abs(p['quantity'] * p['price'])
            for p in positions.values()
        )
        return total_value * 0.25  # 25% margin

    def _default_var_calc(self, positions: dict) -> float:
        """Default VaR calculation (simplified)."""
        total_value = sum(
            abs(p['quantity'] * p['price'])
            for p in positions.values()
        )
        return total_value * 0.05  # 5% 95% VaR assumption

    def analyze_position_change(
        self,
        symbol: str,
        quantity_change: int,
        price: float | None = None,
        asset_class: str = "us_equity",
    ) -> WhatIfResult:
        """
        Analyze impact of changing a single position.

        Args:
            symbol: Symbol to change
            quantity_change: Change in quantity (positive = buy, negative = sell)
            price: Current price (uses existing if not provided)
            asset_class: Asset class for new positions

        Returns:
            WhatIfResult with before/after comparison
        """
        # Get current state
        current_positions = dict(self.scenario_engine._positions)

        # Calculate current metrics
        metrics_before = {
            'total_exposure': sum(
                abs(p['quantity'] * p['price'])
                for p in current_positions.values()
            ),
            'margin': self._margin_calc(current_positions),
            'var': self._var_calc(current_positions),
            'position_count': len(current_positions),
        }

        # Apply change
        new_positions = dict(current_positions)
        if symbol in new_positions:
            current_qty = new_positions[symbol]['quantity']
            new_qty = current_qty + quantity_change
            if new_qty == 0:
                del new_positions[symbol]
            else:
                new_positions[symbol] = {
                    **new_positions[symbol],
                    'quantity': new_qty,
                }
        else:
            if quantity_change != 0:
                new_positions[symbol] = {
                    'quantity': quantity_change,
                    'price': price or 100.0,
                    'asset_class': asset_class,
                }

        # Calculate new metrics
        metrics_after = {
            'total_exposure': sum(
                abs(p['quantity'] * p['price'])
                for p in new_positions.values()
            ),
            'margin': self._margin_calc(new_positions),
            'var': self._var_calc(new_positions),
            'position_count': len(new_positions),
        }

        # Calculate impacts
        pnl_impact = 0  # No immediate P&L from position change
        var_impact = metrics_after['var'] - metrics_before['var']
        margin_impact = metrics_after['margin'] - metrics_before['margin']

        exposure_change = {
            'total': metrics_after['total_exposure'] - metrics_before['total_exposure'],
            'symbol': symbol,
            'quantity_change': quantity_change,
        }

        # Generate recommendation
        if var_impact > 0 and var_impact / max(1, metrics_before['var']) > 0.1:
            recommendation = "CAUTION: This change increases VaR significantly"
        elif margin_impact > 0 and margin_impact / max(1, metrics_before['margin']) > 0.2:
            recommendation = "WARNING: This change increases margin requirements substantially"
        elif var_impact < 0:
            recommendation = "FAVORABLE: This change reduces portfolio risk"
        else:
            recommendation = "NEUTRAL: Minimal impact on risk metrics"

        return WhatIfResult(
            description=f"Change {symbol} by {quantity_change:+d} shares",
            position_changes={symbol: quantity_change},
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            pnl_impact=pnl_impact,
            var_impact=var_impact,
            margin_impact=margin_impact,
            exposure_change=exposure_change,
            recommendation=recommendation,
        )

    def analyze_multiple_changes(
        self,
        changes: dict[str, dict],
    ) -> WhatIfResult:
        """
        Analyze impact of multiple position changes.

        Args:
            changes: Dict of symbol -> {quantity_change, price, asset_class}

        Returns:
            WhatIfResult with aggregate impact
        """
        # Get current state
        current_positions = dict(self.scenario_engine._positions)

        # Calculate current metrics
        metrics_before = {
            'total_exposure': sum(
                abs(p['quantity'] * p['price'])
                for p in current_positions.values()
            ),
            'margin': self._margin_calc(current_positions),
            'var': self._var_calc(current_positions),
            'position_count': len(current_positions),
        }

        # Apply all changes
        new_positions = dict(current_positions)
        position_changes = {}

        for symbol, change in changes.items():
            qty_change = change.get('quantity_change', 0)
            position_changes[symbol] = qty_change

            if symbol in new_positions:
                current_qty = new_positions[symbol]['quantity']
                new_qty = current_qty + qty_change
                if new_qty == 0:
                    del new_positions[symbol]
                else:
                    new_positions[symbol]['quantity'] = new_qty
            else:
                if qty_change != 0:
                    new_positions[symbol] = {
                        'quantity': qty_change,
                        'price': change.get('price', 100.0),
                        'asset_class': change.get('asset_class', 'us_equity'),
                    }

        # Calculate new metrics
        metrics_after = {
            'total_exposure': sum(
                abs(p['quantity'] * p['price'])
                for p in new_positions.values()
            ),
            'margin': self._margin_calc(new_positions),
            'var': self._var_calc(new_positions),
            'position_count': len(new_positions),
        }

        var_impact = metrics_after['var'] - metrics_before['var']
        margin_impact = metrics_after['margin'] - metrics_before['margin']

        exposure_change = {
            'total': metrics_after['total_exposure'] - metrics_before['total_exposure'],
            'num_changes': len(position_changes),
        }

        # Generate recommendation
        if var_impact > 0 and var_impact / max(1, metrics_before['var']) > 0.1:
            recommendation = "CAUTION: These changes increase VaR significantly"
        elif margin_impact > 0 and margin_impact / max(1, metrics_before['margin']) > 0.2:
            recommendation = "WARNING: These changes increase margin requirements substantially"
        elif var_impact < 0 and margin_impact < 0:
            recommendation = "FAVORABLE: These changes reduce both risk and margin"
        else:
            recommendation = "MIXED: Review individual impacts before proceeding"

        return WhatIfResult(
            description=f"Multiple position changes ({len(changes)} symbols)",
            position_changes=position_changes,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            pnl_impact=0,
            var_impact=var_impact,
            margin_impact=margin_impact,
            exposure_change=exposure_change,
            recommendation=recommendation,
        )

    def find_optimal_hedge(
        self,
        target_symbol: str,
        hedge_candidates: list[str],
        target_var_reduction_pct: float = 20.0,
    ) -> list[WhatIfResult]:
        """
        Find optimal hedge for a position.

        Tests various hedge ratios and returns best options.
        """
        if target_symbol not in self.scenario_engine._positions:
            return []

        target_pos = self.scenario_engine._positions[target_symbol]
        target_value = abs(target_pos['quantity'] * target_pos['price'])

        results = []

        for hedge_symbol in hedge_candidates:
            # Test various hedge ratios
            for ratio in [0.25, 0.5, 0.75, 1.0]:
                hedge_value = target_value * ratio

                # Assume opposite direction
                hedge_sign = -1 if target_pos['quantity'] > 0 else 1

                # Estimate hedge quantity
                if hedge_symbol in self.scenario_engine._positions:
                    hedge_price = self.scenario_engine._positions[hedge_symbol]['price']
                else:
                    hedge_price = 100.0  # Default

                hedge_qty = int(hedge_sign * hedge_value / hedge_price)

                result = self.analyze_position_change(
                    hedge_symbol, hedge_qty, hedge_price
                )

                # Check if meets target
                if result.var_impact < 0:
                    var_reduction_pct = abs(result.var_impact / max(1, result.metrics_before['var']) * 100)
                    if var_reduction_pct >= target_var_reduction_pct:
                        results.append(result)

        # Sort by VaR reduction (most reduction first)
        results.sort(key=lambda r: r.var_impact)
        return results[:5]


# =============================================================================
# P2 Enhancement: Historical Scenario Replay
# =============================================================================

@dataclass
class HistoricalReplayConfig:
    """Configuration for historical scenario replay."""
    start_date: date
    end_date: date
    replay_speed: float = 1.0  # 1.0 = real-time, 2.0 = 2x speed
    include_intraday: bool = False
    data_source: str = "synthetic"  # 'synthetic', 'file', 'api'


@dataclass
class ReplaySnapshot:
    """Snapshot of portfolio state during replay."""
    timestamp: datetime
    portfolio_value: float
    pnl: float
    pnl_pct: float
    position_values: dict[str, float]
    market_conditions: dict[str, Any]


class HistoricalScenarioReplayer:
    """
    Historical scenario replay engine (P2 Enhancement).

    Replays historical market conditions to test strategy behavior.
    """

    def __init__(self, scenario_engine: ScenarioEngine):
        self.scenario_engine = scenario_engine
        self._historical_data: dict[str, list[dict]] = {}  # symbol -> [daily data]
        self._replay_snapshots: list[ReplaySnapshot] = []

    def load_historical_event(self, event: HistoricalEvent) -> bool:
        """
        Load historical data for a specific event.

        Generates synthetic data based on event characteristics.
        """
        scenario = HistoricalEventLibrary.get_event(event)
        if not scenario:
            return False

        # Generate synthetic daily data for the event period
        self._historical_data = self._generate_synthetic_event_data(scenario)
        return True

    def _generate_synthetic_event_data(
        self,
        scenario: ScenarioDefinition,
    ) -> dict[str, list[dict]]:
        """Generate synthetic daily data for scenario replay."""
        data = {}
        duration = scenario.duration_days

        for shock in scenario.shocks:
            asset_class = shock.asset_class
            daily_data = []

            # Calculate daily shock (distribute over duration)
            daily_change = shock.price_change_pct / max(1, duration)

            base_price = 100.0
            current_price = base_price

            for day in range(duration):
                # Add some randomness around the trend
                import random
                noise = random.gauss(0, abs(daily_change) * 0.3)
                actual_change = daily_change + noise

                current_price *= (1 + actual_change / 100)

                daily_data.append({
                    'day': day,
                    'price': current_price,
                    'change_pct': actual_change,
                    'volatility': shock.volatility_multiplier * 0.02,  # Base 2% vol
                    'volume_multiplier': 1.0 + abs(daily_change) * 0.5,
                })

            data[asset_class] = daily_data

        return data

    def replay(
        self,
        config: HistoricalReplayConfig,
        callback: Callable[[ReplaySnapshot], None] | None = None,
    ) -> list[ReplaySnapshot]:
        """
        Replay historical scenario day by day.

        Args:
            config: Replay configuration
            callback: Optional callback for each snapshot

        Returns:
            List of replay snapshots
        """
        self._replay_snapshots = []

        if not self._historical_data:
            logger.warning("No historical data loaded for replay")
            return []

        # Get max duration
        max_days = max(len(data) for data in self._historical_data.values())

        initial_value = sum(
            abs(p['quantity'] * p['price'])
            for p in self.scenario_engine._positions.values()
        )

        cumulative_pnl = 0.0

        for day in range(max_days):
            # Get market conditions for this day
            market_conditions = {}
            position_values = {}

            for asset_class, daily_data in self._historical_data.items():
                if day < len(daily_data):
                    market_conditions[asset_class] = daily_data[day]

            # Calculate position values and P&L
            daily_pnl = 0.0
            for symbol, pos in self.scenario_engine._positions.items():
                asset_class = pos.get('asset_class', 'us_equity')
                if asset_class in market_conditions:
                    day_data = market_conditions[asset_class]
                    daily_change = day_data['change_pct'] / 100

                    position_value = pos['quantity'] * pos['price']
                    position_pnl = position_value * daily_change

                    # Update for shorts
                    if pos['quantity'] < 0:
                        position_pnl = -position_pnl

                    daily_pnl += position_pnl
                    position_values[symbol] = position_value + position_pnl

            cumulative_pnl += daily_pnl
            portfolio_value = initial_value + cumulative_pnl
            pnl_pct = (cumulative_pnl / initial_value * 100) if initial_value > 0 else 0

            # Create snapshot
            snapshot = ReplaySnapshot(
                timestamp=datetime.now(timezone.utc),
                portfolio_value=portfolio_value,
                pnl=cumulative_pnl,
                pnl_pct=pnl_pct,
                position_values=position_values,
                market_conditions=market_conditions,
            )

            self._replay_snapshots.append(snapshot)

            if callback:
                callback(snapshot)

        return self._replay_snapshots

    def get_replay_summary(self) -> dict:
        """Get summary of replay results."""
        if not self._replay_snapshots:
            return {}

        values = [s.portfolio_value for s in self._replay_snapshots]
        pnls = [s.pnl for s in self._replay_snapshots]

        return {
            'num_days': len(self._replay_snapshots),
            'initial_value': values[0] if values else 0,
            'final_value': values[-1] if values else 0,
            'total_pnl': pnls[-1] if pnls else 0,
            'max_drawdown': self._calculate_max_drawdown(values),
            'worst_day_pnl': min(
                self._replay_snapshots[i].pnl - self._replay_snapshots[i-1].pnl
                for i in range(1, len(self._replay_snapshots))
            ) if len(self._replay_snapshots) > 1 else 0,
        }

    def _calculate_max_drawdown(self, values: list[float]) -> float:
        """Calculate maximum drawdown from value series."""
        if not values:
            return 0.0

        peak = values[0]
        max_dd = 0.0

        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd


# =============================================================================
# P2 Enhancement: Custom Scenario Builder
# =============================================================================

class ScenarioBuilder:
    """
    Builder pattern for creating custom scenarios (P2 Enhancement).

    Provides fluent interface for scenario construction.
    """

    def __init__(self, name: str):
        self._name = name
        self._description = ""
        self._shocks: list[AssetShock] = []
        self._event_date: date | None = None
        self._duration_days = 1
        self._is_historical = False
        self._interest_rate_change_bps = 0.0
        self._vix_level: float | None = None
        self._credit_spread_change_bps = 0.0
        self._probability = 1.0
        self._severity_rank = 0
        self._category = "custom"

    def with_description(self, description: str) -> "ScenarioBuilder":
        """Set scenario description."""
        self._description = description
        return self

    def with_equity_shock(
        self,
        change_pct: float,
        asset_class: str = "us_equity",
        volatility_mult: float = 1.0,
        correlation_shift: float = 0.0,
    ) -> "ScenarioBuilder":
        """Add equity shock."""
        self._shocks.append(AssetShock(
            asset_class=asset_class,
            price_change_pct=change_pct,
            volatility_multiplier=volatility_mult,
            correlation_shift=correlation_shift,
        ))
        return self

    def with_rate_shock(self, change_bps: float) -> "ScenarioBuilder":
        """Add interest rate shock."""
        self._interest_rate_change_bps = change_bps
        return self

    def with_credit_shock(self, spread_change_bps: float) -> "ScenarioBuilder":
        """Add credit spread shock."""
        self._credit_spread_change_bps = spread_change_bps
        return self

    def with_vix_level(self, vix: float) -> "ScenarioBuilder":
        """Set VIX level."""
        self._vix_level = vix
        return self

    def with_duration(self, days: int) -> "ScenarioBuilder":
        """Set scenario duration."""
        self._duration_days = days
        return self

    def with_event_date(self, event_date: date) -> "ScenarioBuilder":
        """Set event date."""
        self._event_date = event_date
        return self

    def as_historical(self) -> "ScenarioBuilder":
        """Mark as historical scenario."""
        self._is_historical = True
        self._category = "historical"
        return self

    def with_probability(self, probability: float) -> "ScenarioBuilder":
        """Set scenario probability (for weighting)."""
        self._probability = max(0.0, min(1.0, probability))
        return self

    def with_severity_rank(self, rank: int) -> "ScenarioBuilder":
        """Set severity ranking."""
        self._severity_rank = rank
        return self

    def with_category(self, category: str) -> "ScenarioBuilder":
        """Set scenario category."""
        self._category = category
        return self

    def add_custom_shock(self, shock: AssetShock) -> "ScenarioBuilder":
        """Add a custom asset shock."""
        self._shocks.append(shock)
        return self

    def build(self) -> ScenarioDefinition:
        """Build the scenario definition."""
        return ScenarioDefinition(
            name=self._name,
            description=self._description,
            shocks=self._shocks,
            event_date=self._event_date,
            duration_days=self._duration_days,
            is_historical=self._is_historical,
            interest_rate_change_bps=self._interest_rate_change_bps,
            vix_level=self._vix_level,
            credit_spread_change_bps=self._credit_spread_change_bps,
            probability=self._probability,
            severity_rank=self._severity_rank,
            category=self._category,
        )

    @classmethod
    def recession_scenario(cls) -> "ScenarioBuilder":
        """Pre-built recession scenario."""
        return (
            cls("Recession")
            .with_description("Economic recession with market correction")
            .with_equity_shock(-25.0, "us_equity", 2.0, 0.2)
            .with_equity_shock(-30.0, "international_equity", 2.5, 0.3)
            .with_rate_shock(-150)
            .with_credit_shock(200)
            .with_vix_level(35.0)
            .with_duration(180)
            .with_probability(0.15)
            .with_severity_rank(3)
            .with_category("hypothetical")
        )

    @classmethod
    def inflation_shock_scenario(cls) -> "ScenarioBuilder":
        """Pre-built inflation shock scenario."""
        return (
            cls("Inflation Shock")
            .with_description("Unexpected inflation surge forcing aggressive Fed action")
            .with_equity_shock(-15.0, "us_equity", 1.8, 0.15)
            .with_equity_shock(-20.0, "growth_equity", 2.2, 0.2)
            .with_rate_shock(200)
            .with_credit_shock(100)
            .with_vix_level(30.0)
            .with_duration(90)
            .with_probability(0.10)
            .with_severity_rank(4)
            .with_category("hypothetical")
        )

    @classmethod
    def geopolitical_crisis_scenario(cls) -> "ScenarioBuilder":
        """Pre-built geopolitical crisis scenario."""
        return (
            cls("Geopolitical Crisis")
            .with_description("Major geopolitical event causing flight to safety")
            .with_equity_shock(-18.0, "us_equity", 3.0, 0.4)
            .with_equity_shock(-25.0, "emerging_market", 4.0, 0.5)
            .with_rate_shock(-50)
            .with_credit_shock(150)
            .with_vix_level(40.0)
            .with_duration(30)
            .with_probability(0.05)
            .with_severity_rank(2)
            .with_category("hypothetical")
        )


# =============================================================================
# P2 Enhancement: Probability-Weighted Scenario Analysis
# =============================================================================

@dataclass
class ProbabilityWeightedResult:
    """Result with probability weighting."""
    scenario_name: str
    probability: float
    unweighted_pnl: float
    weighted_pnl: float
    unweighted_pnl_pct: float
    weighted_contribution_pct: float

    def to_dict(self) -> dict:
        return {
            'scenario_name': self.scenario_name,
            'probability': self.probability,
            'unweighted_pnl': self.unweighted_pnl,
            'weighted_pnl': self.weighted_pnl,
            'unweighted_pnl_pct': self.unweighted_pnl_pct,
            'weighted_contribution_pct': self.weighted_contribution_pct,
        }


class ProbabilityWeightedAnalyzer:
    """
    Probability-weighted scenario analysis (P2 Enhancement).

    Weights scenario results by their assigned probabilities.
    """

    def __init__(self, scenario_engine: ScenarioEngine):
        self.scenario_engine = scenario_engine
        self._scenario_results: dict[str, tuple[ScenarioResult, float]] = {}

    def run_weighted_analysis(
        self,
        scenarios: list[ScenarioDefinition],
        account_equity: float,
        normalize_probabilities: bool = True,
    ) -> dict:
        """
        Run probability-weighted scenario analysis.

        Args:
            scenarios: List of scenarios with probabilities
            account_equity: Account equity for calculations
            normalize_probabilities: If True, normalize probabilities to sum to 1

        Returns:
            Dictionary with weighted analysis results
        """
        self._scenario_results = {}
        weighted_results = []

        # Calculate total probability for normalization
        total_prob = sum(s.probability for s in scenarios)
        if normalize_probabilities and total_prob > 0:
            prob_scale = 1.0 / total_prob
        else:
            prob_scale = 1.0

        total_weighted_pnl = 0.0

        for scenario in scenarios:
            # Run scenario
            result = self.scenario_engine.run_scenario(scenario, account_equity)

            # Calculate weighted values
            adjusted_prob = scenario.probability * prob_scale
            weighted_pnl = result.total_pnl * adjusted_prob

            self._scenario_results[scenario.name] = (result, adjusted_prob)
            total_weighted_pnl += weighted_pnl

        # Calculate contributions
        for scenario in scenarios:
            result, prob = self._scenario_results[scenario.name]
            weighted_pnl = result.total_pnl * prob

            weighted_results.append(ProbabilityWeightedResult(
                scenario_name=scenario.name,
                probability=prob,
                unweighted_pnl=result.total_pnl,
                weighted_pnl=weighted_pnl,
                unweighted_pnl_pct=result.total_pnl_pct,
                weighted_contribution_pct=(
                    weighted_pnl / total_weighted_pnl * 100
                    if total_weighted_pnl != 0 else 0
                ),
            ))

        # Sort by probability-weighted impact
        weighted_results.sort(key=lambda r: r.weighted_pnl)

        return {
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'num_scenarios': len(scenarios),
            'expected_loss': total_weighted_pnl,
            'expected_loss_pct': (
                total_weighted_pnl / account_equity * 100
                if account_equity > 0 else 0
            ),
            'worst_scenario': weighted_results[0].scenario_name if weighted_results else None,
            'scenario_results': [r.to_dict() for r in weighted_results],
            'risk_summary': self._generate_risk_summary(weighted_results),
        }

    def _generate_risk_summary(
        self,
        weighted_results: list[ProbabilityWeightedResult],
    ) -> dict:
        """Generate risk summary from weighted results."""
        if not weighted_results:
            return {}

        # Calculate percentiles
        unweighted_pnls = sorted([r.unweighted_pnl for r in weighted_results])

        return {
            'median_scenario_loss': unweighted_pnls[len(unweighted_pnls) // 2],
            'worst_case_loss': min(unweighted_pnls),
            'best_case_loss': max(unweighted_pnls),
            'num_scenarios_with_loss': sum(1 for r in weighted_results if r.unweighted_pnl < 0),
            'total_scenarios': len(weighted_results),
        }

    def get_var_weighted(
        self,
        confidence_level: float = 0.95,
    ) -> float:
        """
        Calculate probability-weighted VaR.

        Uses scenario probabilities to estimate VaR.
        """
        if not self._scenario_results:
            return 0.0

        # Sort scenarios by loss
        sorted_scenarios = sorted(
            self._scenario_results.items(),
            key=lambda x: x[1][0].total_pnl
        )

        # Find VaR at confidence level
        cumulative_prob = 0.0
        target_prob = 1 - confidence_level

        for name, (result, prob) in sorted_scenarios:
            cumulative_prob += prob
            if cumulative_prob >= target_prob:
                return -result.total_pnl

        # If we get here, return worst case
        return -sorted_scenarios[0][1][0].total_pnl if sorted_scenarios else 0.0

    def sensitivity_analysis(
        self,
        base_scenarios: list[ScenarioDefinition],
        account_equity: float,
        probability_shifts: list[float] = [-0.05, -0.02, 0.0, 0.02, 0.05],
    ) -> dict:
        """
        Analyze sensitivity of results to probability assumptions.

        Args:
            base_scenarios: Base scenarios to analyze
            account_equity: Account equity
            probability_shifts: Probability shifts to test

        Returns:
            Sensitivity analysis results
        """
        sensitivity_results = []

        for shift in probability_shifts:
            # Adjust probabilities
            adjusted_scenarios = []
            for scenario in base_scenarios:
                adjusted = ScenarioDefinition(
                    name=scenario.name,
                    description=scenario.description,
                    shocks=scenario.shocks,
                    event_date=scenario.event_date,
                    duration_days=scenario.duration_days,
                    is_historical=scenario.is_historical,
                    interest_rate_change_bps=scenario.interest_rate_change_bps,
                    vix_level=scenario.vix_level,
                    credit_spread_change_bps=scenario.credit_spread_change_bps,
                    probability=max(0.01, min(0.99, scenario.probability + shift)),
                    severity_rank=scenario.severity_rank,
                    category=scenario.category,
                )
                adjusted_scenarios.append(adjusted)

            # Run analysis
            result = self.run_weighted_analysis(
                adjusted_scenarios, account_equity, normalize_probabilities=True
            )

            sensitivity_results.append({
                'probability_shift': shift,
                'expected_loss': result['expected_loss'],
                'expected_loss_pct': result['expected_loss_pct'],
            })

        return {
            'base_expected_loss': next(
                (r['expected_loss'] for r in sensitivity_results if r['probability_shift'] == 0),
                0
            ),
            'sensitivity_results': sensitivity_results,
            'max_expected_loss': max(r['expected_loss'] for r in sensitivity_results),
            'min_expected_loss': min(r['expected_loss'] for r in sensitivity_results),
        }

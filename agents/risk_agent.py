"""
Risk Agent
==========

Validates all trading decisions against risk limits.
Implements kill-switch mechanism for emergency halt.

Responsibility: Risk validation ONLY.
Does NOT handle regulatory compliance (see ComplianceAgent).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Optional
from enum import Enum
import numpy as np

from core.agent_base import ValidationAgent, AgentConfig
from core.events import (
    Event,
    EventType,
    DecisionEvent,
    ValidatedDecisionEvent,
    RiskAlertEvent,
    RiskAlertSeverity,
    OrderSide,
    GreeksUpdateEvent,
    StressTestResultEvent,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger
    from core.broker import IBBroker


logger = logging.getLogger(__name__)


class KillSwitchReason(Enum):
    """Reasons for activating kill-switch."""
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    VAR_BREACH = "var_breach"
    MANUAL = "manual"
    CONNECTIVITY_LOSS = "connectivity_loss"
    ANOMALY_DETECTED = "anomaly_detected"
    LATENCY_BREACH = "latency_breach"  # MiFID II RTS 6
    MARKET_DISRUPTION = "market_disruption"
    REGULATORY_HALT = "regulatory_halt"


class DrawdownLevel(Enum):
    """Tiered drawdown response levels."""
    NORMAL = "normal"           # < 5%: Normal trading
    WARNING = "warning"         # 5-7.5%: Warning, log alerts
    REDUCE = "reduce"           # 7.5-10%: Reduce position sizes
    HALT = "halt"               # > 10%: Kill switch, halt trading


class KillSwitchAction(Enum):
    """Actions available during kill-switch activation."""
    HALT_NEW_ORDERS = "halt_new_orders"
    CANCEL_PENDING = "cancel_pending"
    CLOSE_POSITIONS = "close_positions"
    FULL_SHUTDOWN = "full_shutdown"


@dataclass
class PositionInfo:
    """Position information for a single instrument."""
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    weight_pct: float  # % of portfolio
    sector: str = "unknown"


@dataclass
class GreeksState:
    """Portfolio Greeks state with staleness tracking."""
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_stale(self, max_age_seconds: float) -> bool:
        """Check if Greeks data is stale."""
        age = (datetime.now(timezone.utc) - self.last_update).total_seconds()
        return age > max_age_seconds

    def age_seconds(self) -> float:
        """Get age of Greeks data in seconds."""
        return (datetime.now(timezone.utc) - self.last_update).total_seconds()


@dataclass
class PositionGreeks:
    """Per-position Greeks for individual position risk limits (#R4)."""
    symbol: str
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    contracts: int = 0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
            "contracts": self.contracts,
            "last_update": self.last_update.isoformat(),
        }


@dataclass
class LiquidityMetrics:
    """Liquidity risk metrics."""
    avg_daily_volume: dict[str, int] = field(default_factory=dict)
    position_pct_of_adv: dict[str, float] = field(default_factory=dict)
    estimated_liquidation_days: dict[str, float] = field(default_factory=dict)


@dataclass
class DrawdownRecoveryState:
    """
    Tracks drawdown recovery metrics (#R11).

    Monitors:
    - Current drawdown start time
    - Historical recovery times
    - Recovery velocity
    """
    drawdown_start_time: datetime | None = None
    drawdown_start_equity: float = 0.0
    drawdown_trough_equity: float = 0.0
    drawdown_trough_time: datetime | None = None
    recovery_start_time: datetime | None = None  # When we started recovering
    is_recovering: bool = False  # True when climbing back from trough

    # Historical metrics
    recovery_times_days: list[float] = field(default_factory=list)  # Days to recover from past drawdowns
    avg_recovery_time_days: float = 0.0
    max_recovery_time_days: float = 0.0
    recovery_count: int = 0  # Number of completed recoveries


@dataclass
class MarginState:
    """
    Intraday margin monitoring state (#R10).

    Tracks margin requirements throughout the trading day,
    not just at EOD.
    """
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    available_margin: float = 0.0
    margin_utilization_pct: float = 0.0
    margin_excess: float = 0.0  # Available - Maintenance

    # Intraday tracking
    last_margin_check: datetime | None = None
    margin_check_count_today: int = 0
    intraday_peak_utilization: float = 0.0
    intraday_margin_calls: int = 0

    # Thresholds
    warning_utilization_pct: float = 70.0
    critical_utilization_pct: float = 85.0
    margin_call_pct: float = 100.0

    def is_warning(self) -> bool:
        """Check if margin is at warning level."""
        return self.margin_utilization_pct >= self.warning_utilization_pct

    def is_critical(self) -> bool:
        """Check if margin is at critical level."""
        return self.margin_utilization_pct >= self.critical_utilization_pct

    def is_margin_call(self) -> bool:
        """Check if in margin call territory."""
        return self.margin_utilization_pct >= self.margin_call_pct


@dataclass
class RiskState:
    """Current risk state of the portfolio."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Portfolio values
    net_liquidation: float = 1_000_000.0
    total_cash: float = 500_000.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0

    # P&L
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0

    # Drawdown
    peak_equity: float = 1_000_000.0
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0

    # Drawdown recovery tracking (#R11)
    drawdown_recovery: DrawdownRecoveryState = field(default_factory=DrawdownRecoveryState)

    # Margin tracking (#R10)
    margin: MarginState = field(default_factory=MarginState)

    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    leverage: float = 0.0

    # Enhanced VaR metrics
    var_parametric: float = 0.0
    var_historical: float = 0.0
    var_monte_carlo: float = 0.0

    # Positions
    positions: dict[str, PositionInfo] = field(default_factory=dict)
    sector_exposure: dict[str, float] = field(default_factory=dict)

    # Rate limiting
    orders_today: int = 0
    orders_this_minute: list[datetime] = field(default_factory=list)
    last_order_time: Optional[datetime] = None

    # Greeks (for options positions)
    greeks: GreeksState = field(default_factory=GreeksState)

    # Per-position Greeks (#R4)
    position_greeks: dict[str, PositionGreeks] = field(default_factory=dict)

    # Liquidity metrics
    liquidity: LiquidityMetrics = field(default_factory=LiquidityMetrics)

    # Stress test results
    last_stress_test_pnl: float = 0.0
    worst_stress_scenario: str = ""


@dataclass
class RiskCheckResult:
    """Result of a single risk check."""
    check_name: str
    passed: bool
    current_value: float
    limit_value: float
    message: str = ""


@dataclass
class RiskValidationResult:
    """Complete result of risk validation."""
    approved: bool
    checks: list[RiskCheckResult]
    risk_metrics: dict[str, float]
    adjusted_quantity: Optional[int] = None
    rejection_reason: Optional[str] = None


class RiskAgent(ValidationAgent):
    """
    Risk Management Agent.

    Validates all trading decisions against:
    1. Position size limits (5% max per position)
    2. Sector concentration limits (20% max per sector)
    3. Portfolio leverage (2x max)
    4. VaR limits (2% at 95% confidence)
    5. Daily loss limit (-3% triggers halt)
    6. Maximum drawdown (-10% triggers halt)
    7. Rate limits (anti-HFT: 10 orders/min, 100ms interval)

    Implements KILL-SWITCH for emergency situations.
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
        broker: Optional[IBBroker] = None,
    ):
        super().__init__(config, event_bus, audit_logger)
        self._broker = broker

        # Risk limits from config
        limits = config.parameters.get("limits", {})
        self._max_position_pct = limits.get("max_position_size_pct", 5.0) / 100
        self._max_sector_pct = limits.get("max_sector_exposure_pct", 20.0) / 100
        self._max_leverage = limits.get("max_leverage", 2.0)
        self._max_gross_exposure_pct = limits.get("max_gross_exposure_pct", 200.0) / 100
        self._max_var_pct = limits.get("max_portfolio_var_pct", 2.0) / 100
        self._max_daily_loss_pct = limits.get("max_daily_loss_pct", 3.0) / 100
        self._max_drawdown_pct = limits.get("max_drawdown_pct", 10.0) / 100

        # CVaR (Expected Shortfall) threshold alerts (#R13)
        cvar_config = config.parameters.get("cvar_alerts", {})
        self._cvar_alerts_enabled = cvar_config.get("enabled", True)
        self._cvar_warning_pct = cvar_config.get("warning_pct", 2.5) / 100  # Warning at 2.5%
        self._cvar_critical_pct = cvar_config.get("critical_pct", 4.0) / 100  # Critical at 4%
        self._cvar_halt_pct = cvar_config.get("halt_pct", 5.0) / 100  # Halt trading at 5%
        self._last_cvar_alert_level: str | None = None
        self._cvar_alert_cooldown_seconds = cvar_config.get("cooldown_seconds", 300)  # 5 min between alerts
        self._last_cvar_alert_time: datetime | None = None

        # Tiered drawdown thresholds
        drawdown_config = config.parameters.get("drawdown", {})
        self._drawdown_warning_pct = drawdown_config.get("warning_pct", 5.0) / 100      # 5%: Warning
        self._drawdown_reduce_pct = drawdown_config.get("reduce_pct", 7.5) / 100        # 7.5%: Reduce positions
        self._drawdown_halt_pct = drawdown_config.get("halt_pct", 10.0) / 100           # 10%: Halt trading
        self._drawdown_position_reduction = drawdown_config.get("position_reduction_factor", 0.5)  # 50% size reduction

        # Rate limits (anti-HFT per CLAUDE.md)
        rate_limits = config.parameters.get("rate_limits", {})
        self._max_orders_per_minute = rate_limits.get("max_orders_per_minute", 10)
        self._min_order_interval_ms = rate_limits.get("min_order_interval_ms", 100)

        # Greeks limits (for options positions)
        greeks_limits = config.parameters.get("greeks", {})
        self._max_delta = greeks_limits.get("max_portfolio_delta", 500)
        self._max_gamma = greeks_limits.get("max_portfolio_gamma", 100)
        self._max_vega = greeks_limits.get("max_portfolio_vega", 50000)
        self._max_theta = greeks_limits.get("max_portfolio_theta", -10000)

        # Per-position Greeks limits (#R4)
        position_greeks = greeks_limits.get("position_limits", {})
        self._max_position_delta = position_greeks.get("max_delta", 100)  # Max delta per position
        self._max_position_gamma = position_greeks.get("max_gamma", 20)   # Max gamma per position
        self._max_position_vega = position_greeks.get("max_vega", 10000)  # Max vega per position
        self._enforce_position_greeks = position_greeks.get("enabled", True)

        # Greeks staleness configuration
        self._greeks_staleness_warning_seconds = greeks_limits.get("staleness_warning_seconds", 60.0)  # 1 minute
        self._greeks_staleness_critical_seconds = greeks_limits.get("staleness_critical_seconds", 300.0)  # 5 minutes
        self._greeks_staleness_check_enabled = greeks_limits.get("staleness_check_enabled", True)
        self._require_fresh_greeks_for_options = greeks_limits.get("require_fresh_for_options", True)

        # Liquidity limits
        liquidity_limits = config.parameters.get("liquidity", {})
        self._max_position_pct_of_adv = liquidity_limits.get("max_position_pct_of_adv", 10.0) / 100
        self._max_order_pct_of_adv = liquidity_limits.get("max_order_pct_of_adv", 5.0) / 100

        # Stress testing config
        stress_config = config.parameters.get("stress_testing", {})
        self._max_stress_loss_pct = stress_config.get("max_scenario_loss_pct", 25.0) / 100
        self._stress_test_enabled = stress_config.get("enabled", True)

        # Market data staleness config
        staleness_config = config.parameters.get("market_data_staleness", {})
        self._staleness_check_enabled = staleness_config.get("enabled", True)
        self._staleness_warning_seconds = staleness_config.get("warning_seconds", 5.0)
        self._staleness_critical_seconds = staleness_config.get("critical_seconds", 30.0)

        # Sector mappings
        self._sector_map = config.parameters.get("sector_map", {})

        # Enhanced VaR calculator (lazy init)
        self._var_calculator = None
        self._stress_tester = None
        self._correlation_manager = None

        # Risk notifier for limit breach notifications (#R27)
        self._risk_notifier = None

        # State
        self._risk_state = RiskState()
        self._kill_switch_active = False
        self._kill_switch_reason: Optional[KillSwitchReason] = None
        self._kill_switch_time: Optional[datetime] = None
        self._kill_switch_action: KillSwitchAction = KillSwitchAction.HALT_NEW_ORDERS

        # Tiered drawdown state
        self._current_drawdown_level = DrawdownLevel.NORMAL
        self._drawdown_level_time: Optional[datetime] = None

        # MiFID II RTS 6 Kill Switch Configuration
        self._max_latency_ms = config.parameters.get("max_latency_ms", 500)
        self._auto_close_positions = config.parameters.get("auto_close_positions", False)
        self._kill_switch_cooldown_minutes = config.parameters.get("kill_switch_cooldown_minutes", 15)
        self._require_dual_authorization = config.parameters.get("require_dual_authorization", True)

        # Kill switch audit trail
        self._kill_switch_activations: list[dict] = []
        self._authorized_users: set[str] = set(config.parameters.get("authorized_users", ["admin"]))

        # Historical returns for VaR calculation
        self._returns_history: list[float] = []
        self._max_history_days = 252  # 1 year of trading days

        # Pre-allocated rolling buffer for returns (PERF-P1-001)
        self._returns_buffer = np.zeros(252)
        self._buffer_idx = 0
        self._buffer_filled = False  # True once we've filled the buffer once

        # Portfolio refresh tracking (ERR-006)
        self._last_successful_refresh: datetime | None = None
        self._max_stale_data_seconds = config.parameters.get("max_stale_data_seconds", 60.0)
        self._stale_data_kill_switch_enabled = config.parameters.get("stale_data_kill_switch", True)

        # Intraday VaR recalculation triggers (#R2)
        var_recalc_config = config.parameters.get("var_recalculation", {})
        self._var_recalc_enabled = var_recalc_config.get("enabled", True)
        self._var_recalc_position_change_pct = var_recalc_config.get("position_change_pct", 5.0) / 100  # 5% change
        self._var_recalc_exposure_change_pct = var_recalc_config.get("exposure_change_pct", 10.0) / 100  # 10% exposure change
        self._var_recalc_min_interval_seconds = var_recalc_config.get("min_interval_seconds", 60)  # Min 60s between recalcs
        self._var_recalc_max_interval_seconds = var_recalc_config.get("max_interval_seconds", 3600)  # Force every hour
        self._last_var_calc_time: datetime | None = None
        self._last_var_calc_positions: dict[str, float] = {}  # symbol -> market_value at last calc
        self._last_var_calc_exposure: float = 0.0

        # Position aging alerts (P2)
        position_aging_config = config.parameters.get("position_aging", {})
        self._position_aging_enabled = position_aging_config.get("enabled", True)
        self._position_aging_warning_days = position_aging_config.get("warning_days", 30)  # Warn at 30 days
        self._position_aging_critical_days = position_aging_config.get("critical_days", 60)  # Critical at 60 days
        self._position_aging_max_days = position_aging_config.get("max_days", 90)  # Force review at 90 days
        self._position_entry_times: dict[str, datetime] = {}  # symbol -> position entry time
        self._position_aging_alerts: list[dict] = []  # Historical aging alerts

        # Correlation breakdown detection (P2)
        correlation_config = config.parameters.get("correlation_breakdown", {})
        self._correlation_monitoring_enabled = correlation_config.get("enabled", True)
        self._correlation_breakdown_threshold = correlation_config.get("breakdown_threshold", 0.3)  # 0.3 correlation change
        self._correlation_lookback_days = correlation_config.get("lookback_days", 60)  # 60 day baseline
        self._correlation_check_window_days = correlation_config.get("check_window_days", 5)  # 5 day recent window
        self._historical_correlations: dict[tuple[str, str], float] = {}  # (sym1, sym2) -> baseline correlation
        self._current_correlations: dict[tuple[str, str], float] = {}  # (sym1, sym2) -> recent correlation
        self._correlation_breakdown_alerts: list[dict] = []  # Historical breakdown alerts

        # Monitoring
        self._check_latencies: list[float] = []
        self._var_recalc_count_today: int = 0

    async def initialize(self) -> None:
        """Initialize risk state from broker."""
        logger.info(f"RiskAgent initializing with limits: "
                   f"position={self._max_position_pct*100:.1f}%, "
                   f"drawdown={self._max_drawdown_pct*100:.1f}%")

        if self._broker:
            await self._refresh_portfolio_state()

        logger.info("RiskAgent initialized")

    def get_subscribed_events(self) -> list[EventType]:
        """Risk agent subscribes to decisions."""
        return [EventType.DECISION]

    async def process_event(self, event: Event) -> None:
        """Validate trading decisions."""
        if not isinstance(event, DecisionEvent):
            return

        start_time = datetime.now(timezone.utc)

        # Check kill-switch first
        if self._kill_switch_active:
            await self._reject_decision(
                event,
                f"KILL-SWITCH ACTIVE: {self._kill_switch_reason.value if self._kill_switch_reason else 'unknown'}"
            )
            return

        # Refresh portfolio state
        await self._refresh_portfolio_state()

        # Run all risk checks
        result = await self._validate_decision(event)

        # Track latency
        latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        self._check_latencies.append(latency_ms)
        if len(self._check_latencies) > 1000:
            self._check_latencies = self._check_latencies[-1000:]

        # Create validated event
        validated_event = ValidatedDecisionEvent(
            source_agent=self.name,
            original_decision_id=event.event_id,
            approved=result.approved,
            adjusted_quantity=result.adjusted_quantity,
            rejection_reason=result.rejection_reason,
            risk_metrics=result.risk_metrics,
            compliance_checks=tuple(c.check_name for c in result.checks if c.passed),
        )

        # Publish result
        await self._event_bus.publish(validated_event)

        # Log for audit
        self._audit_logger.log_event(validated_event)

        if result.approved:
            logger.info(f"Decision {event.event_id} APPROVED for {event.symbol} "
                       f"(latency={latency_ms:.1f}ms)")
            # Update rate limiting
            self._risk_state.orders_this_minute.append(datetime.now(timezone.utc))
            self._risk_state.last_order_time = datetime.now(timezone.utc)
            self._risk_state.orders_today += 1
        else:
            logger.warning(f"Decision {event.event_id} REJECTED for {event.symbol}: "
                          f"{result.rejection_reason}")

    async def _validate_decision(self, decision: DecisionEvent) -> RiskValidationResult:
        """Run all risk checks on a decision."""
        checks: list[RiskCheckResult] = []
        adjusted_quantity = decision.quantity

        # 1. Position size limit
        pos_check, adj_qty = await self._check_position_limit(decision)
        checks.append(pos_check)
        if not pos_check.passed:
            return RiskValidationResult(
                approved=False,
                checks=checks,
                risk_metrics=self._get_risk_metrics(),
                rejection_reason=pos_check.message
            )
        if adj_qty:
            adjusted_quantity = adj_qty

        # 2. Sector concentration
        sector_check = await self._check_sector_limit(decision)
        checks.append(sector_check)
        if not sector_check.passed:
            return RiskValidationResult(
                approved=False,
                checks=checks,
                risk_metrics=self._get_risk_metrics(),
                rejection_reason=sector_check.message
            )

        # 3. Leverage limit
        leverage_check = await self._check_leverage_limit(decision)
        checks.append(leverage_check)
        if not leverage_check.passed:
            return RiskValidationResult(
                approved=False,
                checks=checks,
                risk_metrics=self._get_risk_metrics(),
                rejection_reason=leverage_check.message
            )

        # 4. VaR limit
        var_check = await self._check_var_limit(decision)
        checks.append(var_check)
        if not var_check.passed:
            return RiskValidationResult(
                approved=False,
                checks=checks,
                risk_metrics=self._get_risk_metrics(),
                rejection_reason=var_check.message
            )

        # 5. Daily loss limit (may trigger kill-switch)
        daily_check = await self._check_daily_loss_limit()
        checks.append(daily_check)
        if not daily_check.passed:
            await self._activate_kill_switch(KillSwitchReason.DAILY_LOSS_LIMIT)
            return RiskValidationResult(
                approved=False,
                checks=checks,
                risk_metrics=self._get_risk_metrics(),
                rejection_reason=daily_check.message
            )

        # 6. Drawdown limit (may trigger kill-switch)
        dd_check = await self._check_drawdown_limit()
        checks.append(dd_check)
        if not dd_check.passed:
            await self._activate_kill_switch(KillSwitchReason.MAX_DRAWDOWN)
            return RiskValidationResult(
                approved=False,
                checks=checks,
                risk_metrics=self._get_risk_metrics(),
                rejection_reason=dd_check.message
            )

        # 7. Rate limit (anti-HFT)
        rate_check = self._check_rate_limit()
        checks.append(rate_check)
        if not rate_check.passed:
            return RiskValidationResult(
                approved=False,
                checks=checks,
                risk_metrics=self._get_risk_metrics(),
                rejection_reason=rate_check.message
            )

        # 8. Order interval (anti-HFT)
        interval_check = self._check_order_interval()
        checks.append(interval_check)
        if not interval_check.passed:
            return RiskValidationResult(
                approved=False,
                checks=checks,
                risk_metrics=self._get_risk_metrics(),
                rejection_reason=interval_check.message
            )

        # 9. Greeks limits (for options)
        greeks_check = await self._check_greeks_limits(decision)
        checks.append(greeks_check)
        if not greeks_check.passed:
            return RiskValidationResult(
                approved=False,
                checks=checks,
                risk_metrics=self._get_risk_metrics(),
                rejection_reason=greeks_check.message
            )

        # 10. Liquidity check
        liquidity_check = await self._check_liquidity(decision)
        checks.append(liquidity_check)
        if not liquidity_check.passed:
            return RiskValidationResult(
                approved=False,
                checks=checks,
                risk_metrics=self._get_risk_metrics(),
                rejection_reason=liquidity_check.message
            )

        # 11. Stress test check (if enabled)
        if self._stress_test_enabled:
            stress_check = await self._check_stress_test_limits()
            checks.append(stress_check)
            if not stress_check.passed:
                return RiskValidationResult(
                    approved=False,
                    checks=checks,
                    risk_metrics=self._get_risk_metrics(),
                    rejection_reason=stress_check.message
                )

        # 12. Market data staleness check
        staleness_check = await self._check_market_data_staleness(decision)
        checks.append(staleness_check)
        if not staleness_check.passed:
            return RiskValidationResult(
                approved=False,
                checks=checks,
                risk_metrics=self._get_risk_metrics(),
                rejection_reason=staleness_check.message
            )

        # All checks passed
        return RiskValidationResult(
            approved=True,
            checks=checks,
            risk_metrics=self._get_risk_metrics(),
            adjusted_quantity=adjusted_quantity if adjusted_quantity != decision.quantity else None
        )

    async def _check_position_limit(
        self, decision: DecisionEvent
    ) -> tuple[RiskCheckResult, Optional[int]]:
        """
        Check position size limit (5% max per CLAUDE.md).

        Also applies tiered drawdown position size reduction when in REDUCE mode.
        """
        symbol = decision.symbol
        current_pos = self._risk_state.positions.get(symbol)
        current_qty = current_pos.quantity if current_pos else 0

        # Calculate new position
        requested_qty = decision.quantity
        delta = requested_qty if decision.action == OrderSide.BUY else -requested_qty
        new_qty = current_qty + delta

        # Apply drawdown-based position size reduction
        size_multiplier = self.get_position_size_multiplier()
        if size_multiplier < 1.0 and requested_qty > 0:
            adjusted_qty = max(1, int(requested_qty * size_multiplier))
            if adjusted_qty != requested_qty:
                logger.info(
                    f"Drawdown REDUCE mode: Position size reduced from {requested_qty} to {adjusted_qty} "
                    f"({size_multiplier*100:.0f}% of original)"
                )
                requested_qty = adjusted_qty
                delta = requested_qty if decision.action == OrderSide.BUY else -requested_qty
                new_qty = current_qty + delta

        # Estimate value (use limit price or last known price)
        price = decision.limit_price or 100.0  # Fallback
        new_value = abs(new_qty) * price

        # Check against limit
        max_value = self._risk_state.net_liquidation * self._max_position_pct
        current_pct = new_value / self._risk_state.net_liquidation if self._risk_state.net_liquidation > 0 else 0

        if new_value > max_value:
            # Try to adjust quantity
            max_qty = int(max_value / price) - abs(current_qty)
            # Also apply drawdown reduction to max_qty
            max_qty = max(1, int(max_qty * size_multiplier)) if size_multiplier < 1.0 else max_qty

            if max_qty >= 10:  # Minimum viable order
                return RiskCheckResult(
                    check_name="position_limit",
                    passed=True,
                    current_value=current_pct,
                    limit_value=self._max_position_pct,
                    message=f"Quantity adjusted from {decision.quantity} to {max_qty}"
                ), max_qty
            else:
                return RiskCheckResult(
                    check_name="position_limit",
                    passed=False,
                    current_value=current_pct,
                    limit_value=self._max_position_pct,
                    message=f"Position would exceed {self._max_position_pct*100:.1f}% limit"
                ), None

        # Return adjusted quantity if drawdown reduction was applied
        if requested_qty != decision.quantity:
            return RiskCheckResult(
                check_name="position_limit",
                passed=True,
                current_value=current_pct,
                limit_value=self._max_position_pct,
                message=f"Drawdown reduction applied: {decision.quantity} → {requested_qty}"
            ), requested_qty

        return RiskCheckResult(
            check_name="position_limit",
            passed=True,
            current_value=current_pct,
            limit_value=self._max_position_pct,
        ), None

    async def _check_sector_limit(self, decision: DecisionEvent) -> RiskCheckResult:
        """
        Check sector and asset class concentration limits (#R8).

        Enhanced checks:
        1. Sector concentration (20% max)
        2. Asset class concentration
        3. HHI (Herfindahl-Hirschman Index) for portfolio concentration
        4. Early warning when approaching limits
        """
        symbol = decision.symbol
        sector = self._sector_map.get(symbol, "unknown")

        current_sector_exposure = self._risk_state.sector_exposure.get(sector, 0.0)

        # Estimate additional exposure
        price = decision.limit_price or 100.0
        additional_exposure = decision.quantity * price / self._risk_state.net_liquidation

        if decision.action == OrderSide.SELL:
            additional_exposure = -additional_exposure

        new_exposure = current_sector_exposure + additional_exposure

        # Check sector limit
        sector_exceeded = new_exposure > self._max_sector_pct
        sector_warning = new_exposure > self._max_sector_pct * 0.8 and not sector_exceeded

        if sector_warning:
            logger.warning(
                f"Sector {sector} approaching limit: {new_exposure*100:.1f}% "
                f"(limit: {self._max_sector_pct*100:.1f}%)"
            )

        # Calculate HHI for portfolio concentration (#R8)
        hhi = self._calculate_portfolio_hhi()

        # HHI thresholds (US DOJ/FTC guidelines adapted for portfolios)
        # HHI < 1500: Unconcentrated
        # HHI 1500-2500: Moderately concentrated
        # HHI > 2500: Highly concentrated
        hhi_warning = hhi > 2500

        if hhi_warning and not sector_exceeded:
            logger.warning(
                f"Portfolio highly concentrated: HHI={hhi:.0f} "
                f"(recommended < 2500 for diversification)"
            )

        message = ""
        if sector_exceeded:
            message = f"Sector {sector} exposure would be {new_exposure*100:.1f}%"
        elif hhi_warning:
            message = f"Portfolio concentration high: HHI={hhi:.0f}"

        return RiskCheckResult(
            check_name="sector_limit",
            passed=not sector_exceeded,
            current_value=new_exposure,
            limit_value=self._max_sector_pct,
            message=message,
            details={
                "sector": sector,
                "sector_exposure_pct": new_exposure * 100,
                "hhi": hhi,
                "hhi_warning": hhi_warning,
                "approaching_limit": sector_warning,
            }
        )

    def _calculate_portfolio_hhi(self) -> float:
        """
        Calculate Herfindahl-Hirschman Index for portfolio concentration (#R8).

        The HHI is a measure of market/portfolio concentration calculated as
        the sum of squared market share (position weight) percentages.

        Formula:
            HHI = sum(w_i^2) * 10000

        Where w_i is the weight of position i (as a decimal).
        Equivalently: sum of (weight_pct)^2 where weight_pct is 0-100.

        Range:
            - Minimum: 10000/N (equally weighted N positions)
            - Maximum: 10000 (single position holds 100%)

        Example:
            - 10 equal positions of 10% each: HHI = 10 * 10^2 = 1000
            - 4 equal positions of 25% each: HHI = 4 * 25^2 = 2500
            - 1 position of 100%: HHI = 100^2 = 10000

        Thresholds (adapted from DOJ/FTC merger guidelines):
            - HHI < 1500: Unconcentrated (well diversified)
            - HHI 1500-2500: Moderately concentrated
            - HHI > 2500: Highly concentrated (concentration risk)

        A useful related metric is "effective number of positions":
            N_eff = 10000 / HHI

        Returns:
            HHI value between ~0 and 10000
        """
        total_value = self._risk_state.net_liquidation
        if total_value <= 0:
            return 10000.0  # Maximum concentration if no value

        # Calculate squared weights: HHI = sum((w_i * 100)^2)
        hhi = 0.0
        for pos in self._risk_state.positions.values():
            weight = abs(pos.market_value) / total_value  # Weight as decimal
            hhi += (weight * 100) ** 2  # Convert to percentage and square

        return hhi

    def get_concentration_metrics(self) -> dict:
        """
        Get comprehensive portfolio concentration metrics (#R8).

        Returns metrics for monitoring:
        - HHI (Herfindahl-Hirschman Index)
        - Top N positions concentration
        - Sector concentration breakdown
        - Effective number of positions
        """
        total_value = self._risk_state.net_liquidation
        if total_value <= 0:
            return {"hhi": 10000, "effective_n": 1, "top_5_pct": 100.0}

        # Calculate position weights
        weights = []
        for pos in self._risk_state.positions.values():
            weight = abs(pos.market_value) / total_value
            weights.append(weight)

        weights.sort(reverse=True)

        # HHI
        hhi = sum((w * 100) ** 2 for w in weights)

        # Effective number of positions (1/HHI * 10000)
        effective_n = 10000 / hhi if hhi > 0 else len(weights)

        # Top 5 concentration
        top_5_pct = sum(weights[:5]) * 100 if weights else 0

        # Sector concentration
        sector_conc = {
            sector: exp * 100
            for sector, exp in self._risk_state.sector_exposure.items()
        }

        return {
            "hhi": hhi,
            "hhi_category": (
                "unconcentrated" if hhi < 1500
                else "moderate" if hhi < 2500
                else "highly_concentrated"
            ),
            "effective_n": effective_n,
            "top_5_pct": top_5_pct,
            "position_count": len(weights),
            "sector_exposure": sector_conc,
            "max_sector": max(sector_conc.values()) if sector_conc else 0,
        }

    async def _check_leverage_limit(self, decision: DecisionEvent) -> RiskCheckResult:
        """
        Check leverage limit (2x max) with position netting consideration (#R3).

        Uses net exposure when positions offset each other, but also checks
        gross exposure for risk management purposes.
        """
        portfolio_value = self._risk_state.net_liquidation
        if portfolio_value <= 0:
            return RiskCheckResult(
                check_name="leverage_limit",
                passed=False,
                current_value=0.0,
                limit_value=self._max_leverage,
                message="Portfolio value is zero or negative"
            )

        # Calculate current gross and net exposure
        gross_exposure = self._risk_state.gross_exposure
        net_exposure = self._risk_state.net_exposure if hasattr(self._risk_state, 'net_exposure') else gross_exposure

        # Estimate impact of this order
        price = decision.limit_price or 100.0
        order_value = abs(decision.quantity * price)

        # Determine if this adds to or reduces exposure
        symbol = decision.symbol
        current_pos = self._risk_state.positions.get(symbol)
        current_qty = current_pos.quantity if current_pos else 0

        is_buy = decision.action == OrderSide.BUY
        new_qty = current_qty + decision.quantity if is_buy else current_qty - decision.quantity

        # Calculate exposure change
        if (current_qty >= 0 and is_buy) or (current_qty <= 0 and not is_buy):
            # Adding to existing direction - increases exposure
            gross_delta = order_value
        else:
            # Reducing position - decreases exposure
            gross_delta = -min(order_value, abs(current_qty * price))
            if abs(new_qty) > abs(current_qty):
                # Flip and establish new position
                gross_delta = abs(new_qty - current_qty) * price - abs(current_qty * price)

        projected_gross = gross_exposure + gross_delta

        # Calculate both gross and net leverage
        gross_leverage = projected_gross / portfolio_value
        net_leverage = abs(net_exposure + (order_value if is_buy else -order_value)) / portfolio_value

        # Use the more conservative (higher) of gross leverage and net leverage for limit check
        # But provide netting benefit info in the message
        effective_leverage = max(gross_leverage, net_leverage)
        netting_benefit = gross_leverage - net_leverage if gross_leverage > net_leverage else 0

        if effective_leverage > self._max_leverage:
            message = f"Leverage would exceed {self._max_leverage}x limit (gross={gross_leverage:.2f}x, net={net_leverage:.2f}x)"
        elif netting_benefit > 0.1:
            message = f"Netting benefit: {netting_benefit:.2f}x reduction in effective leverage"
        else:
            message = ""

        return RiskCheckResult(
            check_name="leverage_limit",
            passed=effective_leverage <= self._max_leverage,
            current_value=effective_leverage,
            limit_value=self._max_leverage,
            message=message
        )

    async def _check_var_limit(self, decision: DecisionEvent) -> RiskCheckResult:
        """Check VaR limit (2% at 95% confidence) using VaRCalculator."""
        current_var = self._risk_state.var_95
        portfolio_value = self._risk_state.net_liquidation

        # Get order value
        price = decision.limit_price or 100.0
        order_value = decision.quantity * price
        if decision.action and decision.action.value == "sell":
            order_value = -order_value

        # Use VaRCalculator if available for proper incremental VaR
        if self._var_calculator is not None and portfolio_value > 0:
            try:
                # Get current positions as values dict
                positions = {
                    sym: pos.market_value
                    for sym, pos in self._risk_state.positions.items()
                }

                # Calculate incremental VaR
                incr_result = self._var_calculator.calculate_incremental_var(
                    positions=positions,
                    portfolio_value=portfolio_value,
                    new_position_symbol=decision.symbol,
                    new_position_value=order_value,
                )

                # Projected VaR as percentage of portfolio
                projected_var_pct = incr_result.new_var / portfolio_value if portfolio_value > 0 else 0

                return RiskCheckResult(
                    check_name="var_limit",
                    passed=projected_var_pct <= self._max_var_pct,
                    current_value=projected_var_pct,
                    limit_value=self._max_var_pct,
                    message=f"VaR would exceed {self._max_var_pct*100:.1f}% limit (incremental: {incr_result.incremental_var:.2f})" if projected_var_pct > self._max_var_pct else ""
                )

            except Exception as e:
                logger.warning(f"VaR calculation failed, using simplified method: {e}")

        # Fallback to simplified method if VaRCalculator not available
        # Use historical volatility if available, otherwise default
        estimated_vol = self._risk_state.portfolio_volatility if hasattr(self._risk_state, 'portfolio_volatility') and self._risk_state.portfolio_volatility > 0 else 0.02
        marginal_var = (abs(order_value) / portfolio_value) * estimated_vol * 1.645 if portfolio_value > 0 else 0

        projected_var = current_var + marginal_var

        return RiskCheckResult(
            check_name="var_limit",
            passed=projected_var <= self._max_var_pct,
            current_value=projected_var,
            limit_value=self._max_var_pct,
            message=f"VaR would exceed {self._max_var_pct*100:.1f}% limit" if projected_var > self._max_var_pct else ""
        )

    async def _check_daily_loss_limit(self) -> RiskCheckResult:
        """Check daily loss limit (-3% triggers kill-switch)."""
        daily_pnl_pct = self._risk_state.daily_pnl_pct

        breached = daily_pnl_pct < -self._max_daily_loss_pct

        return RiskCheckResult(
            check_name="daily_loss_limit",
            passed=not breached,
            current_value=daily_pnl_pct,
            limit_value=-self._max_daily_loss_pct,
            message=f"KILL-SWITCH: Daily loss {daily_pnl_pct*100:.2f}% exceeds limit" if breached else ""
        )

    async def _check_drawdown_limit(self) -> RiskCheckResult:
        """
        Check drawdown with tiered response.

        Tiers:
        - NORMAL (<5%): Normal trading
        - WARNING (5-7.5%): Log warnings, continue trading
        - REDUCE (7.5-10%): Reduce position sizes by configured factor
        - HALT (>10%): Trigger kill-switch

        Returns:
            RiskCheckResult with appropriate pass/fail based on tier
        """
        current_dd = self._risk_state.current_drawdown_pct

        # Determine current drawdown level
        previous_level = self._current_drawdown_level

        if current_dd >= self._drawdown_halt_pct:
            new_level = DrawdownLevel.HALT
        elif current_dd >= self._drawdown_reduce_pct:
            new_level = DrawdownLevel.REDUCE
        elif current_dd >= self._drawdown_warning_pct:
            new_level = DrawdownLevel.WARNING
        else:
            new_level = DrawdownLevel.NORMAL

        # Track level changes
        if new_level != previous_level:
            self._current_drawdown_level = new_level
            self._drawdown_level_time = datetime.now(timezone.utc)
            await self._handle_drawdown_level_change(previous_level, new_level, current_dd)

        # Return result based on level
        if new_level == DrawdownLevel.HALT:
            return RiskCheckResult(
                check_name="max_drawdown",
                passed=False,
                current_value=current_dd,
                limit_value=self._drawdown_halt_pct,
                message=f"KILL-SWITCH: Drawdown {current_dd*100:.2f}% exceeds {self._drawdown_halt_pct*100:.1f}% halt threshold"
            )

        # REDUCE and WARNING levels pass but with context
        if new_level == DrawdownLevel.REDUCE:
            return RiskCheckResult(
                check_name="max_drawdown",
                passed=True,
                current_value=current_dd,
                limit_value=self._drawdown_halt_pct,
                message=f"REDUCE MODE: Drawdown {current_dd*100:.2f}% - position sizes reduced by {(1-self._drawdown_position_reduction)*100:.0f}%"
            )

        if new_level == DrawdownLevel.WARNING:
            return RiskCheckResult(
                check_name="max_drawdown",
                passed=True,
                current_value=current_dd,
                limit_value=self._drawdown_halt_pct,
                message=f"WARNING: Drawdown {current_dd*100:.2f}% approaching reduce threshold ({self._drawdown_reduce_pct*100:.1f}%)"
            )

        # Normal level
        return RiskCheckResult(
            check_name="max_drawdown",
            passed=True,
            current_value=current_dd,
            limit_value=self._drawdown_halt_pct,
        )

    async def _handle_drawdown_level_change(
        self,
        previous: DrawdownLevel,
        current: DrawdownLevel,
        drawdown_pct: float
    ) -> None:
        """
        Handle transitions between drawdown levels.

        Publishes appropriate alerts and logs for audit trail.
        """
        level_order = [DrawdownLevel.NORMAL, DrawdownLevel.WARNING, DrawdownLevel.REDUCE, DrawdownLevel.HALT]
        is_escalation = level_order.index(current) > level_order.index(previous)

        if current == DrawdownLevel.WARNING:
            severity = RiskAlertSeverity.WARNING
            message = f"Drawdown WARNING: {drawdown_pct*100:.2f}% (threshold: {self._drawdown_warning_pct*100:.1f}%)"
            logger.warning(message)
        elif current == DrawdownLevel.REDUCE:
            severity = RiskAlertSeverity.HIGH
            message = f"Drawdown REDUCE MODE: {drawdown_pct*100:.2f}% - reducing position sizes by {(1-self._drawdown_position_reduction)*100:.0f}%"
            logger.warning(message)
        elif current == DrawdownLevel.HALT:
            severity = RiskAlertSeverity.EMERGENCY
            message = f"Drawdown HALT: {drawdown_pct*100:.2f}% exceeds {self._drawdown_halt_pct*100:.1f}% - KILL SWITCH ACTIVATED"
            logger.critical(message)
        elif current == DrawdownLevel.NORMAL and is_escalation is False:
            # Recovery
            severity = RiskAlertSeverity.LOW
            message = f"Drawdown recovered to NORMAL: {drawdown_pct*100:.2f}%"
            logger.info(message)
        else:
            return  # No alert needed

        # Publish alert
        alert = RiskAlertEvent(
            source_agent=self.name,
            severity=severity,
            alert_type="drawdown_level_change",
            message=message,
            current_value=drawdown_pct,
            threshold_value=self._drawdown_halt_pct,
            halt_trading=(current == DrawdownLevel.HALT),
        )
        await self._event_bus.publish(alert)

        # Log for audit
        self._audit_logger.log_risk_alert(
            agent_name=self.name,
            alert_type="drawdown_level_change",
            severity=severity.value,
            message=message,
            current_value=drawdown_pct,
            threshold_value=self._drawdown_halt_pct,
            halt_trading=(current == DrawdownLevel.HALT),
        )

    def get_drawdown_level(self) -> DrawdownLevel:
        """Get current drawdown level for external queries."""
        return self._current_drawdown_level

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on drawdown level.

        Returns:
            1.0 for NORMAL/WARNING, configured reduction for REDUCE, 0.0 for HALT
        """
        if self._current_drawdown_level == DrawdownLevel.REDUCE:
            return self._drawdown_position_reduction
        elif self._current_drawdown_level == DrawdownLevel.HALT:
            return 0.0
        return 1.0

    def _check_rate_limit(self) -> RiskCheckResult:
        """Check order rate limit (anti-HFT: 10 orders/min)."""
        now = datetime.now(timezone.utc)
        one_minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self._risk_state.orders_this_minute = [
            t for t in self._risk_state.orders_this_minute if t > one_minute_ago
        ]

        orders_count = len(self._risk_state.orders_this_minute)
        breached = orders_count >= self._max_orders_per_minute

        return RiskCheckResult(
            check_name="rate_limit",
            passed=not breached,
            current_value=float(orders_count),
            limit_value=float(self._max_orders_per_minute),
            message=f"Rate limit: {orders_count}/{self._max_orders_per_minute} orders in last minute" if breached else ""
        )

    def _check_order_interval(self) -> RiskCheckResult:
        """Check minimum order interval (anti-HFT: 100ms)."""
        if self._risk_state.last_order_time is None:
            return RiskCheckResult(
                check_name="order_interval",
                passed=True,
                current_value=float('inf'),
                limit_value=float(self._min_order_interval_ms),
            )

        elapsed_ms = (datetime.now(timezone.utc) - self._risk_state.last_order_time).total_seconds() * 1000
        breached = elapsed_ms < self._min_order_interval_ms

        return RiskCheckResult(
            check_name="order_interval",
            passed=not breached,
            current_value=elapsed_ms,
            limit_value=float(self._min_order_interval_ms),
            message=f"Order interval {elapsed_ms:.0f}ms < {self._min_order_interval_ms}ms minimum" if breached else ""
        )

    def _get_risk_metrics(self) -> dict[str, float]:
        """Get current risk metrics dictionary."""
        return {
            "net_liquidation": self._risk_state.net_liquidation,
            "daily_pnl_pct": self._risk_state.daily_pnl_pct,
            "current_drawdown_pct": self._risk_state.current_drawdown_pct,
            "max_drawdown_pct": self._risk_state.max_drawdown_pct,
            "var_95_pct": self._risk_state.var_95,
            "leverage": self._risk_state.leverage,
            "gross_exposure": self._risk_state.gross_exposure,
            "orders_today": float(self._risk_state.orders_today),
        }

    async def _refresh_portfolio_state(self) -> None:
        """Refresh portfolio state from broker."""
        if not self._broker:
            return

        try:
            portfolio = await self._broker.get_portfolio_state()

            self._risk_state.timestamp = datetime.now(timezone.utc)
            self._risk_state.net_liquidation = portfolio.net_liquidation
            self._risk_state.total_cash = portfolio.total_cash
            self._risk_state.daily_pnl = portfolio.daily_pnl

            # Calculate daily P&L percentage
            if portfolio.net_liquidation > 0:
                self._risk_state.daily_pnl_pct = portfolio.daily_pnl / portfolio.net_liquidation

            # Update peak and drawdown
            if portfolio.net_liquidation > self._risk_state.peak_equity:
                self._risk_state.peak_equity = portfolio.net_liquidation

            if self._risk_state.peak_equity > 0:
                self._risk_state.current_drawdown_pct = (
                    (self._risk_state.peak_equity - portfolio.net_liquidation)
                    / self._risk_state.peak_equity
                )
                self._risk_state.max_drawdown_pct = max(
                    self._risk_state.max_drawdown_pct,
                    self._risk_state.current_drawdown_pct
                )

            # Update positions
            self._risk_state.positions = {}
            self._risk_state.gross_exposure = 0.0
            self._risk_state.sector_exposure = {}

            for symbol, pos in portfolio.positions.items():
                sector = self._sector_map.get(symbol, "unknown")
                weight = pos.market_value / portfolio.net_liquidation if portfolio.net_liquidation > 0 else 0

                self._risk_state.positions[symbol] = PositionInfo(
                    symbol=symbol,
                    quantity=pos.quantity,
                    avg_cost=pos.avg_cost,
                    market_value=pos.market_value,
                    unrealized_pnl=pos.unrealized_pnl,
                    weight_pct=weight,
                    sector=sector
                )

                self._risk_state.gross_exposure += abs(pos.market_value)
                self._risk_state.sector_exposure[sector] = (
                    self._risk_state.sector_exposure.get(sector, 0.0) + weight
                )

            # Calculate leverage
            if portfolio.net_liquidation > 0:
                self._risk_state.leverage = self._risk_state.gross_exposure / portfolio.net_liquidation

            # Update drawdown recovery tracking (#R11)
            self._update_drawdown_recovery_state(portfolio.net_liquidation)

            # Check for intraday VaR recalculation (#R2)
            await self._maybe_recalculate_var()

            # Check for intraday margin update (#R10)
            await self.refresh_margin_from_broker()

            # Mark successful refresh (ERR-006)
            self._last_successful_refresh = datetime.now(timezone.utc)

        except Exception as e:
            # Broad catch - portfolio refresh is periodic, system continues on failure
            logger.exception(f"Failed to refresh portfolio state: {e}")
            # Check if data is too stale (ERR-006)
            await self._check_stale_data_kill_switch()

    async def _check_stale_data_kill_switch(self) -> None:
        """
        Check if portfolio data is too stale and activate kill switch if needed (ERR-006).

        If the last successful portfolio refresh was more than max_stale_data_seconds ago,
        activates the kill switch to prevent trading with outdated risk data.
        """
        if not self._stale_data_kill_switch_enabled:
            return

        if self._last_successful_refresh is None:
            # No successful refresh yet - allow initial startup grace period
            return

        now = datetime.now(timezone.utc)
        staleness = (now - self._last_successful_refresh).total_seconds()

        if staleness > self._max_stale_data_seconds:
            logger.error(
                f"Portfolio data is stale ({staleness:.1f}s > {self._max_stale_data_seconds}s). "
                f"Activating kill switch to prevent trading with outdated risk data. "
                f"Last successful refresh: {self._last_successful_refresh.isoformat()}"
            )
            await self._activate_kill_switch(
                reason=KillSwitchReason.CONNECTIVITY_LOSS,
                action=KillSwitchAction.HALT_NEW_ORDERS,
                triggered_by=f"stale_portfolio_data_{staleness:.0f}s"
            )

    def get_data_staleness_seconds(self) -> float | None:
        """Get current portfolio data staleness in seconds (ERR-006)."""
        if self._last_successful_refresh is None:
            return None
        return (datetime.now(timezone.utc) - self._last_successful_refresh).total_seconds()

    def _calculate_var(self) -> None:
        """
        Calculate Value at Risk (VaR) using the parametric (variance-covariance) method.

        VaR represents the maximum expected loss over a given time horizon
        at a specified confidence level. For example, 95% 1-day VaR of 2%
        means there's a 5% chance of losing more than 2% in one day.

        Parametric VaR Formula:
            VaR_alpha = -(mu - z_alpha * sigma)

        Where:
            mu = mean daily return
            sigma = standard deviation of daily returns
            z_alpha = z-score for confidence level (1.645 for 95%, 2.326 for 99%)

        The formula assumes normally distributed returns. The negative sign
        converts the return threshold to a positive loss number.

        Expected Shortfall (CVaR / Conditional VaR) is also calculated:
            ES = E[Loss | Loss > VaR]
            This is the average of all returns in the left tail beyond VaR.
            ES is a more conservative risk measure than VaR because it
            accounts for the severity of losses beyond the VaR threshold.

        Example:
            If mu=0.05%, sigma=1.5%, then:
            VaR_95 = -(0.0005 - 1.645 * 0.015) = -(-0.0242) = 2.42%

        Note:
            Uses a pre-allocated rolling buffer for performance (PERF-P1-001).
            Requires minimum 20 observations for meaningful calculation.
        """
        if len(self._returns_history) < 20:
            self._risk_state.var_95 = 0.02  # Default 2%
            self._risk_state.var_99 = 0.03  # Default 3%
            return

        # Use pre-allocated rolling buffer for performance (PERF-P1-001)
        # Avoid copying 252 elements on every check
        if self._buffer_filled:
            # Buffer is full, use entire buffer (no slice copy needed)
            returns = self._returns_buffer
        else:
            # Buffer not yet full, use only filled portion
            returns = self._returns_buffer[:self._buffer_idx] if self._buffer_idx > 0 else np.array(self._returns_history[-252:])

        # Calculate return statistics
        mean_return = np.mean(returns)  # mu: expected daily return
        std_return = np.std(returns)  # sigma: daily volatility

        # Parametric VaR: VaR = -(mu - z * sigma)
        # Z-scores: 1.645 for 95% (one-tailed), 2.326 for 99%
        # The negative sign converts to a positive loss percentage
        self._risk_state.var_95 = -(mean_return - 1.645 * std_return)
        self._risk_state.var_99 = -(mean_return - 2.326 * std_return)

        # Expected Shortfall (CVaR): Average loss in the tail beyond VaR
        # Find the 5th percentile (for 95% VaR) and average all returns below it
        var_95_threshold = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_95_threshold]
        self._risk_state.expected_shortfall = -np.mean(tail_returns) if len(tail_returns) > 0 else self._risk_state.var_95

        # Check CVaR alerts (#R13) - schedule async check
        asyncio.create_task(self._check_cvar_alerts())

    # =========================================================================
    # CVAR THRESHOLD ALERTS (#R13)
    # =========================================================================

    async def _check_cvar_alerts(self) -> None:
        """
        Check CVaR (Expected Shortfall) against thresholds and issue alerts (#R13).

        Three alert levels:
        - WARNING: CVaR exceeds warning threshold (default 2.5%)
        - CRITICAL: CVaR exceeds critical threshold (default 4%)
        - HALT: CVaR exceeds halt threshold (default 5%), triggers trading halt
        """
        if not self._cvar_alerts_enabled:
            return

        cvar = self._risk_state.expected_shortfall
        if cvar is None or cvar == 0:
            return

        now = datetime.now(timezone.utc)

        # Check cooldown to avoid alert spam
        if self._last_cvar_alert_time:
            elapsed = (now - self._last_cvar_alert_time).total_seconds()
            if elapsed < self._cvar_alert_cooldown_seconds:
                return

        # Determine alert level
        new_alert_level = None
        halt_trading = False

        if cvar >= self._cvar_halt_pct:
            new_alert_level = "HALT"
            halt_trading = True
        elif cvar >= self._cvar_critical_pct:
            new_alert_level = "CRITICAL"
        elif cvar >= self._cvar_warning_pct:
            new_alert_level = "WARNING"
        else:
            # Below all thresholds - clear previous alert
            if self._last_cvar_alert_level is not None:
                logger.info(f"CVaR returned to normal ({cvar*100:.2f}%)")
                self._last_cvar_alert_level = None
            return

        # Only alert if level changed or is critical/halt
        if new_alert_level == self._last_cvar_alert_level and new_alert_level == "WARNING":
            return  # Don't repeat warnings

        self._last_cvar_alert_level = new_alert_level
        self._last_cvar_alert_time = now

        # Log alert
        logger.warning(
            f"CVAR ALERT [{new_alert_level}]: Expected Shortfall = {cvar*100:.2f}% "
            f"(warning={self._cvar_warning_pct*100:.1f}%, critical={self._cvar_critical_pct*100:.1f}%, "
            f"halt={self._cvar_halt_pct*100:.1f}%)"
        )

        # Audit log
        self._audit_logger.log_risk_alert(
            agent_name=self.agent_id,
            alert_type="CVaR_THRESHOLD",
            severity=new_alert_level,
            message=f"CVaR of {cvar*100:.2f}% exceeds {new_alert_level.lower()} threshold",
            current_value=cvar,
            threshold_value=self._cvar_warning_pct if new_alert_level == "WARNING" else (
                self._cvar_critical_pct if new_alert_level == "CRITICAL" else self._cvar_halt_pct
            ),
            halt_trading=halt_trading,
        )

        # Publish alert event
        alert_event = RiskAlertEvent(
            source_agent=self.agent_id,
            alert_type=f"CVaR_{new_alert_level}",
            severity=new_alert_level.lower(),
            message=f"Expected Shortfall (CVaR) = {cvar*100:.2f}%",
            current_value=cvar,
            threshold_value=self._cvar_halt_pct if halt_trading else (
                self._cvar_critical_pct if new_alert_level == "CRITICAL" else self._cvar_warning_pct
            ),
            timestamp=now,
        )
        await self._event_bus.publish(alert_event)

        # Trigger halt if necessary
        if halt_trading:
            await self._trigger_cvar_halt(cvar)

    async def _trigger_cvar_halt(self, cvar: float) -> None:
        """
        Trigger trading halt due to excessive CVaR (#R13).

        This is a severe risk event requiring manual intervention.
        """
        reason = f"CVaR ({cvar*100:.2f}%) exceeds halt threshold ({self._cvar_halt_pct*100:.1f}%)"
        logger.critical(f"TRADING HALT: {reason}")

        self._kill_switch_active = True
        self._kill_switch_reason = KillSwitchReason.RISK_LIMIT_BREACH
        self._kill_switch_time = datetime.now(timezone.utc)

        # Publish kill switch event
        kill_event = KillSwitchEvent(
            source_agent=self.agent_id,
            reason=reason,
            cancel_pending_orders=True,
            close_positions=False,  # Don't auto-liquidate, but require human review
        )
        await self._event_bus.publish(kill_event)

    def get_cvar_alert_status(self) -> dict:
        """Get current CVaR alert status (#R13)."""
        return {
            "enabled": self._cvar_alerts_enabled,
            "current_cvar": self._risk_state.expected_shortfall,
            "warning_threshold": self._cvar_warning_pct,
            "critical_threshold": self._cvar_critical_pct,
            "halt_threshold": self._cvar_halt_pct,
            "last_alert_level": self._last_cvar_alert_level,
            "last_alert_time": self._last_cvar_alert_time.isoformat() if self._last_cvar_alert_time else None,
            "status": "OK" if self._last_cvar_alert_level is None else self._last_cvar_alert_level,
        }

    # =========================================================================
    # INTRADAY VAR RECALCULATION (#R2)
    # =========================================================================

    def _should_recalculate_var(self) -> tuple[bool, str]:
        """
        Determine if VaR should be recalculated based on position/exposure changes.

        Triggers recalculation when:
        1. Any single position changes by more than position_change_pct (default 5%)
        2. Total gross exposure changes by more than exposure_change_pct (default 10%)
        3. Maximum interval exceeded (default 1 hour)

        Returns:
            Tuple of (should_recalc, reason)
        """
        if not self._var_recalc_enabled:
            return False, "VaR recalculation disabled"

        now = datetime.now(timezone.utc)

        # Check minimum interval (rate limiting)
        if self._last_var_calc_time:
            elapsed = (now - self._last_var_calc_time).total_seconds()
            if elapsed < self._var_recalc_min_interval_seconds:
                return False, f"Min interval not reached ({elapsed:.0f}s < {self._var_recalc_min_interval_seconds}s)"

        # Check maximum interval (force recalc)
        if self._last_var_calc_time:
            elapsed = (now - self._last_var_calc_time).total_seconds()
            if elapsed >= self._var_recalc_max_interval_seconds:
                return True, f"Max interval exceeded ({elapsed:.0f}s >= {self._var_recalc_max_interval_seconds}s)"
        else:
            return True, "Initial VaR calculation"

        # Check for significant position changes
        current_positions = {
            sym: pos.market_value
            for sym, pos in self._risk_state.positions.items()
        }

        for symbol, current_value in current_positions.items():
            last_value = self._last_var_calc_positions.get(symbol, 0.0)

            if last_value == 0 and current_value != 0:
                # New position opened
                pct_of_portfolio = abs(current_value) / self._risk_state.net_liquidation if self._risk_state.net_liquidation > 0 else 0
                if pct_of_portfolio >= self._var_recalc_position_change_pct:
                    return True, f"New position {symbol} ({pct_of_portfolio*100:.1f}% of portfolio)"

            elif last_value != 0:
                # Position change
                change_pct = abs(current_value - last_value) / abs(last_value)
                if change_pct >= self._var_recalc_position_change_pct:
                    return True, f"Position {symbol} changed {change_pct*100:.1f}%"

        # Check for closed positions
        for symbol in self._last_var_calc_positions:
            if symbol not in current_positions:
                last_value = self._last_var_calc_positions[symbol]
                pct_of_portfolio = abs(last_value) / self._risk_state.net_liquidation if self._risk_state.net_liquidation > 0 else 0
                if pct_of_portfolio >= self._var_recalc_position_change_pct:
                    return True, f"Position {symbol} closed ({pct_of_portfolio*100:.1f}% of portfolio)"

        # Check gross exposure change
        current_exposure = self._risk_state.gross_exposure
        if self._last_var_calc_exposure > 0:
            exposure_change = abs(current_exposure - self._last_var_calc_exposure) / self._last_var_calc_exposure
            if exposure_change >= self._var_recalc_exposure_change_pct:
                return True, f"Gross exposure changed {exposure_change*100:.1f}%"

        return False, "No significant changes"

    async def _maybe_recalculate_var(self) -> None:
        """
        Check if VaR recalculation is needed and perform it.

        This is called after portfolio state refresh to ensure timely risk updates.
        """
        should_recalc, reason = self._should_recalculate_var()

        if not should_recalc:
            return

        logger.info(f"Triggering intraday VaR recalculation: {reason}")

        # Perform VaR calculation
        await self._perform_var_recalculation()

        # Update tracking state
        self._last_var_calc_time = datetime.now(timezone.utc)
        self._last_var_calc_positions = {
            sym: pos.market_value
            for sym, pos in self._risk_state.positions.items()
        }
        self._last_var_calc_exposure = self._risk_state.gross_exposure
        self._var_recalc_count_today += 1

        logger.info(
            f"VaR recalculated: 95%={self._risk_state.var_95*100:.2f}%, "
            f"99%={self._risk_state.var_99*100:.2f}%, "
            f"recalc count today={self._var_recalc_count_today}"
        )

    async def _perform_var_recalculation(self) -> None:
        """
        Perform full VaR recalculation using all available methods.

        Uses enhanced VaRCalculator if available, otherwise falls back to parametric.
        """
        # Use enhanced calculator if available
        if self._var_calculator is not None:
            try:
                positions = {
                    sym: pos.market_value
                    for sym, pos in self._risk_state.positions.items()
                }
                portfolio_value = self._risk_state.net_liquidation

                if portfolio_value > 0 and positions:
                    # Calculate all VaR methods
                    result = self._var_calculator.calculate_all_var(
                        positions=positions,
                        portfolio_value=portfolio_value,
                        confidence_level=0.95,
                    )

                    # Update risk state with all VaR measures
                    self._risk_state.var_95 = result.parametric_var / portfolio_value if result.parametric_var else self._risk_state.var_95
                    self._risk_state.var_parametric = result.parametric_var or 0.0
                    self._risk_state.var_historical = result.historical_var or 0.0
                    self._risk_state.var_monte_carlo = result.monte_carlo_var or 0.0

                    # 99% VaR
                    result_99 = self._var_calculator.calculate_all_var(
                        positions=positions,
                        portfolio_value=portfolio_value,
                        confidence_level=0.99,
                    )
                    self._risk_state.var_99 = result_99.parametric_var / portfolio_value if result_99.parametric_var else self._risk_state.var_99

                    # Expected Shortfall
                    if hasattr(result, 'expected_shortfall') and result.expected_shortfall:
                        self._risk_state.expected_shortfall = result.expected_shortfall / portfolio_value

                    return

            except Exception as e:
                logger.warning(f"Enhanced VaR calculation failed, using parametric: {e}")

        # Fallback to simple parametric VaR
        self._calculate_var()

    def trigger_var_recalculation(self, reason: str = "manual") -> None:
        """
        Manually trigger VaR recalculation.

        Useful for external systems that detect significant market events.
        """
        logger.info(f"Manual VaR recalculation triggered: {reason}")
        asyncio.create_task(self._perform_var_recalculation())

        self._last_var_calc_time = datetime.now(timezone.utc)
        self._last_var_calc_positions = {
            sym: pos.market_value
            for sym, pos in self._risk_state.positions.items()
        }
        self._last_var_calc_exposure = self._risk_state.gross_exposure
        self._var_recalc_count_today += 1

    async def _activate_kill_switch(
        self,
        reason: KillSwitchReason,
        action: KillSwitchAction = KillSwitchAction.HALT_NEW_ORDERS,
        triggered_by: str = "system"
    ) -> None:
        """
        Activate kill-switch - halt all trading (MiFID II RTS 6 compliant).

        Per MiFID II RTS 6 Article 18, algorithmic trading systems must have:
        - Immediate ability to cancel all unexecuted orders
        - Ability to prevent new orders from being sent
        - Clear audit trail of all kill switch activations

        Args:
            reason: The reason for activation
            action: The action to take (halt, cancel, close, shutdown)
            triggered_by: Who/what triggered the kill switch
        """
        activation_time = datetime.now(timezone.utc)

        self._kill_switch_active = True
        self._kill_switch_reason = reason
        self._kill_switch_time = activation_time
        self._kill_switch_action = action

        logger.critical(
            f"KILL-SWITCH ACTIVATED: {reason.value} | Action: {action.value} | "
            f"Triggered by: {triggered_by}"
        )

        # Record in audit trail (MiFID II requirement)
        activation_record = {
            "timestamp": activation_time.isoformat(),
            "reason": reason.value,
            "action": action.value,
            "triggered_by": triggered_by,
            "positions_at_activation": len(self._risk_state.positions),
            "daily_pnl_pct": self._risk_state.daily_pnl_pct,
            "drawdown_pct": self._risk_state.current_drawdown_pct,
        }
        self._kill_switch_activations.append(activation_record)

        # Determine current value and threshold for alert
        if reason == KillSwitchReason.DAILY_LOSS_LIMIT:
            current_value = self._risk_state.daily_pnl_pct
            threshold_value = self._max_daily_loss_pct
        elif reason == KillSwitchReason.MAX_DRAWDOWN:
            current_value = self._risk_state.current_drawdown_pct
            threshold_value = self._max_drawdown_pct
        elif reason == KillSwitchReason.LATENCY_BREACH:
            current_value = max(self._check_latencies) if self._check_latencies else 0
            threshold_value = self._max_latency_ms
        else:
            current_value = 0
            threshold_value = 0

        # Publish emergency alert
        alert = RiskAlertEvent(
            source_agent=self.name,
            severity=RiskAlertSeverity.EMERGENCY,
            alert_type="kill_switch",
            message=f"Trading halted: {reason.value} | Action: {action.value}",
            current_value=current_value,
            threshold_value=threshold_value,
            halt_trading=True,
        )

        await self._event_bus.publish(alert)

        # Execute action
        if action == KillSwitchAction.CANCEL_PENDING:
            await self._cancel_all_pending_orders()
        elif action == KillSwitchAction.CLOSE_POSITIONS:
            if self._auto_close_positions:
                await self._initiate_position_closure()
            else:
                logger.warning("Auto-close disabled, positions remain open")
        elif action == KillSwitchAction.FULL_SHUTDOWN:
            await self._cancel_all_pending_orders()
            if self._auto_close_positions:
                await self._initiate_position_closure()

        # Log for compliance
        self._audit_logger.log_risk_alert(
            agent_name=self.name,
            alert_type="kill_switch",
            severity="emergency",
            message=f"Kill-switch activated: {reason.value}",
            current_value=alert.current_value,
            threshold_value=alert.threshold_value,
            halt_trading=True,
        )

    async def _cancel_all_pending_orders(self) -> None:
        """Cancel all pending orders (MiFID II RTS 6 requirement)."""
        if not self._broker:
            logger.warning("No broker connected, cannot cancel orders")
            return

        try:
            open_orders = self._broker.get_open_orders()
            cancelled_count = 0

            for order in open_orders:
                try:
                    success = await self._broker.cancel_order(order["order_id"])
                    if success:
                        cancelled_count += 1
                except Exception as e:
                    # Critical path - log full trace for kill switch debugging
                    logger.exception(f"Failed to cancel order {order['order_id']}: {e}")

            logger.info(f"Kill-switch: Cancelled {cancelled_count}/{len(open_orders)} pending orders")

        except Exception as e:
            # Kill switch - must log full trace for post-incident analysis
            logger.exception(f"Error during order cancellation: {e}")

    async def _initiate_position_closure(self) -> None:
        """
        Initiate orderly closure of all positions.

        Note: This is a safety feature - actual closure requires
        proper execution through the execution agent.
        """
        logger.warning("Kill-switch: Position closure requested")

        positions_to_close = list(self._risk_state.positions.keys())

        for symbol in positions_to_close:
            pos = self._risk_state.positions[symbol]
            logger.info(
                f"Kill-switch: Position {symbol} ({pos.quantity} @ {pos.avg_cost}) "
                f"marked for closure"
            )

        # Publish closure request event
        alert = RiskAlertEvent(
            source_agent=self.name,
            severity=RiskAlertSeverity.EMERGENCY,
            alert_type="position_closure_requested",
            message=f"Kill-switch: Close all {len(positions_to_close)} positions",
            affected_symbols=tuple(positions_to_close),
            halt_trading=True,
        )

        await self._event_bus.publish(alert)

    def check_latency_breach(self, latency_ms: float) -> bool:
        """
        Check if latency exceeds threshold (MiFID II RTS 6).

        Returns True if latency is acceptable, False if breached.
        """
        if latency_ms > self._max_latency_ms:
            logger.warning(
                f"Latency breach: {latency_ms:.0f}ms > {self._max_latency_ms}ms threshold"
            )
            return False
        return True

    async def activate_kill_switch_manual(
        self,
        authorized_by: str,
        reason: str = "manual",
        action: KillSwitchAction = KillSwitchAction.HALT_NEW_ORDERS
    ) -> bool:
        """
        Manually activate kill switch (requires authorization).

        Args:
            authorized_by: Username of person activating
            reason: Reason for manual activation
            action: Action to take

        Returns:
            True if successfully activated
        """
        if authorized_by not in self._authorized_users:
            logger.error(f"Unauthorized kill switch activation attempt by: {authorized_by}")
            return False

        await self._activate_kill_switch(
            reason=KillSwitchReason.MANUAL,
            action=action,
            triggered_by=authorized_by
        )

        return True

    async def _reject_decision(self, decision: DecisionEvent, reason: str) -> None:
        """Reject a decision and publish rejection event."""
        rejection = ValidatedDecisionEvent(
            source_agent=self.name,
            original_decision_id=decision.event_id,
            approved=False,
            rejection_reason=reason,
            risk_metrics=self._get_risk_metrics(),
            compliance_checks=(),
        )
        await self._event_bus.publish(rejection)
        self._audit_logger.log_event(rejection)

    def deactivate_kill_switch(
        self,
        authorized_by: str,
        second_authorization: str | None = None
    ) -> tuple[bool, str]:
        """
        Deactivate kill-switch (requires authorization, optionally dual).

        MiFID II RTS 6 recommends dual authorization for reactivation
        after emergency halt.

        Args:
            authorized_by: Primary authorizing user
            second_authorization: Second authorizing user (if dual auth required)

        Returns:
            Tuple of (success, message)
        """
        if not self._kill_switch_active:
            return True, "Kill switch not active"

        # Check authorization
        if authorized_by not in self._authorized_users:
            msg = f"Unauthorized deactivation attempt by: {authorized_by}"
            logger.error(msg)
            return False, msg

        # Check dual authorization if required
        if self._require_dual_authorization:
            if not second_authorization:
                msg = "Dual authorization required - provide second_authorization"
                logger.warning(msg)
                return False, msg

            if second_authorization not in self._authorized_users:
                msg = f"Second authorization invalid: {second_authorization}"
                logger.error(msg)
                return False, msg

            if second_authorization == authorized_by:
                msg = "Dual authorization requires two different users"
                logger.error(msg)
                return False, msg

        # Check cooldown period
        if self._kill_switch_time:
            elapsed = (datetime.now(timezone.utc) - self._kill_switch_time).total_seconds() / 60
            if elapsed < self._kill_switch_cooldown_minutes:
                remaining = self._kill_switch_cooldown_minutes - elapsed
                msg = f"Cooldown period active: {remaining:.1f} minutes remaining"
                logger.warning(msg)
                return False, msg

        # Deactivate
        deactivation_time = datetime.now(timezone.utc)
        duration_minutes = (deactivation_time - self._kill_switch_time).total_seconds() / 60 if self._kill_switch_time else 0

        authorizers = [authorized_by]
        if second_authorization:
            authorizers.append(second_authorization)

        logger.warning(
            f"Kill-switch deactivated by: {', '.join(authorizers)} "
            f"(was active for {duration_minutes:.1f} minutes)"
        )

        # Update last activation record
        if self._kill_switch_activations:
            self._kill_switch_activations[-1]["deactivation_time"] = deactivation_time.isoformat()
            self._kill_switch_activations[-1]["deactivated_by"] = authorizers
            self._kill_switch_activations[-1]["duration_minutes"] = duration_minutes

        self._audit_logger.log_risk_alert(
            agent_name=self.name,
            alert_type="kill_switch_deactivated",
            severity="warning",
            message=f"Kill-switch deactivated by {', '.join(authorizers)} after {duration_minutes:.1f} minutes",
            current_value=0,
            threshold_value=0,
            halt_trading=False,
        )

        self._kill_switch_active = False
        self._kill_switch_reason = None
        self._kill_switch_time = None
        self._kill_switch_action = KillSwitchAction.HALT_NEW_ORDERS

        return True, f"Kill switch deactivated successfully after {duration_minutes:.1f} minutes"

    def get_kill_switch_history(self) -> list[dict]:
        """Get audit trail of all kill switch activations."""
        return list(self._kill_switch_activations)

    def add_daily_return(self, return_pct: float) -> None:
        """Add a daily return to history for VaR calculation."""
        self._returns_history.append(return_pct)
        if len(self._returns_history) > self._max_history_days:
            self._returns_history = self._returns_history[-self._max_history_days:]

        # Update pre-allocated rolling buffer (PERF-P1-001)
        self._returns_buffer[self._buffer_idx] = return_pct
        self._buffer_idx = (self._buffer_idx + 1) % 252
        if self._buffer_idx == 0 and len(self._returns_history) >= 252:
            self._buffer_filled = True

    # =========================================================================
    # ENHANCED RISK CHECKS
    # =========================================================================

    async def _check_greeks_limits(self, decision: DecisionEvent) -> RiskCheckResult:
        """
        Check portfolio Greeks limits with staleness validation.

        Limits:
        - max_delta: 500
        - max_gamma: 100
        - max_vega: 50000
        - max_theta: -10000

        Staleness:
        - Warning if Greeks older than 60 seconds
        - Reject options trades if Greeks older than 5 minutes
        """
        greeks = self._risk_state.greeks

        # Check staleness first
        if self._greeks_staleness_check_enabled:
            greeks_age = greeks.age_seconds()

            # Check if data is critically stale
            if greeks_age > self._greeks_staleness_critical_seconds:
                # For options positions, this is critical
                is_options_trade = self._is_options_trade(decision)
                if is_options_trade and self._require_fresh_greeks_for_options:
                    return RiskCheckResult(
                        check_name="greeks_staleness",
                        passed=False,
                        current_value=greeks_age,
                        limit_value=self._greeks_staleness_critical_seconds,
                        message=f"Greeks data critically stale ({greeks_age:.0f}s > {self._greeks_staleness_critical_seconds}s) - options trade requires fresh Greeks"
                    )
                else:
                    logger.warning(
                        f"Greeks data critically stale ({greeks_age:.0f}s) - proceeding with caution for non-options trade"
                    )

            # Warn if data is stale but not critical
            elif greeks_age > self._greeks_staleness_warning_seconds:
                logger.warning(
                    f"Greeks data is stale ({greeks_age:.0f}s > {self._greeks_staleness_warning_seconds}s) - "
                    f"last update: {greeks.last_update.isoformat()}"
                )

        # Check delta
        if abs(greeks.delta) > self._max_delta:
            return RiskCheckResult(
                check_name="greeks_delta",
                passed=False,
                current_value=greeks.delta,
                limit_value=self._max_delta,
                message=f"Portfolio delta {greeks.delta:.0f} exceeds limit {self._max_delta}"
            )

        # Check gamma
        if abs(greeks.gamma) > self._max_gamma:
            return RiskCheckResult(
                check_name="greeks_gamma",
                passed=False,
                current_value=greeks.gamma,
                limit_value=self._max_gamma,
                message=f"Portfolio gamma {greeks.gamma:.0f} exceeds limit {self._max_gamma}"
            )

        # Check vega
        if abs(greeks.vega) > self._max_vega:
            return RiskCheckResult(
                check_name="greeks_vega",
                passed=False,
                current_value=greeks.vega,
                limit_value=self._max_vega,
                message=f"Portfolio vega {greeks.vega:.0f} exceeds limit {self._max_vega}"
            )

        # Check theta (negative is normal for long options)
        if greeks.theta < self._max_theta:
            return RiskCheckResult(
                check_name="greeks_theta",
                passed=False,
                current_value=greeks.theta,
                limit_value=self._max_theta,
                message=f"Portfolio theta {greeks.theta:.0f} exceeds limit {self._max_theta}"
            )

        # Check per-position Greeks limits (#R4)
        if self._enforce_position_greeks:
            position_check = self._check_position_greeks_limits(decision.symbol)
            if not position_check.passed:
                return position_check

        return RiskCheckResult(
            check_name="greeks_limits",
            passed=True,
            current_value=abs(greeks.delta),
            limit_value=self._max_delta,
        )

    def _check_position_greeks_limits(self, symbol: str) -> RiskCheckResult:
        """
        Check Greeks limits for a specific position (#R4).

        Enforces per-position limits:
        - max_position_delta: Maximum delta for single position
        - max_position_gamma: Maximum gamma for single position
        - max_position_vega: Maximum vega for single position

        Returns:
            RiskCheckResult with pass/fail status
        """
        position_greeks = self._risk_state.position_greeks.get(symbol)

        if position_greeks is None:
            # No position Greeks data - pass (non-options or no data)
            return RiskCheckResult(
                check_name="position_greeks",
                passed=True,
                current_value=0,
                limit_value=self._max_position_delta,
            )

        # Check position delta
        if abs(position_greeks.delta) > self._max_position_delta:
            return RiskCheckResult(
                check_name="position_greeks_delta",
                passed=False,
                current_value=position_greeks.delta,
                limit_value=self._max_position_delta,
                message=f"Position {symbol} delta {position_greeks.delta:.0f} exceeds limit {self._max_position_delta}"
            )

        # Check position gamma
        if abs(position_greeks.gamma) > self._max_position_gamma:
            return RiskCheckResult(
                check_name="position_greeks_gamma",
                passed=False,
                current_value=position_greeks.gamma,
                limit_value=self._max_position_gamma,
                message=f"Position {symbol} gamma {position_greeks.gamma:.1f} exceeds limit {self._max_position_gamma}"
            )

        # Check position vega
        if abs(position_greeks.vega) > self._max_position_vega:
            return RiskCheckResult(
                check_name="position_greeks_vega",
                passed=False,
                current_value=position_greeks.vega,
                limit_value=self._max_position_vega,
                message=f"Position {symbol} vega {position_greeks.vega:.0f} exceeds limit {self._max_position_vega}"
            )

        return RiskCheckResult(
            check_name="position_greeks",
            passed=True,
            current_value=abs(position_greeks.delta),
            limit_value=self._max_position_delta,
        )

    def _is_options_trade(self, decision: DecisionEvent) -> bool:
        """
        Check if decision involves options.

        Determines based on symbol pattern or metadata.
        """
        symbol = decision.symbol.upper()

        # Check for common options indicators
        # Options typically have expiry dates, strikes in symbol
        # Examples: "AAPL230120C00150000", "SPY_230120_C_400"
        if any(c in symbol for c in ['C00', 'P00', '_C_', '_P_', '/C/', '/P/']):
            return True

        # Check metadata if available
        if hasattr(decision, 'metadata') and decision.metadata:
            instrument_type = decision.metadata.get('instrument_type', '')
            if instrument_type.lower() in ['option', 'call', 'put']:
                return True

        return False

    async def _check_liquidity(self, decision: DecisionEvent) -> RiskCheckResult:
        """
        Check liquidity risk.

        Ensures:
        - Position < 10% of ADV
        - Order < 5% of ADV
        """
        symbol = decision.symbol
        quantity = decision.quantity

        # Get ADV for this symbol
        adv = self._risk_state.liquidity.avg_daily_volume.get(symbol, 100000)  # Default

        if adv <= 0:
            return RiskCheckResult(
                check_name="liquidity",
                passed=True,
                current_value=0,
                limit_value=self._max_order_pct_of_adv,
                message="No ADV data available"
            )

        # Check order size vs ADV
        order_pct_of_adv = quantity / adv

        if order_pct_of_adv > self._max_order_pct_of_adv:
            return RiskCheckResult(
                check_name="liquidity_order",
                passed=False,
                current_value=order_pct_of_adv,
                limit_value=self._max_order_pct_of_adv,
                message=f"Order {quantity} is {order_pct_of_adv*100:.1f}% of ADV, exceeds {self._max_order_pct_of_adv*100}% limit"
            )

        # Check existing position vs ADV
        current_pos = self._risk_state.positions.get(symbol)
        if current_pos:
            current_qty = abs(current_pos.quantity)
            position_pct_of_adv = current_qty / adv

            if position_pct_of_adv > self._max_position_pct_of_adv:
                return RiskCheckResult(
                    check_name="liquidity_position",
                    passed=False,
                    current_value=position_pct_of_adv,
                    limit_value=self._max_position_pct_of_adv,
                    message=f"Position is {position_pct_of_adv*100:.1f}% of ADV, exceeds {self._max_position_pct_of_adv*100}% limit"
                )

        return RiskCheckResult(
            check_name="liquidity",
            passed=True,
            current_value=order_pct_of_adv,
            limit_value=self._max_order_pct_of_adv,
        )

    async def _check_stress_test_limits(self) -> RiskCheckResult:
        """
        Check that portfolio passes stress test scenarios.

        Ensures worst-case scenario loss does not exceed limit.
        """
        worst_loss_pct = abs(self._risk_state.last_stress_test_pnl) / self._risk_state.net_liquidation \
            if self._risk_state.net_liquidation > 0 else 0

        if worst_loss_pct > self._max_stress_loss_pct:
            return RiskCheckResult(
                check_name="stress_test",
                passed=False,
                current_value=worst_loss_pct,
                limit_value=self._max_stress_loss_pct,
                message=f"Stress test loss {worst_loss_pct*100:.1f}% exceeds {self._max_stress_loss_pct*100}% limit. Scenario: {self._risk_state.worst_stress_scenario}"
            )

        return RiskCheckResult(
            check_name="stress_test",
            passed=True,
            current_value=worst_loss_pct,
            limit_value=self._max_stress_loss_pct,
        )

    async def _check_market_data_staleness(self, decision: DecisionEvent) -> RiskCheckResult:
        """
        Check if market data for the symbol is stale.

        Rejects decisions if market data is critically stale (>30s by default).
        Warns if data is stale (>5s by default).

        This prevents trading on outdated market information which could lead
        to significant slippage or adverse selection.
        """
        if not self._staleness_check_enabled:
            return RiskCheckResult(
                check_name="market_data_staleness",
                passed=True,
                current_value=0.0,
                limit_value=self._staleness_critical_seconds,
                message="Staleness check disabled"
            )

        symbol = decision.symbol

        # Check staleness via broker if available
        if self._broker and hasattr(self._broker, 'check_data_staleness'):
            staleness = self._broker.check_data_staleness(symbol)

            if not staleness.has_data:
                return RiskCheckResult(
                    check_name="market_data_staleness",
                    passed=False,
                    current_value=float('inf'),
                    limit_value=self._staleness_critical_seconds,
                    message=f"No market data available for {symbol}"
                )

            if staleness.is_critical:
                return RiskCheckResult(
                    check_name="market_data_staleness",
                    passed=False,
                    current_value=staleness.age_seconds,
                    limit_value=self._staleness_critical_seconds,
                    message=f"Market data for {symbol} is critically stale ({staleness.age_seconds:.1f}s > {self._staleness_critical_seconds}s)"
                )

            if staleness.is_stale:
                logger.warning(
                    f"Market data for {symbol} is stale ({staleness.age_seconds:.1f}s > {self._staleness_warning_seconds}s) - proceeding with caution"
                )
                return RiskCheckResult(
                    check_name="market_data_staleness",
                    passed=True,
                    current_value=staleness.age_seconds,
                    limit_value=self._staleness_critical_seconds,
                    message=f"WARNING: Market data stale ({staleness.age_seconds:.1f}s)"
                )

            return RiskCheckResult(
                check_name="market_data_staleness",
                passed=True,
                current_value=staleness.age_seconds,
                limit_value=self._staleness_critical_seconds,
            )

        # No broker available - pass with warning
        return RiskCheckResult(
            check_name="market_data_staleness",
            passed=True,
            current_value=0.0,
            limit_value=self._staleness_critical_seconds,
            message="No broker available for staleness check"
        )

    def update_greeks(
        self,
        delta: float,
        gamma: float,
        vega: float,
        theta: float,
        rho: float = 0.0,
        update_time: datetime | None = None
    ) -> None:
        """
        Update portfolio Greeks with timestamp.

        Args:
            delta: Portfolio delta
            gamma: Portfolio gamma
            vega: Portfolio vega
            theta: Portfolio theta
            rho: Portfolio rho (optional)
            update_time: Time of Greeks calculation (defaults to now)
        """
        now = update_time or datetime.now(timezone.utc)

        self._risk_state.greeks.delta = delta
        self._risk_state.greeks.gamma = gamma
        self._risk_state.greeks.vega = vega
        self._risk_state.greeks.theta = theta
        self._risk_state.greeks.rho = rho
        self._risk_state.greeks.last_update = now

        # Calculate time since last update for logging
        greeks_age = self._risk_state.greeks.age_seconds()

        # Publish Greeks update event
        greeks_event = GreeksUpdateEvent(
            source_agent=self.name,
            portfolio_delta=delta,
            portfolio_gamma=gamma,
            portfolio_vega=vega,
            portfolio_theta=theta,
            delta_limit_pct=abs(delta) / self._max_delta * 100 if self._max_delta > 0 else 0,
            gamma_limit_pct=abs(gamma) / self._max_gamma * 100 if self._max_gamma > 0 else 0,
            vega_limit_pct=abs(vega) / self._max_vega * 100 if self._max_vega > 0 else 0,
            any_breach=(
                abs(delta) > self._max_delta or
                abs(gamma) > self._max_gamma or
                abs(vega) > self._max_vega or
                theta < self._max_theta
            ),
        )

        logger.info(
            f"Greeks updated: delta={delta:.0f}, gamma={gamma:.1f}, vega={vega:.0f}, theta={theta:.0f} "
            f"(data_age={greeks_age:.1f}s)"
        )

    def update_position_greeks(
        self,
        symbol: str,
        delta: float,
        gamma: float,
        vega: float,
        theta: float,
        rho: float = 0.0,
        contracts: int = 0,
        update_time: datetime | None = None
    ) -> None:
        """
        Update Greeks for a specific position (#R4).

        Args:
            symbol: Position symbol
            delta: Position delta
            gamma: Position gamma
            vega: Position vega
            theta: Position theta
            rho: Position rho (optional)
            contracts: Number of contracts
            update_time: Time of calculation (defaults to now)
        """
        now = update_time or datetime.now(timezone.utc)

        self._risk_state.position_greeks[symbol] = PositionGreeks(
            symbol=symbol,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            contracts=contracts,
            last_update=now,
        )

        # Check if position exceeds limits and log warning
        if self._enforce_position_greeks:
            if abs(delta) > self._max_position_delta:
                logger.warning(
                    f"Position {symbol} delta {delta:.0f} exceeds limit {self._max_position_delta}"
                )
            if abs(gamma) > self._max_position_gamma:
                logger.warning(
                    f"Position {symbol} gamma {gamma:.1f} exceeds limit {self._max_position_gamma}"
                )
            if abs(vega) > self._max_position_vega:
                logger.warning(
                    f"Position {symbol} vega {vega:.0f} exceeds limit {self._max_position_vega}"
                )

        logger.debug(
            f"Position Greeks updated for {symbol}: delta={delta:.1f}, gamma={gamma:.2f}, "
            f"vega={vega:.0f}, theta={theta:.0f}, contracts={contracts}"
        )

    def get_position_greeks(self, symbol: str) -> PositionGreeks | None:
        """Get Greeks for a specific position (#R4)."""
        return self._risk_state.position_greeks.get(symbol)

    def get_all_position_greeks(self) -> dict[str, PositionGreeks]:
        """Get Greeks for all positions (#R4)."""
        return dict(self._risk_state.position_greeks)

    def clear_position_greeks(self, symbol: str | None = None) -> None:
        """
        Clear position Greeks data (#R4).

        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol:
            self._risk_state.position_greeks.pop(symbol, None)
        else:
            self._risk_state.position_greeks.clear()

    def get_greeks_status(self) -> dict:
        """Get detailed Greeks status including staleness."""
        greeks = self._risk_state.greeks
        age = greeks.age_seconds()

        return {
            "delta": greeks.delta,
            "gamma": greeks.gamma,
            "vega": greeks.vega,
            "theta": greeks.theta,
            "rho": greeks.rho,
            "last_update": greeks.last_update.isoformat(),
            "age_seconds": age,
            "is_stale_warning": age > self._greeks_staleness_warning_seconds,
            "is_stale_critical": age > self._greeks_staleness_critical_seconds,
            "staleness_warning_threshold": self._greeks_staleness_warning_seconds,
            "staleness_critical_threshold": self._greeks_staleness_critical_seconds,
            "limits": {
                "max_delta": self._max_delta,
                "max_gamma": self._max_gamma,
                "max_vega": self._max_vega,
                "max_theta": self._max_theta,
            },
            "utilization_pct": {
                "delta": abs(greeks.delta) / self._max_delta * 100 if self._max_delta > 0 else 0,
                "gamma": abs(greeks.gamma) / self._max_gamma * 100 if self._max_gamma > 0 else 0,
                "vega": abs(greeks.vega) / self._max_vega * 100 if self._max_vega > 0 else 0,
            },
        }

    def update_liquidity_data(self, symbol: str, avg_daily_volume: int) -> None:
        """Update ADV data for a symbol."""
        self._risk_state.liquidity.avg_daily_volume[symbol] = avg_daily_volume

        # Calculate position as % of ADV
        pos = self._risk_state.positions.get(symbol)
        if pos and avg_daily_volume > 0:
            self._risk_state.liquidity.position_pct_of_adv[symbol] = abs(pos.quantity) / avg_daily_volume
            # Estimate liquidation days (assuming 10% of ADV per day)
            self._risk_state.liquidity.estimated_liquidation_days[symbol] = abs(pos.quantity) / (avg_daily_volume * 0.1)

    def update_stress_test_result(self, pnl_impact: float, scenario_name: str) -> None:
        """Update worst stress test result."""
        if pnl_impact < self._risk_state.last_stress_test_pnl:
            self._risk_state.last_stress_test_pnl = pnl_impact
            self._risk_state.worst_stress_scenario = scenario_name

    def set_var_calculator(self, var_calculator) -> None:
        """Set enhanced VaR calculator."""
        self._var_calculator = var_calculator

    def set_stress_tester(self, stress_tester) -> None:
        """Set stress tester."""
        self._stress_tester = stress_tester

    def set_correlation_manager(self, correlation_manager) -> None:
        """Set correlation manager."""
        self._correlation_manager = correlation_manager

    # =========================================================================
    # POSITION AGING ALERTS (P2)
    # =========================================================================

    def record_position_entry(self, symbol: str, entry_time: datetime | None = None) -> None:
        """
        Record when a position was entered for aging tracking (P2).

        Args:
            symbol: Position symbol
            entry_time: Time position was entered (defaults to now)
        """
        self._position_entry_times[symbol] = entry_time or datetime.now(timezone.utc)
        logger.debug(f"Position entry recorded for {symbol}")

    def clear_position_entry(self, symbol: str) -> None:
        """
        Clear position entry time when position is closed (P2).

        Args:
            symbol: Position symbol to clear
        """
        if symbol in self._position_entry_times:
            del self._position_entry_times[symbol]
            logger.debug(f"Position entry cleared for {symbol}")

    def get_position_age_days(self, symbol: str) -> float | None:
        """
        Get the age of a position in days (P2).

        Args:
            symbol: Position symbol

        Returns:
            Age in days, or None if entry time not recorded
        """
        entry_time = self._position_entry_times.get(symbol)
        if entry_time is None:
            return None

        now = datetime.now(timezone.utc)
        age_days = (now - entry_time).total_seconds() / 86400  # 86400 seconds in a day
        return age_days

    async def check_position_aging_alerts(self) -> list[dict]:
        """
        Check all positions for aging alerts (P2).

        Returns:
            List of aging alerts for positions exceeding thresholds
        """
        if not self._position_aging_enabled:
            return []

        alerts = []
        now = datetime.now(timezone.utc)

        for symbol, entry_time in self._position_entry_times.items():
            age_days = (now - entry_time).total_seconds() / 86400

            alert = None
            severity = None

            if age_days >= self._position_aging_max_days:
                severity = "CRITICAL"
                alert = {
                    "symbol": symbol,
                    "age_days": round(age_days, 1),
                    "threshold_days": self._position_aging_max_days,
                    "severity": severity,
                    "action_required": "MANDATORY_REVIEW",
                    "message": f"Position {symbol} held for {age_days:.1f} days - mandatory review required",
                    "entry_time": entry_time.isoformat(),
                    "timestamp": now.isoformat(),
                }
                logger.critical(
                    f"POSITION AGING CRITICAL: {symbol} held for {age_days:.1f} days "
                    f"(max threshold: {self._position_aging_max_days} days)"
                )

            elif age_days >= self._position_aging_critical_days:
                severity = "HIGH"
                alert = {
                    "symbol": symbol,
                    "age_days": round(age_days, 1),
                    "threshold_days": self._position_aging_critical_days,
                    "severity": severity,
                    "action_required": "REVIEW_RECOMMENDED",
                    "message": f"Position {symbol} aging - {age_days:.1f} days (critical threshold: {self._position_aging_critical_days})",
                    "entry_time": entry_time.isoformat(),
                    "timestamp": now.isoformat(),
                }
                logger.warning(
                    f"POSITION AGING HIGH: {symbol} held for {age_days:.1f} days "
                    f"(critical threshold: {self._position_aging_critical_days} days)"
                )

            elif age_days >= self._position_aging_warning_days:
                severity = "MEDIUM"
                alert = {
                    "symbol": symbol,
                    "age_days": round(age_days, 1),
                    "threshold_days": self._position_aging_warning_days,
                    "severity": severity,
                    "action_required": "MONITOR",
                    "message": f"Position {symbol} approaching aging threshold - {age_days:.1f} days",
                    "entry_time": entry_time.isoformat(),
                    "timestamp": now.isoformat(),
                }
                logger.info(
                    f"POSITION AGING WARNING: {symbol} held for {age_days:.1f} days "
                    f"(warning threshold: {self._position_aging_warning_days} days)"
                )

            if alert:
                alerts.append(alert)
                self._position_aging_alerts.append(alert)

                # Publish risk alert event
                alert_event = RiskAlertEvent(
                    source_agent=self.name,
                    severity=RiskAlertSeverity.HIGH if severity == "CRITICAL" else RiskAlertSeverity.WARNING,
                    alert_type="position_aging",
                    message=alert["message"],
                    affected_symbols=(symbol,),
                    current_value=age_days,
                    threshold_value=alert["threshold_days"],
                    halt_trading=False,
                )
                await self._event_bus.publish(alert_event)

        return alerts

    def get_position_aging_summary(self) -> dict:
        """
        Get summary of position aging across the portfolio (P2).

        Returns:
            Summary of position ages and alerts
        """
        now = datetime.now(timezone.utc)

        ages = []
        for symbol, entry_time in self._position_entry_times.items():
            age_days = (now - entry_time).total_seconds() / 86400
            ages.append({
                "symbol": symbol,
                "age_days": round(age_days, 1),
                "entry_time": entry_time.isoformat(),
                "status": (
                    "CRITICAL" if age_days >= self._position_aging_max_days
                    else "HIGH" if age_days >= self._position_aging_critical_days
                    else "WARNING" if age_days >= self._position_aging_warning_days
                    else "OK"
                ),
            })

        # Sort by age (oldest first)
        ages.sort(key=lambda x: x["age_days"], reverse=True)

        critical_count = sum(1 for a in ages if a["status"] == "CRITICAL")
        high_count = sum(1 for a in ages if a["status"] == "HIGH")
        warning_count = sum(1 for a in ages if a["status"] == "WARNING")

        return {
            "total_tracked": len(ages),
            "critical_count": critical_count,
            "high_count": high_count,
            "warning_count": warning_count,
            "avg_age_days": sum(a["age_days"] for a in ages) / len(ages) if ages else 0,
            "max_age_days": max(a["age_days"] for a in ages) if ages else 0,
            "positions": ages,
            "thresholds": {
                "warning_days": self._position_aging_warning_days,
                "critical_days": self._position_aging_critical_days,
                "max_days": self._position_aging_max_days,
            },
            "recent_alerts": self._position_aging_alerts[-10:],
        }

    def get_position_aging_alerts(self) -> list[dict]:
        """Get historical position aging alerts (P2)."""
        return list(self._position_aging_alerts)

    # =========================================================================
    # CORRELATION BREAKDOWN DETECTION (P2)
    # =========================================================================

    def update_historical_correlation(
        self,
        symbol1: str,
        symbol2: str,
        correlation: float
    ) -> None:
        """
        Update baseline historical correlation between two symbols (P2).

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            correlation: Historical correlation coefficient (-1 to 1)
        """
        # Normalize key order for consistent lookup
        key = tuple(sorted([symbol1, symbol2]))
        self._historical_correlations[key] = correlation
        logger.debug(f"Historical correlation updated: {key} = {correlation:.3f}")

    def update_current_correlation(
        self,
        symbol1: str,
        symbol2: str,
        correlation: float
    ) -> None:
        """
        Update current/recent correlation between two symbols (P2).

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            correlation: Current correlation coefficient (-1 to 1)
        """
        key = tuple(sorted([symbol1, symbol2]))
        self._current_correlations[key] = correlation

    def detect_correlation_breakdown(
        self,
        symbol1: str,
        symbol2: str,
        current_correlation: float | None = None
    ) -> dict | None:
        """
        Detect if correlation between two symbols has broken down (P2).

        A correlation breakdown occurs when the current correlation differs
        significantly from the historical baseline, indicating a regime change.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            current_correlation: Current correlation (uses stored if None)

        Returns:
            Breakdown alert dict if detected, None otherwise
        """
        if not self._correlation_monitoring_enabled:
            return None

        key = tuple(sorted([symbol1, symbol2]))

        # Get correlations
        historical = self._historical_correlations.get(key)
        current = current_correlation or self._current_correlations.get(key)

        if historical is None or current is None:
            return None

        # Calculate correlation change
        correlation_change = abs(current - historical)

        if correlation_change >= self._correlation_breakdown_threshold:
            alert = {
                "pair": key,
                "symbol1": symbol1,
                "symbol2": symbol2,
                "historical_correlation": round(historical, 3),
                "current_correlation": round(current, 3),
                "correlation_change": round(correlation_change, 3),
                "threshold": self._correlation_breakdown_threshold,
                "direction": "decreased" if current < historical else "increased",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_impact": self._assess_correlation_risk_impact(historical, current),
            }

            logger.warning(
                f"CORRELATION BREAKDOWN DETECTED: {symbol1}/{symbol2} "
                f"changed from {historical:.3f} to {current:.3f} "
                f"(change: {correlation_change:.3f}, threshold: {self._correlation_breakdown_threshold})"
            )

            self._correlation_breakdown_alerts.append(alert)
            return alert

        return None

    def _assess_correlation_risk_impact(
        self,
        historical: float,
        current: float
    ) -> str:
        """
        Assess the risk impact of a correlation change (P2).

        Args:
            historical: Historical correlation
            current: Current correlation

        Returns:
            Risk impact assessment string
        """
        change = current - historical

        # Correlation going to zero (decorrelation)
        if abs(historical) > 0.5 and abs(current) < 0.3:
            return "HIGH - Previously correlated pair now decorrelated"

        # Correlation reversal
        if historical * current < 0:
            return "CRITICAL - Correlation reversed sign"

        # Strong correlation weakening
        if abs(historical) > 0.7 and abs(current) < 0.5:
            return "HIGH - Strong correlation significantly weakened"

        # Moderate change
        if abs(change) > 0.3:
            return "MEDIUM - Significant correlation shift"

        return "LOW - Notable but manageable change"

    async def check_all_correlation_breakdowns(self) -> list[dict]:
        """
        Check all tracked symbol pairs for correlation breakdowns (P2).

        Returns:
            List of correlation breakdown alerts
        """
        if not self._correlation_monitoring_enabled:
            return []

        alerts = []

        for key in self._historical_correlations:
            symbol1, symbol2 = key
            alert = self.detect_correlation_breakdown(symbol1, symbol2)
            if alert:
                alerts.append(alert)

                # Publish risk alert event
                alert_event = RiskAlertEvent(
                    source_agent=self.name,
                    severity=RiskAlertSeverity.WARNING,
                    alert_type="correlation_breakdown",
                    message=f"Correlation breakdown: {symbol1}/{symbol2}",
                    affected_symbols=(symbol1, symbol2),
                    current_value=alert["current_correlation"],
                    threshold_value=self._correlation_breakdown_threshold,
                    halt_trading=False,
                )
                await self._event_bus.publish(alert_event)

        return alerts

    def get_correlation_status(self) -> dict:
        """
        Get status of correlation monitoring (P2).

        Returns:
            Correlation monitoring status and metrics
        """
        # Calculate breakdown counts
        breakdown_count = 0
        significant_changes = []

        for key in self._historical_correlations:
            historical = self._historical_correlations.get(key)
            current = self._current_correlations.get(key)

            if historical is not None and current is not None:
                change = abs(current - historical)
                if change >= self._correlation_breakdown_threshold:
                    breakdown_count += 1
                    significant_changes.append({
                        "pair": key,
                        "historical": round(historical, 3),
                        "current": round(current, 3),
                        "change": round(change, 3),
                    })

        return {
            "enabled": self._correlation_monitoring_enabled,
            "pairs_tracked": len(self._historical_correlations),
            "pairs_with_current_data": len(self._current_correlations),
            "breakdown_threshold": self._correlation_breakdown_threshold,
            "current_breakdowns": breakdown_count,
            "significant_changes": significant_changes,
            "total_alerts": len(self._correlation_breakdown_alerts),
            "recent_alerts": self._correlation_breakdown_alerts[-5:],
        }

    def get_correlation_breakdown_alerts(self) -> list[dict]:
        """Get historical correlation breakdown alerts (P2)."""
        return list(self._correlation_breakdown_alerts)

    def clear_correlation_data(self) -> None:
        """Clear all correlation data (P2)."""
        self._historical_correlations.clear()
        self._current_correlations.clear()
        logger.info("Correlation data cleared")

    def calculate_correlation_adjusted_risk(
        self,
        strategy_returns: dict[str, list[float]] | None = None
    ) -> dict[str, float]:
        """
        Calculate correlation-adjusted risk contribution for strategies (PM-04).

        Risk parity allocation that accounts for correlations between strategies,
        not treating them as independent. Uses the correlation manager if available,
        otherwise falls back to estimating correlations from returns.

        Args:
            strategy_returns: Optional dict of strategy_name -> list of returns.
                             If not provided, uses internal data if available.

        Returns:
            Dict of strategy_name -> correlation-adjusted risk contribution (0-1)
        """
        if not strategy_returns or len(strategy_returns) < 2:
            # Not enough data for correlation analysis
            logger.debug("Insufficient strategy returns data for correlation-adjusted risk")
            return {}

        strategies = list(strategy_returns.keys())
        n_strategies = len(strategies)

        # Get correlation matrix
        correlation_matrix = np.eye(n_strategies)  # Default to identity (no correlation)

        if self._correlation_manager and hasattr(self._correlation_manager, 'get_correlation_matrix'):
            # Use correlation manager if available
            try:
                corr_data = self._correlation_manager.get_correlation_matrix(strategies)
                if corr_data is not None:
                    correlation_matrix = corr_data
            except Exception as e:
                logger.warning(f"Failed to get correlation matrix from manager: {e}")

        if np.allclose(correlation_matrix, np.eye(n_strategies)):
            # No correlation data from manager, estimate from returns
            try:
                # Build returns matrix
                min_len = min(len(r) for r in strategy_returns.values())
                if min_len >= 20:  # Need at least 20 observations
                    returns_matrix = np.array([
                        strategy_returns[s][-min_len:] for s in strategies
                    ])
                    correlation_matrix = np.corrcoef(returns_matrix)
                    # Handle NaN correlations
                    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
                    np.fill_diagonal(correlation_matrix, 1.0)
            except Exception as e:
                logger.warning(f"Failed to estimate correlations from returns: {e}")

        # Calculate individual strategy volatilities
        volatilities = {}
        for strategy in strategies:
            returns = strategy_returns[strategy]
            if len(returns) >= 2:
                volatilities[strategy] = np.std(returns)
            else:
                volatilities[strategy] = 0.02  # Default 2% vol

        vol_array = np.array([volatilities[s] for s in strategies])

        # Calculate covariance matrix from correlation and volatilities
        # Cov = diag(vol) @ Corr @ diag(vol)
        vol_diag = np.diag(vol_array)
        covariance_matrix = vol_diag @ correlation_matrix @ vol_diag

        # Calculate marginal risk contributions
        # For equal-weighted portfolio as baseline
        weights = np.ones(n_strategies) / n_strategies

        # Portfolio variance = w' * Cov * w
        portfolio_variance = weights @ covariance_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance) if portfolio_variance > 0 else 1e-6

        # Marginal risk contribution: (Cov * w) / portfolio_vol
        marginal_contrib = (covariance_matrix @ weights) / portfolio_vol

        # Risk contribution: w_i * marginal_contrib_i
        risk_contributions = weights * marginal_contrib

        # Normalize to sum to 1
        total_contrib = np.sum(np.abs(risk_contributions))
        if total_contrib > 0:
            normalized_contrib = np.abs(risk_contributions) / total_contrib
        else:
            normalized_contrib = np.ones(n_strategies) / n_strategies

        result = {
            strategies[i]: float(normalized_contrib[i])
            for i in range(n_strategies)
        }

        logger.debug(
            f"Correlation-adjusted risk contributions: {result}, "
            f"avg correlation: {np.mean(correlation_matrix[np.triu_indices(n_strategies, 1)]):.2f}"
        )

        return result

    def get_risk_parity_weights(
        self,
        strategy_returns: dict[str, list[float]],
        target_contributions: dict[str, float] | None = None
    ) -> dict[str, float]:
        """
        Calculate risk parity weights that equalize risk contributions (PM-04).

        Uses correlation-adjusted risk to determine optimal weights where each
        strategy contributes equally to total portfolio risk.

        Args:
            strategy_returns: Dict of strategy_name -> list of returns
            target_contributions: Optional target risk contribution per strategy.
                                 Defaults to equal contribution.

        Returns:
            Dict of strategy_name -> optimal weight (sums to 1)
        """
        if not strategy_returns or len(strategy_returns) < 2:
            # Equal weight fallback
            n = len(strategy_returns) if strategy_returns else 1
            return {s: 1.0 / n for s in strategy_returns} if strategy_returns else {}

        strategies = list(strategy_returns.keys())
        n_strategies = len(strategies)

        # Default to equal risk contribution
        if target_contributions is None:
            target_contributions = {s: 1.0 / n_strategies for s in strategies}

        # Calculate individual volatilities (inverse volatility as starting point)
        volatilities = {}
        for strategy in strategies:
            returns = strategy_returns[strategy]
            if len(returns) >= 2:
                vol = np.std(returns)
                volatilities[strategy] = max(vol, 1e-6)  # Avoid division by zero
            else:
                volatilities[strategy] = 0.02

        # Get correlation-adjusted contributions
        corr_adj_risk = self.calculate_correlation_adjusted_risk(strategy_returns)

        # Start with inverse volatility weights
        inv_vol_weights = {s: 1.0 / volatilities[s] for s in strategies}
        total_inv_vol = sum(inv_vol_weights.values())
        weights = {s: inv_vol_weights[s] / total_inv_vol for s in strategies}

        # Iterative adjustment towards target risk contributions
        # Simple gradient descent approach
        for _ in range(10):  # Limited iterations
            if not corr_adj_risk:
                break

            # Adjust weights based on deviation from target
            for strategy in strategies:
                current_contrib = corr_adj_risk.get(strategy, 1.0 / n_strategies)
                target_contrib = target_contributions.get(strategy, 1.0 / n_strategies)

                # Reduce weight if contributing too much risk, increase if too little
                adjustment = 1.0 + 0.1 * (target_contrib - current_contrib)
                weights[strategy] *= adjustment

            # Renormalize
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {s: w / total_weight for s, w in weights.items()}

            # Recalculate contributions with new weights
            # (simplified - full recalc would need matrix operations)

        logger.debug(f"Risk parity weights: {weights}")
        return weights

    def set_risk_notifier(self, risk_notifier) -> None:
        """Set risk limit breach notifier (#R27)."""
        self._risk_notifier = risk_notifier
        logger.info("RiskLimitBreachNotifier connected to RiskAgent")

    # =========================================================================
    # INTRADAY MARGIN MONITORING (#R10)
    # =========================================================================

    async def update_margin_state(
        self,
        initial_margin: float,
        maintenance_margin: float,
        available_margin: float | None = None
    ) -> None:
        """
        Update margin state for intraday monitoring (#R10).

        Should be called periodically (e.g., every trade or every minute)
        to monitor margin utilization throughout the day.

        Args:
            initial_margin: Required initial margin
            maintenance_margin: Required maintenance margin
            available_margin: Available margin (calculated if not provided)
        """
        now = datetime.now(timezone.utc)
        margin = self._risk_state.margin

        # Update margin values
        margin.initial_margin = initial_margin
        margin.maintenance_margin = maintenance_margin

        if available_margin is not None:
            margin.available_margin = available_margin
        else:
            # Calculate from net liquidation minus maintenance
            margin.available_margin = self._risk_state.net_liquidation - maintenance_margin

        margin.margin_excess = margin.available_margin - maintenance_margin

        # Calculate utilization
        if self._risk_state.net_liquidation > 0:
            margin.margin_utilization_pct = (initial_margin / self._risk_state.net_liquidation) * 100
        else:
            margin.margin_utilization_pct = 100.0

        # Track intraday peak
        if margin.margin_utilization_pct > margin.intraday_peak_utilization:
            margin.intraday_peak_utilization = margin.margin_utilization_pct

        margin.last_margin_check = now
        margin.margin_check_count_today += 1

        # Check for alerts
        await self._check_margin_alerts(margin)

        logger.debug(
            f"Margin updated: utilization={margin.margin_utilization_pct:.1f}%, "
            f"excess=${margin.margin_excess:,.0f}"
        )

    async def _check_margin_alerts(self, margin: MarginState) -> None:
        """Check margin levels and publish alerts (#R10)."""
        if margin.is_margin_call():
            # Critical - margin call territory
            margin.intraday_margin_calls += 1

            alert = RiskAlertEvent(
                source_agent=self.name,
                severity=RiskAlertSeverity.EMERGENCY,
                alert_type="margin_call",
                message=f"MARGIN CALL: Utilization {margin.margin_utilization_pct:.1f}% exceeds {margin.margin_call_pct}%",
                current_value=margin.margin_utilization_pct,
                threshold_value=margin.margin_call_pct,
                halt_trading=True,
            )
            await self._event_bus.publish(alert)

            logger.critical(
                f"MARGIN CALL: Utilization {margin.margin_utilization_pct:.1f}%, "
                f"excess=${margin.margin_excess:,.0f}"
            )

            # Consider triggering kill switch on margin call
            if margin.margin_excess < 0:
                await self._activate_kill_switch(
                    KillSwitchReason.ANOMALY_DETECTED,
                    KillSwitchAction.CANCEL_PENDING,
                    triggered_by="margin_call"
                )

        elif margin.is_critical():
            alert = RiskAlertEvent(
                source_agent=self.name,
                severity=RiskAlertSeverity.HIGH,
                alert_type="margin_critical",
                message=f"Critical margin: Utilization {margin.margin_utilization_pct:.1f}% approaching call level",
                current_value=margin.margin_utilization_pct,
                threshold_value=margin.critical_utilization_pct,
                halt_trading=False,
            )
            await self._event_bus.publish(alert)

            logger.warning(
                f"Margin CRITICAL: Utilization {margin.margin_utilization_pct:.1f}%"
            )

        elif margin.is_warning():
            logger.info(
                f"Margin WARNING: Utilization {margin.margin_utilization_pct:.1f}% above {margin.warning_utilization_pct}%"
            )

    async def refresh_margin_from_broker(self) -> None:
        """
        Refresh margin state from broker (#R10).

        Should be called periodically for real-time margin monitoring.
        """
        if not self._broker:
            return

        try:
            # Get margin info from broker
            if hasattr(self._broker, 'get_margin_state'):
                broker_margin = await self._broker.get_margin_state()
                await self.update_margin_state(
                    initial_margin=broker_margin.initial_margin,
                    maintenance_margin=broker_margin.maintenance_margin,
                    available_margin=broker_margin.available_margin,
                )
            elif hasattr(self._broker, 'get_account_summary'):
                # Fallback to account summary
                summary = await self._broker.get_account_summary()
                if 'InitMarginReq' in summary and 'MaintMarginReq' in summary:
                    await self.update_margin_state(
                        initial_margin=float(summary.get('InitMarginReq', 0)),
                        maintenance_margin=float(summary.get('MaintMarginReq', 0)),
                        available_margin=float(summary.get('AvailableFunds', 0)),
                    )
        except Exception as e:
            # Broker communication may fail - log trace for debugging
            logger.exception(f"Failed to refresh margin from broker: {e}")

    def get_margin_status(self) -> dict:
        """Get current margin monitoring status (#R10)."""
        margin = self._risk_state.margin
        return {
            "initial_margin": margin.initial_margin,
            "maintenance_margin": margin.maintenance_margin,
            "available_margin": margin.available_margin,
            "margin_excess": margin.margin_excess,
            "utilization_pct": margin.margin_utilization_pct,
            "status": (
                "MARGIN_CALL" if margin.is_margin_call()
                else "CRITICAL" if margin.is_critical()
                else "WARNING" if margin.is_warning()
                else "NORMAL"
            ),
            "intraday": {
                "peak_utilization_pct": margin.intraday_peak_utilization,
                "margin_calls_today": margin.intraday_margin_calls,
                "checks_today": margin.margin_check_count_today,
                "last_check": margin.last_margin_check.isoformat() if margin.last_margin_check else None,
            },
            "thresholds": {
                "warning_pct": margin.warning_utilization_pct,
                "critical_pct": margin.critical_utilization_pct,
                "margin_call_pct": margin.margin_call_pct,
            },
        }

    def reset_intraday_margin_tracking(self) -> None:
        """Reset intraday margin tracking (call at start of day)."""
        margin = self._risk_state.margin
        margin.intraday_peak_utilization = margin.margin_utilization_pct
        margin.intraday_margin_calls = 0
        margin.margin_check_count_today = 0

    # =========================================================================
    # DRAWDOWN RECOVERY TIME TRACKING (#R11)
    # =========================================================================

    def _update_drawdown_recovery_state(self, current_equity: float) -> None:
        """
        Update drawdown recovery tracking state (#R11).

        Called whenever portfolio equity changes to track:
        - When drawdowns begin
        - When troughs are reached
        - Recovery progress
        - Time to recover

        Args:
            current_equity: Current portfolio equity
        """
        now = datetime.now(timezone.utc)
        recovery = self._risk_state.drawdown_recovery
        peak_equity = self._risk_state.peak_equity
        current_dd = self._risk_state.current_drawdown_pct

        # Case 1: New drawdown starting (equity dropped below previous peak)
        if current_equity < peak_equity and recovery.drawdown_start_time is None:
            recovery.drawdown_start_time = now
            recovery.drawdown_start_equity = peak_equity
            recovery.drawdown_trough_equity = current_equity
            recovery.drawdown_trough_time = now
            recovery.is_recovering = False
            logger.info(
                f"Drawdown started: peak=${peak_equity:,.0f}, "
                f"current=${current_equity:,.0f} ({current_dd*100:.2f}%)"
            )

        # Case 2: In drawdown, equity still falling
        elif recovery.drawdown_start_time and current_equity < recovery.drawdown_trough_equity:
            recovery.drawdown_trough_equity = current_equity
            recovery.drawdown_trough_time = now
            recovery.is_recovering = False
            logger.debug(f"Drawdown deepening: trough=${current_equity:,.0f}")

        # Case 3: In drawdown, equity recovering
        elif recovery.drawdown_start_time and current_equity > recovery.drawdown_trough_equity:
            if not recovery.is_recovering:
                recovery.recovery_start_time = now
                recovery.is_recovering = True
                logger.info(
                    f"Recovery started from trough ${recovery.drawdown_trough_equity:,.0f}"
                )

        # Case 4: Full recovery (equity back to or above peak)
        if recovery.drawdown_start_time and current_equity >= recovery.drawdown_start_equity:
            # Calculate recovery time
            recovery_time_days = (now - recovery.drawdown_start_time).total_seconds() / 86400
            trough_to_recovery_days = (
                (now - recovery.drawdown_trough_time).total_seconds() / 86400
                if recovery.drawdown_trough_time else recovery_time_days
            )

            # Record historical recovery
            recovery.recovery_times_days.append(recovery_time_days)
            recovery.recovery_count += 1

            # Update averages
            if recovery.recovery_times_days:
                recovery.avg_recovery_time_days = sum(recovery.recovery_times_days) / len(recovery.recovery_times_days)
                recovery.max_recovery_time_days = max(recovery.max_recovery_time_days, recovery_time_days)

            logger.info(
                f"DRAWDOWN RECOVERED: Total time {recovery_time_days:.1f} days, "
                f"recovery phase {trough_to_recovery_days:.1f} days, "
                f"trough=${recovery.drawdown_trough_equity:,.0f}, "
                f"max_dd={((recovery.drawdown_start_equity - recovery.drawdown_trough_equity) / recovery.drawdown_start_equity * 100):.2f}%"
            )

            # Reset state for next drawdown
            recovery.drawdown_start_time = None
            recovery.drawdown_start_equity = 0.0
            recovery.drawdown_trough_equity = 0.0
            recovery.drawdown_trough_time = None
            recovery.recovery_start_time = None
            recovery.is_recovering = False

    def get_drawdown_recovery_status(self) -> dict:
        """
        Get current drawdown recovery status (#R11).

        Returns detailed metrics about current and historical drawdowns.
        """
        recovery = self._risk_state.drawdown_recovery
        now = datetime.now(timezone.utc)

        # Calculate current drawdown duration if in drawdown
        current_duration_days = None
        time_since_trough_days = None
        recovery_progress_pct = None

        if recovery.drawdown_start_time:
            current_duration_days = (now - recovery.drawdown_start_time).total_seconds() / 86400

            if recovery.drawdown_trough_time:
                time_since_trough_days = (now - recovery.drawdown_trough_time).total_seconds() / 86400

            if recovery.is_recovering and recovery.drawdown_start_equity > recovery.drawdown_trough_equity:
                # Progress from trough back to peak
                total_to_recover = recovery.drawdown_start_equity - recovery.drawdown_trough_equity
                recovered_so_far = self._risk_state.net_liquidation - recovery.drawdown_trough_equity
                recovery_progress_pct = max(0, min(100, (recovered_so_far / total_to_recover) * 100))

        # Estimate time to recovery based on historical data
        estimated_recovery_days = None
        if recovery.avg_recovery_time_days > 0 and recovery_progress_pct is not None and recovery_progress_pct > 0:
            # Simple linear extrapolation
            estimated_recovery_days = (100 - recovery_progress_pct) / recovery_progress_pct * (time_since_trough_days or 0)

        return {
            "in_drawdown": recovery.drawdown_start_time is not None,
            "is_recovering": recovery.is_recovering,
            "current": {
                "duration_days": current_duration_days,
                "start_equity": recovery.drawdown_start_equity,
                "trough_equity": recovery.drawdown_trough_equity,
                "trough_drawdown_pct": (
                    (recovery.drawdown_start_equity - recovery.drawdown_trough_equity) / recovery.drawdown_start_equity * 100
                    if recovery.drawdown_start_equity > 0 else 0
                ),
                "time_since_trough_days": time_since_trough_days,
                "recovery_progress_pct": recovery_progress_pct,
                "estimated_remaining_days": estimated_recovery_days,
            } if recovery.drawdown_start_time else None,
            "historical": {
                "total_recoveries": recovery.recovery_count,
                "avg_recovery_time_days": recovery.avg_recovery_time_days,
                "max_recovery_time_days": recovery.max_recovery_time_days,
                "recovery_times": recovery.recovery_times_days[-10:],  # Last 10
            },
        }

    async def run_stress_tests(self) -> None:
        """Run all stress tests and update state."""
        if not self._stress_tester:
            return

        # Get current positions and prices
        positions = {
            s: p.market_value for s, p in self._risk_state.positions.items()
        }
        prices = {
            s: p.market_value / p.quantity if p.quantity != 0 else 0
            for s, p in self._risk_state.positions.items()
        }

        try:
            results = self._stress_tester.run_all_scenarios(
                positions=positions,
                portfolio_value=self._risk_state.net_liquidation,
                prices=prices,
            )

            # Find worst case
            if results:
                worst = min(results, key=lambda r: r.pnl_impact)
                self.update_stress_test_result(worst.pnl_impact, worst.scenario.name)

                logger.info(f"Stress tests completed. Worst case: {worst.scenario.name} ({worst.pnl_impact_pct:.1f}%)")

        except Exception as e:
            # Stress testing failure - log trace for risk model debugging
            logger.exception(f"Stress test failed: {e}")

    def get_status(self) -> dict:
        """Get current risk agent status for monitoring."""
        return {
            "kill_switch_active": self._kill_switch_active,
            "kill_switch_reason": self._kill_switch_reason.value if self._kill_switch_reason else None,
            "risk_state": {
                "net_liquidation": self._risk_state.net_liquidation,
                "daily_pnl_pct": self._risk_state.daily_pnl_pct,
                "drawdown_pct": self._risk_state.current_drawdown_pct,
                "leverage": self._risk_state.leverage,
                "var_95": self._risk_state.var_95,
                "orders_today": self._risk_state.orders_today,
            },
            "drawdown_control": {
                "level": self._current_drawdown_level.value,
                "warning_threshold_pct": self._drawdown_warning_pct * 100,
                "reduce_threshold_pct": self._drawdown_reduce_pct * 100,
                "halt_threshold_pct": self._drawdown_halt_pct * 100,
                "position_size_multiplier": self.get_position_size_multiplier(),
                "level_since": self._drawdown_level_time.isoformat() if self._drawdown_level_time else None,
            },
            # Drawdown recovery tracking (#R11)
            "drawdown_recovery": self.get_drawdown_recovery_status(),
            # Intraday margin monitoring (#R10)
            "margin": self.get_margin_status(),
            "greeks": {
                "delta": self._risk_state.greeks.delta,
                "gamma": self._risk_state.greeks.gamma,
                "vega": self._risk_state.greeks.vega,
                "theta": self._risk_state.greeks.theta,
            },
            "stress_test": {
                "worst_scenario": self._risk_state.worst_stress_scenario,
                "worst_pnl": self._risk_state.last_stress_test_pnl,
            },
            "var_recalculation": {
                "enabled": self._var_recalc_enabled,
                "last_calc_time": self._last_var_calc_time.isoformat() if self._last_var_calc_time else None,
                "recalc_count_today": self._var_recalc_count_today,
                "position_change_threshold_pct": self._var_recalc_position_change_pct * 100,
                "exposure_change_threshold_pct": self._var_recalc_exposure_change_pct * 100,
                "var_parametric": self._risk_state.var_parametric,
                "var_historical": self._risk_state.var_historical,
                "var_monte_carlo": self._risk_state.var_monte_carlo,
            },
            # P2: Position aging alerts
            "position_aging": self.get_position_aging_summary(),
            # P2: Correlation breakdown detection
            "correlation_monitoring": self.get_correlation_status(),
            "avg_check_latency_ms": np.mean(self._check_latencies) if self._check_latencies else 0,
        }

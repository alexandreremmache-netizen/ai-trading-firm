"""
Risk & Compliance Agent
=======================

Validates all trading decisions before execution.
Enforces risk limits and regulatory compliance (EU/AMF).

Responsibility: Risk validation and compliance checking ONLY.
Does NOT make trading decisions or send orders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from core.agent_base import ValidationAgent, AgentConfig
from core.events import (
    Event,
    EventType,
    DecisionEvent,
    ValidatedDecisionEvent,
    RiskAlertEvent,
    RiskAlertSeverity,
    OrderSide,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger
    from core.broker import IBBroker


logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """Current risk state of the portfolio."""
    net_liquidation: float = 1_000_000.0
    total_exposure: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_equity: float = 1_000_000.0
    var_95: float = 0.0
    positions: dict[str, int] = None
    sector_exposure: dict[str, float] = None
    orders_today: int = 0
    last_order_time: datetime = None

    def __post_init__(self):
        self.positions = self.positions or {}
        self.sector_exposure = self.sector_exposure or {}


class RiskComplianceAgent(ValidationAgent):
    """
    Risk & Compliance Agent.

    Validates all trading decisions against:
    1. Position limits
    2. Portfolio risk limits (VaR, drawdown)
    3. Rate limits (anti-HFT)
    4. Regulatory compliance (EU/AMF)

    EVERY decision must pass through this agent.
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
        broker: IBBroker | None = None,
    ):
        super().__init__(config, event_bus, audit_logger)
        self._broker = broker

        # Risk limits from config
        self._max_position_pct = config.parameters.get("max_position_size_pct", 5.0) / 100
        self._max_sector_pct = config.parameters.get("max_sector_exposure_pct", 20.0) / 100
        self._max_daily_loss_pct = config.parameters.get("max_daily_loss_pct", 3.0) / 100
        self._max_drawdown_pct = config.parameters.get("max_drawdown_pct", 10.0) / 100
        self._max_leverage = config.parameters.get("max_leverage", 2.0)
        self._max_var_pct = config.parameters.get("max_portfolio_var_pct", 2.0) / 100

        # Rate limits (anti-HFT)
        self._max_orders_per_minute = config.parameters.get("max_orders_per_minute", 10)
        self._min_order_interval_ms = config.parameters.get("min_order_interval_ms", 100)

        # Compliance settings
        self._jurisdiction = config.parameters.get("jurisdiction", "EU")
        self._banned_instruments = set(config.parameters.get("banned_instruments", []))

        # State
        self._risk_state = RiskState()
        self._trading_halted = False
        self._halt_reason: str | None = None
        self._orders_this_minute: list[datetime] = []

    async def initialize(self) -> None:
        """Initialize risk state from broker."""
        logger.info(f"RiskComplianceAgent initializing (jurisdiction={self._jurisdiction})")

        if self._broker:
            await self._update_portfolio_state()

    def get_subscribed_events(self) -> list[EventType]:
        """Risk agent subscribes to decisions."""
        return [EventType.DECISION]

    async def process_event(self, event: Event) -> None:
        """
        Validate trading decisions.

        Every decision must pass ALL checks before execution.
        """
        if not isinstance(event, DecisionEvent):
            return

        # Check if trading is halted
        if self._trading_halted:
            await self._reject_decision(
                event,
                f"Trading halted: {self._halt_reason}",
            )
            return

        # Run all validation checks
        validation_result = await self._validate_decision(event)

        if validation_result.approved:
            # Publish validated decision
            await self._event_bus.publish(validation_result)
            logger.info(f"Decision {event.event_id} APPROVED for {event.symbol}")
        else:
            logger.warning(
                f"Decision {event.event_id} REJECTED for {event.symbol}: "
                f"{validation_result.rejection_reason}"
            )

        # Log for compliance
        self._audit_logger.log_event(validation_result)

    async def _validate_decision(
        self,
        decision: DecisionEvent,
    ) -> ValidatedDecisionEvent:
        """
        Run all validation checks on a decision.

        Returns ValidatedDecisionEvent with approval status.
        """
        checks_passed = []
        rejection_reason = None
        adjusted_quantity = decision.quantity
        risk_metrics = {}

        # 1. Banned instruments check
        if decision.symbol in self._banned_instruments:
            return self._create_rejection(
                decision,
                f"Instrument {decision.symbol} is banned",
                checks_passed,
            )
        checks_passed.append("banned_instruments")

        # 2. Position size limit
        position_check, adjusted_qty = await self._check_position_limit(decision)
        if not position_check:
            return self._create_rejection(
                decision,
                f"Position size exceeds limit ({self._max_position_pct*100:.1f}% of portfolio)",
                checks_passed,
            )
        adjusted_quantity = adjusted_qty
        checks_passed.append("position_limit")

        # 3. Daily loss limit
        if not await self._check_daily_loss_limit():
            return self._create_rejection(
                decision,
                f"Daily loss limit reached ({self._max_daily_loss_pct*100:.1f}%)",
                checks_passed,
            )
        checks_passed.append("daily_loss_limit")

        # 4. Drawdown limit
        if not await self._check_drawdown_limit():
            return self._create_rejection(
                decision,
                f"Max drawdown reached ({self._max_drawdown_pct*100:.1f}%)",
                checks_passed,
            )
        checks_passed.append("drawdown_limit")

        # 5. Leverage limit
        if not await self._check_leverage_limit(decision):
            return self._create_rejection(
                decision,
                f"Leverage limit exceeded ({self._max_leverage}x)",
                checks_passed,
            )
        checks_passed.append("leverage_limit")

        # 6. Rate limit (anti-HFT)
        if not self._check_rate_limit():
            return self._create_rejection(
                decision,
                f"Rate limit exceeded ({self._max_orders_per_minute}/min)",
                checks_passed,
            )
        checks_passed.append("rate_limit")

        # 7. Order interval (anti-HFT)
        if not self._check_order_interval():
            return self._create_rejection(
                decision,
                f"Order interval too short (min {self._min_order_interval_ms}ms)",
                checks_passed,
            )
        checks_passed.append("order_interval")

        # Calculate risk metrics
        risk_metrics = await self._calculate_risk_metrics(decision)

        # All checks passed
        return ValidatedDecisionEvent(
            source_agent=self.name,
            original_decision_id=decision.event_id,
            approved=True,
            adjusted_quantity=adjusted_quantity if adjusted_quantity != decision.quantity else None,
            rejection_reason=None,
            risk_metrics=risk_metrics,
            compliance_checks=tuple(checks_passed),
        )

    async def _check_position_limit(
        self,
        decision: DecisionEvent,
    ) -> tuple[bool, int]:
        """
        Check if position size is within limits.

        Returns (passed, adjusted_quantity).
        """
        # Get current position
        current_position = self._risk_state.positions.get(decision.symbol, 0)

        # Calculate new position
        quantity_delta = decision.quantity
        if decision.action == OrderSide.SELL:
            quantity_delta = -quantity_delta

        new_position = current_position + quantity_delta

        # Estimate position value (simplified)
        # TODO: Get actual prices
        estimated_price = 100.0  # Placeholder
        position_value = abs(new_position) * estimated_price

        # Check against limit
        max_position_value = self._risk_state.net_liquidation * self._max_position_pct

        if position_value > max_position_value:
            # Adjust quantity to fit limit
            max_quantity = int(max_position_value / estimated_price)
            if max_quantity < 10:
                return False, 0
            return True, max_quantity

        return True, decision.quantity

    async def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been breached."""
        if abs(self._risk_state.daily_pnl_pct) > self._max_daily_loss_pct:
            if self._risk_state.daily_pnl_pct < 0:
                # Halt trading
                await self._halt_trading(
                    f"Daily loss limit breached: {self._risk_state.daily_pnl_pct*100:.2f}%"
                )
                return False
        return True

    async def _check_drawdown_limit(self) -> bool:
        """Check if max drawdown has been breached."""
        if self._risk_state.max_drawdown_pct > self._max_drawdown_pct:
            await self._halt_trading(
                f"Max drawdown breached: {self._risk_state.max_drawdown_pct*100:.2f}%"
            )
            return False
        return True

    async def _check_leverage_limit(self, decision: DecisionEvent) -> bool:
        """Check if leverage limit would be breached."""
        # Simplified leverage calculation
        # TODO: Implement proper leverage calculation
        current_leverage = self._risk_state.total_exposure / self._risk_state.net_liquidation
        return current_leverage < self._max_leverage

    def _check_rate_limit(self) -> bool:
        """Check order rate limit (anti-HFT)."""
        now = datetime.now(timezone.utc)
        one_minute_ago = now - timedelta(minutes=1)

        # Clean old orders
        self._orders_this_minute = [
            t for t in self._orders_this_minute if t > one_minute_ago
        ]

        return len(self._orders_this_minute) < self._max_orders_per_minute

    def _check_order_interval(self) -> bool:
        """Check minimum order interval (anti-HFT)."""
        if self._risk_state.last_order_time is None:
            return True

        elapsed_ms = (
            datetime.now(timezone.utc) - self._risk_state.last_order_time
        ).total_seconds() * 1000

        return elapsed_ms >= self._min_order_interval_ms

    async def _calculate_risk_metrics(
        self,
        decision: DecisionEvent,
    ) -> dict[str, float]:
        """Calculate risk metrics for the decision."""
        # TODO: Implement proper risk calculations
        # - VaR (parametric, historical, Monte Carlo)
        # - Expected shortfall
        # - Beta exposure
        # - Sector concentration

        return {
            "current_drawdown_pct": self._risk_state.max_drawdown_pct,
            "daily_pnl_pct": self._risk_state.daily_pnl_pct,
            "var_95_pct": self._risk_state.var_95,
            "leverage": self._risk_state.total_exposure / max(1, self._risk_state.net_liquidation),
        }

    def _create_rejection(
        self,
        decision: DecisionEvent,
        reason: str,
        checks_passed: list[str],
    ) -> ValidatedDecisionEvent:
        """Create a rejection event."""
        return ValidatedDecisionEvent(
            source_agent=self.name,
            original_decision_id=decision.event_id,
            approved=False,
            adjusted_quantity=None,
            rejection_reason=reason,
            risk_metrics={},
            compliance_checks=tuple(checks_passed),
        )

    async def _reject_decision(
        self,
        decision: DecisionEvent,
        reason: str,
    ) -> None:
        """Reject a decision and publish rejection event."""
        rejection = ValidatedDecisionEvent(
            source_agent=self.name,
            original_decision_id=decision.event_id,
            approved=False,
            rejection_reason=reason,
            compliance_checks=(),
        )
        await self._event_bus.publish(rejection)
        self._audit_logger.log_event(rejection)

    async def _halt_trading(self, reason: str) -> None:
        """Halt all trading."""
        self._trading_halted = True
        self._halt_reason = reason

        alert = RiskAlertEvent(
            source_agent=self.name,
            severity=RiskAlertSeverity.EMERGENCY,
            alert_type="trading_halt",
            message=reason,
            halt_trading=True,
        )

        await self._event_bus.publish(alert)

        self._audit_logger.log_risk_alert(
            agent_name=self.name,
            alert_type="trading_halt",
            severity="emergency",
            message=reason,
            current_value=0,
            threshold_value=0,
            halt_trading=True,
        )

        logger.critical(f"TRADING HALTED: {reason}")

    async def _update_portfolio_state(self) -> None:
        """Update risk state from broker."""
        if not self._broker:
            return

        try:
            portfolio = await self._broker.get_portfolio_state()
            self._risk_state.net_liquidation = portfolio.net_liquidation
            self._risk_state.daily_pnl = portfolio.daily_pnl

            if portfolio.net_liquidation > 0:
                self._risk_state.daily_pnl_pct = (
                    portfolio.daily_pnl / portfolio.net_liquidation
                )

            # Update peak equity and drawdown
            if portfolio.net_liquidation > self._risk_state.peak_equity:
                self._risk_state.peak_equity = portfolio.net_liquidation

            self._risk_state.max_drawdown_pct = (
                (self._risk_state.peak_equity - portfolio.net_liquidation)
                / self._risk_state.peak_equity
            )

        except Exception as e:
            logger.error(f"Failed to update portfolio state: {e}")

    def resume_trading(self) -> None:
        """Resume trading after halt (requires manual intervention)."""
        logger.warning("Trading resumed by manual intervention")
        self._trading_halted = False
        self._halt_reason = None

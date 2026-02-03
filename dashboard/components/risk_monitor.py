"""
Risk Monitor
============

Real-time risk monitoring panel for the trading system dashboard.

Features:
- Real-time risk metrics tracking (VaR, drawdown, leverage)
- Risk limit monitoring with utilization tracking
- Kill switch status and history
- Risk alerts with severity levels
- Export for gauges and charts visualization

Integrates with:
- core/var_calculator.py for VaR calculations
- agents/risk_agent.py for risk state
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.var_calculator import VaRCalculator
    from agents.risk_agent import RiskAgent, RiskState


logger = logging.getLogger(__name__)


class LimitStatus(Enum):
    """Status of a risk limit."""
    OK = "ok"              # Below 70% utilization
    WARNING = "warning"    # 70-90% utilization
    BREACH = "breach"      # Above limit


class AlertSeverity(Enum):
    """Severity levels for risk alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class KillSwitchState(Enum):
    """Kill switch operational state."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    COOLDOWN = "cooldown"


@dataclass
class RiskMetrics:
    """
    Current portfolio risk metrics.

    Contains all key risk indicators for monitoring and display.
    """
    # VaR metrics
    var_95: float = 0.0              # 95% VaR in currency
    var_99: float = 0.0              # 99% VaR in currency
    var_95_pct: float = 0.0          # 95% VaR as percentage of portfolio
    var_99_pct: float = 0.0          # 99% VaR as percentage of portfolio
    expected_shortfall_95: float = 0.0  # CVaR at 95%
    expected_shortfall_99: float = 0.0  # CVaR at 99%

    # Drawdown metrics
    current_drawdown: float = 0.0    # Current drawdown percentage
    max_drawdown: float = 0.0        # Maximum drawdown percentage
    peak_equity: float = 0.0         # Peak portfolio value

    # Exposure metrics
    leverage: float = 0.0            # Current leverage ratio
    gross_exposure: float = 0.0      # Total gross exposure
    net_exposure: float = 0.0        # Net exposure (long - short)

    # Concentration metrics
    position_concentration: float = 0.0  # Largest position as % of portfolio
    sector_exposure: dict[str, float] = field(default_factory=dict)  # Sector -> % exposure
    correlation_risk: float = 0.0    # Average portfolio correlation
    hhi: float = 0.0                 # Herfindahl-Hirschman Index

    # P&L metrics
    daily_pnl: float = 0.0           # Today's P&L in currency
    daily_pnl_pct: float = 0.0       # Today's P&L as percentage
    unrealized_pnl: float = 0.0      # Unrealized P&L
    realized_pnl: float = 0.0        # Realized P&L today

    # Greeks (for options positions)
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    portfolio_theta: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    portfolio_value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization/WebSocket streaming."""
        return {
            "var_95": round(self.var_95, 2),
            "var_99": round(self.var_99, 2),
            "var_95_pct": round(self.var_95_pct * 100, 4),
            "var_99_pct": round(self.var_99_pct * 100, 4),
            "expected_shortfall_95": round(self.expected_shortfall_95, 2),
            "expected_shortfall_99": round(self.expected_shortfall_99, 2),
            "current_drawdown": round(self.current_drawdown * 100, 4),
            "max_drawdown": round(self.max_drawdown * 100, 4),
            "peak_equity": round(self.peak_equity, 2),
            "leverage": round(self.leverage, 4),
            "gross_exposure": round(self.gross_exposure, 2),
            "net_exposure": round(self.net_exposure, 2),
            "position_concentration": round(self.position_concentration * 100, 4),
            "sector_exposure": {k: round(v * 100, 4) for k, v in self.sector_exposure.items()},
            "correlation_risk": round(self.correlation_risk, 4),
            "hhi": round(self.hhi, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct * 100, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "portfolio_delta": round(self.portfolio_delta, 2),
            "portfolio_gamma": round(self.portfolio_gamma, 4),
            "portfolio_vega": round(self.portfolio_vega, 2),
            "portfolio_theta": round(self.portfolio_theta, 2),
            "timestamp": self.timestamp.isoformat(),
            "portfolio_value": round(self.portfolio_value, 2),
        }


@dataclass
class RiskLimit:
    """
    A single risk limit with current utilization tracking.

    Tracks the relationship between current value and limit threshold.
    """
    limit_name: str
    current_value: float
    limit_value: float
    utilization_pct: float = 0.0
    status: LimitStatus = LimitStatus.OK
    description: str = ""
    last_breach_time: datetime | None = None
    breach_count_today: int = 0

    def __post_init__(self):
        """Calculate utilization and status after initialization."""
        self._update_utilization()

    def _update_utilization(self) -> None:
        """Update utilization percentage and status."""
        if self.limit_value > 0:
            self.utilization_pct = (self.current_value / self.limit_value) * 100
        else:
            self.utilization_pct = 0.0 if self.current_value <= 0 else 100.0

        # Determine status based on utilization
        if self.utilization_pct >= 100:
            self.status = LimitStatus.BREACH
        elif self.utilization_pct >= 70:
            self.status = LimitStatus.WARNING
        else:
            self.status = LimitStatus.OK

    def update(self, new_value: float) -> None:
        """Update the current value and recalculate status."""
        old_status = self.status
        self.current_value = new_value
        self._update_utilization()

        # Track breaches
        if self.status == LimitStatus.BREACH and old_status != LimitStatus.BREACH:
            self.last_breach_time = datetime.now(timezone.utc)
            self.breach_count_today += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "limit_name": self.limit_name,
            "current_value": round(self.current_value, 4),
            "limit_value": round(self.limit_value, 4),
            "utilization_pct": round(self.utilization_pct, 2),
            "status": self.status.value,
            "description": self.description,
            "last_breach_time": self.last_breach_time.isoformat() if self.last_breach_time else None,
            "breach_count_today": self.breach_count_today,
        }


@dataclass
class RiskAlert:
    """
    A risk alert event with severity and context.
    """
    alert_id: str
    severity: AlertSeverity
    category: str              # e.g., "var", "drawdown", "leverage", "kill_switch"
    title: str
    message: str
    current_value: float = 0.0
    threshold_value: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None

    def acknowledge(self, user: str) -> None:
        """Mark alert as acknowledged."""
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "category": self.category,
            "title": self.title,
            "message": self.message,
            "current_value": round(self.current_value, 4),
            "threshold_value": round(self.threshold_value, 4),
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


@dataclass
class KillSwitchHistory:
    """
    Record of a kill switch activation event.
    """
    activation_id: str
    activated_at: datetime
    deactivated_at: datetime | None = None
    reason: str = ""
    triggered_by: str = ""
    action_taken: str = ""
    duration_minutes: float = 0.0
    was_manual: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "activation_id": self.activation_id,
            "activated_at": self.activated_at.isoformat(),
            "deactivated_at": self.deactivated_at.isoformat() if self.deactivated_at else None,
            "reason": self.reason,
            "triggered_by": self.triggered_by,
            "action_taken": self.action_taken,
            "duration_minutes": round(self.duration_minutes, 2),
            "was_manual": self.was_manual,
            "notes": self.notes,
        }


@dataclass
class KillSwitchStatus:
    """
    Current kill switch status with full context.
    """
    state: KillSwitchState = KillSwitchState.INACTIVE
    reason: str | None = None
    activated_at: datetime | None = None
    activated_by: str | None = None
    cooldown_ends_at: datetime | None = None
    action: str = "halt_new_orders"
    can_deactivate: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "state": self.state.value,
            "is_active": self.state == KillSwitchState.ACTIVE,
            "reason": self.reason,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "activated_by": self.activated_by,
            "cooldown_ends_at": self.cooldown_ends_at.isoformat() if self.cooldown_ends_at else None,
            "action": self.action,
            "can_deactivate": self.can_deactivate,
            "duration_seconds": (
                (datetime.now(timezone.utc) - self.activated_at).total_seconds()
                if self.activated_at else 0
            ),
        }


class RiskMonitor:
    """
    Real-time risk monitoring panel.

    Provides comprehensive risk monitoring with:
    - Live risk metrics tracking
    - Risk limit utilization monitoring
    - Kill switch status and history
    - Alert management with severity levels
    - Export functions for dashboard gauges and charts

    Usage:
        monitor = RiskMonitor(config)

        # Update from risk agent state
        monitor.update_from_risk_agent(risk_agent)

        # Get current dashboard data
        dashboard = monitor.get_risk_dashboard()

        # Get data for specific visualizations
        gauges = monitor.get_gauge_data()
        charts = monitor.get_chart_data()
    """

    # Alert history buffer size
    MAX_ALERTS = 500
    MAX_KILL_SWITCH_HISTORY = 100

    # Default limit configurations
    DEFAULT_LIMITS = {
        "max_position_size": {
            "limit": 0.05,  # 5% per position
            "description": "Maximum single position size as % of portfolio",
        },
        "max_leverage": {
            "limit": 2.0,
            "description": "Maximum portfolio leverage ratio",
        },
        "max_daily_loss": {
            "limit": 0.03,  # 3% daily loss limit
            "description": "Maximum daily loss as % of portfolio",
        },
        "max_drawdown": {
            "limit": 0.10,  # 10% max drawdown
            "description": "Maximum drawdown from peak equity",
        },
        "max_sector_exposure": {
            "limit": 0.20,  # 20% per sector
            "description": "Maximum exposure to any single sector",
        },
        "max_var_95": {
            "limit": 0.02,  # 2% VaR limit
            "description": "Maximum 1-day 95% VaR as % of portfolio",
        },
        "max_gross_exposure": {
            "limit": 2.0,  # 200% gross exposure
            "description": "Maximum gross exposure as ratio of equity",
        },
    }

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        var_calculator: VaRCalculator | None = None,
    ):
        """
        Initialize the risk monitor.

        Args:
            config: Configuration with risk limits and alert thresholds
            var_calculator: Optional VaR calculator for enhanced metrics
        """
        self._config = config or {}
        self._var_calculator = var_calculator

        # Current metrics
        self._metrics = RiskMetrics()

        # Risk limits with configured values
        self._limits: dict[str, RiskLimit] = {}
        self._initialize_limits()

        # Kill switch state
        self._kill_switch_status = KillSwitchStatus()
        self._kill_switch_history: deque[KillSwitchHistory] = deque(
            maxlen=self.MAX_KILL_SWITCH_HISTORY
        )

        # Alert management
        self._alerts: deque[RiskAlert] = deque(maxlen=self.MAX_ALERTS)
        self._alert_counter = 0

        # Metrics history for charts (last 24 hours, 1-minute intervals)
        self._metrics_history: deque[dict[str, Any]] = deque(maxlen=1440)
        self._last_history_update: datetime | None = None

        # Last update tracking
        self._last_update: datetime | None = None

        logger.info("RiskMonitor initialized")

    def _initialize_limits(self) -> None:
        """Initialize risk limits from configuration."""
        limits_config = self._config.get("limits", {})

        for limit_name, default_config in self.DEFAULT_LIMITS.items():
            # Get configured limit or use default
            configured = limits_config.get(limit_name, {})
            limit_value = configured.get("limit", default_config["limit"])
            description = configured.get("description", default_config["description"])

            self._limits[limit_name] = RiskLimit(
                limit_name=limit_name,
                current_value=0.0,
                limit_value=limit_value,
                description=description,
            )

        # Add any custom limits from config
        for limit_name, limit_config in limits_config.items():
            if limit_name not in self._limits:
                self._limits[limit_name] = RiskLimit(
                    limit_name=limit_name,
                    current_value=0.0,
                    limit_value=limit_config.get("limit", 1.0),
                    description=limit_config.get("description", "Custom limit"),
                )

    def update_metrics(
        self,
        var_95: float | None = None,
        var_99: float | None = None,
        current_drawdown: float | None = None,
        max_drawdown: float | None = None,
        leverage: float | None = None,
        position_concentration: float | None = None,
        sector_exposure: dict[str, float] | None = None,
        correlation_risk: float | None = None,
        daily_pnl: float | None = None,
        daily_pnl_pct: float | None = None,
        portfolio_value: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Update risk metrics with new values.

        Args:
            var_95: 95% Value at Risk
            var_99: 99% Value at Risk
            current_drawdown: Current drawdown as decimal (0.05 = 5%)
            max_drawdown: Maximum drawdown as decimal
            leverage: Current leverage ratio
            position_concentration: Largest position as % of portfolio
            sector_exposure: Dictionary of sector -> exposure %
            correlation_risk: Average portfolio correlation
            daily_pnl: Today's P&L in currency
            daily_pnl_pct: Today's P&L as decimal percentage
            portfolio_value: Current portfolio value
            **kwargs: Additional metrics to update
        """
        now = datetime.now(timezone.utc)

        # Update core metrics if provided
        if var_95 is not None:
            self._metrics.var_95 = var_95
        if var_99 is not None:
            self._metrics.var_99 = var_99
        if current_drawdown is not None:
            self._metrics.current_drawdown = current_drawdown
        if max_drawdown is not None:
            self._metrics.max_drawdown = max_drawdown
        if leverage is not None:
            self._metrics.leverage = leverage
        if position_concentration is not None:
            self._metrics.position_concentration = position_concentration
        if sector_exposure is not None:
            self._metrics.sector_exposure = sector_exposure
        if correlation_risk is not None:
            self._metrics.correlation_risk = correlation_risk
        if daily_pnl is not None:
            self._metrics.daily_pnl = daily_pnl
        if daily_pnl_pct is not None:
            self._metrics.daily_pnl_pct = daily_pnl_pct
        if portfolio_value is not None:
            self._metrics.portfolio_value = portfolio_value
            # Calculate VaR percentages
            if portfolio_value > 0:
                self._metrics.var_95_pct = self._metrics.var_95 / portfolio_value
                self._metrics.var_99_pct = self._metrics.var_99 / portfolio_value

        # Update any additional metrics
        for key, value in kwargs.items():
            if hasattr(self._metrics, key):
                setattr(self._metrics, key, value)

        self._metrics.timestamp = now
        self._last_update = now

        # Update limits
        self._update_limits()

        # Check for alerts
        self._check_for_alerts()

        # Update history for charts (every minute)
        self._update_history()

    def update_from_risk_agent(self, risk_agent: RiskAgent) -> None:
        """
        Update metrics from a RiskAgent instance.

        Args:
            risk_agent: The RiskAgent to extract metrics from
        """
        if not risk_agent:
            return

        # Get risk state from agent
        state = risk_agent._risk_state

        # Update metrics
        self.update_metrics(
            var_95=state.var_95,
            var_99=state.var_99,
            current_drawdown=state.current_drawdown_pct,
            max_drawdown=state.max_drawdown_pct,
            leverage=state.leverage,
            daily_pnl=state.daily_pnl,
            daily_pnl_pct=state.daily_pnl_pct,
            portfolio_value=state.net_liquidation,
            unrealized_pnl=state.unrealized_pnl,
            realized_pnl=state.realized_pnl_today,
            gross_exposure=state.gross_exposure,
            net_exposure=state.net_exposure,
            peak_equity=state.peak_equity,
            portfolio_delta=state.greeks.delta,
            portfolio_gamma=state.greeks.gamma,
            portfolio_vega=state.greeks.vega,
            portfolio_theta=state.greeks.theta,
            sector_exposure=state.sector_exposure,
        )

        # Calculate position concentration
        if state.positions and state.net_liquidation > 0:
            max_position_pct = max(
                abs(pos.market_value) / state.net_liquidation
                for pos in state.positions.values()
            ) if state.positions else 0.0
            self._metrics.position_concentration = max_position_pct

        # Update kill switch status
        self._update_kill_switch_from_agent(risk_agent)

    def _update_limits(self) -> None:
        """Update all risk limit values based on current metrics."""
        metrics = self._metrics

        # Position size limit (use position concentration)
        if "max_position_size" in self._limits:
            self._limits["max_position_size"].update(metrics.position_concentration)

        # Leverage limit
        if "max_leverage" in self._limits:
            self._limits["max_leverage"].update(metrics.leverage)

        # Daily loss limit (use absolute value since it's a loss)
        if "max_daily_loss" in self._limits:
            daily_loss = abs(min(0, metrics.daily_pnl_pct))
            self._limits["max_daily_loss"].update(daily_loss)

        # Drawdown limit
        if "max_drawdown" in self._limits:
            self._limits["max_drawdown"].update(metrics.current_drawdown)

        # Sector exposure limit (use max sector)
        if "max_sector_exposure" in self._limits:
            max_sector = max(metrics.sector_exposure.values()) if metrics.sector_exposure else 0.0
            self._limits["max_sector_exposure"].update(max_sector)

        # VaR limit
        if "max_var_95" in self._limits:
            self._limits["max_var_95"].update(metrics.var_95_pct)

        # Gross exposure limit
        if "max_gross_exposure" in self._limits:
            gross_exposure_ratio = (
                metrics.gross_exposure / metrics.portfolio_value
                if metrics.portfolio_value > 0 else 0.0
            )
            self._limits["max_gross_exposure"].update(gross_exposure_ratio)

    def _check_for_alerts(self) -> None:
        """Check limits and metrics for alert conditions."""
        now = datetime.now(timezone.utc)

        # Check each limit for breach or warning
        for limit_name, limit in self._limits.items():
            if limit.status == LimitStatus.BREACH:
                self._create_alert(
                    severity=AlertSeverity.CRITICAL,
                    category="limit_breach",
                    title=f"{limit_name} Limit Breached",
                    message=f"{limit.description}. Current: {limit.current_value:.4f}, Limit: {limit.limit_value:.4f}",
                    current_value=limit.current_value,
                    threshold_value=limit.limit_value,
                )
            elif limit.status == LimitStatus.WARNING:
                self._create_alert(
                    severity=AlertSeverity.WARNING,
                    category="limit_warning",
                    title=f"{limit_name} Approaching Limit",
                    message=f"{limit.description}. Utilization: {limit.utilization_pct:.1f}%",
                    current_value=limit.current_value,
                    threshold_value=limit.limit_value,
                )

        # Check for specific metric alerts
        metrics = self._metrics

        # High VaR alert
        if metrics.var_95_pct > 0.015:  # > 1.5% VaR
            self._create_alert(
                severity=AlertSeverity.WARNING,
                category="var",
                title="Elevated VaR",
                message=f"95% VaR is {metrics.var_95_pct*100:.2f}% of portfolio",
                current_value=metrics.var_95_pct,
                threshold_value=0.015,
            )

        # Negative correlation warning (tail risk)
        if metrics.correlation_risk < -0.5:
            self._create_alert(
                severity=AlertSeverity.INFO,
                category="correlation",
                title="High Negative Correlation",
                message="Portfolio may have significant tail risk from highly correlated positions",
                current_value=metrics.correlation_risk,
                threshold_value=-0.5,
            )

    def _create_alert(
        self,
        severity: AlertSeverity,
        category: str,
        title: str,
        message: str,
        current_value: float = 0.0,
        threshold_value: float = 0.0,
    ) -> RiskAlert:
        """Create and store a new alert, avoiding duplicates."""
        # Check for recent duplicate alerts (within 5 minutes)
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
        for alert in self._alerts:
            if (
                alert.category == category
                and alert.title == title
                and alert.timestamp > cutoff
            ):
                return alert  # Don't create duplicate

        self._alert_counter += 1
        alert = RiskAlert(
            alert_id=f"RISK-{self._alert_counter:06d}",
            severity=severity,
            category=category,
            title=title,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
        )
        self._alerts.append(alert)

        logger.log(
            logging.WARNING if severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY) else logging.INFO,
            f"Risk alert created: [{severity.value}] {title} - {message}"
        )

        return alert

    def _update_history(self) -> None:
        """Update metrics history for charting."""
        now = datetime.now(timezone.utc)

        # Only update every minute
        if self._last_history_update:
            elapsed = (now - self._last_history_update).total_seconds()
            if elapsed < 60:
                return

        # Store snapshot of key metrics
        snapshot = {
            "timestamp": now.isoformat(),
            "var_95_pct": self._metrics.var_95_pct,
            "var_99_pct": self._metrics.var_99_pct,
            "current_drawdown": self._metrics.current_drawdown,
            "leverage": self._metrics.leverage,
            "daily_pnl_pct": self._metrics.daily_pnl_pct,
            "position_concentration": self._metrics.position_concentration,
            "portfolio_value": self._metrics.portfolio_value,
        }
        self._metrics_history.append(snapshot)
        self._last_history_update = now

    def _update_kill_switch_from_agent(self, risk_agent: RiskAgent) -> None:
        """Update kill switch status from risk agent."""
        was_active = self._kill_switch_status.state == KillSwitchState.ACTIVE
        is_active = risk_agent._kill_switch_active

        if is_active:
            self._kill_switch_status.state = KillSwitchState.ACTIVE
            self._kill_switch_status.reason = (
                risk_agent._kill_switch_reason.value
                if risk_agent._kill_switch_reason else "unknown"
            )
            self._kill_switch_status.activated_at = risk_agent._kill_switch_time
            self._kill_switch_status.action = (
                risk_agent._kill_switch_action.value
                if risk_agent._kill_switch_action else "halt_new_orders"
            )
        else:
            # Check for cooldown period
            if was_active and self._kill_switch_status.activated_at:
                # Record to history
                history_entry = KillSwitchHistory(
                    activation_id=f"KS-{len(self._kill_switch_history)+1:04d}",
                    activated_at=self._kill_switch_status.activated_at,
                    deactivated_at=datetime.now(timezone.utc),
                    reason=self._kill_switch_status.reason or "",
                    triggered_by=self._kill_switch_status.activated_by or "system",
                    action_taken=self._kill_switch_status.action,
                    duration_minutes=(
                        (datetime.now(timezone.utc) - self._kill_switch_status.activated_at).total_seconds() / 60
                    ),
                )
                self._kill_switch_history.append(history_entry)

            self._kill_switch_status.state = KillSwitchState.INACTIVE
            self._kill_switch_status.reason = None
            self._kill_switch_status.activated_at = None

    def check_limits(self) -> dict[str, RiskLimit]:
        """
        Check all risk limits and return their current status.

        Returns:
            Dictionary of limit name to RiskLimit
        """
        return dict(self._limits)

    def get_limit_status(self, limit_name: str) -> RiskLimit | None:
        """
        Get status of a specific limit.

        Args:
            limit_name: Name of the limit to check

        Returns:
            RiskLimit or None if not found
        """
        return self._limits.get(limit_name)

    def get_kill_switch_status(self) -> KillSwitchStatus:
        """
        Get current kill switch status.

        Returns:
            Current KillSwitchStatus
        """
        return self._kill_switch_status

    def get_kill_switch_history(self, limit: int = 10) -> list[KillSwitchHistory]:
        """
        Get recent kill switch activation history.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of KillSwitchHistory records
        """
        history = list(self._kill_switch_history)
        history.reverse()  # Most recent first
        return history[:limit]

    def get_alerts(
        self,
        severity: AlertSeverity | None = None,
        category: str | None = None,
        unacknowledged_only: bool = False,
        limit: int = 50,
    ) -> list[RiskAlert]:
        """
        Get risk alerts with optional filtering.

        Args:
            severity: Filter by severity level
            category: Filter by alert category
            unacknowledged_only: Only return unacknowledged alerts
            limit: Maximum number of alerts to return

        Returns:
            List of RiskAlert objects
        """
        alerts = list(self._alerts)
        alerts.reverse()  # Most recent first

        # Apply filters
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if category:
            alerts = [a for a in alerts if a.category == category]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        return alerts[:limit]

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: ID of the alert to acknowledge
            user: Username acknowledging the alert

        Returns:
            True if alert was found and acknowledged
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledge(user)
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
        return False

    def get_risk_dashboard(self) -> dict[str, Any]:
        """
        Get complete risk dashboard data.

        Returns:
            Dictionary with all risk monitoring data
        """
        return {
            "metrics": self._metrics.to_dict(),
            "limits": {name: limit.to_dict() for name, limit in self._limits.items()},
            "kill_switch": self._kill_switch_status.to_dict(),
            "alerts": {
                "total": len(self._alerts),
                "unacknowledged": sum(1 for a in self._alerts if not a.acknowledged),
                "critical": sum(1 for a in self._alerts if a.severity == AlertSeverity.CRITICAL),
                "recent": [a.to_dict() for a in self.get_alerts(limit=10)],
            },
            "last_update": self._last_update.isoformat() if self._last_update else None,
        }

    def get_gauge_data(self) -> dict[str, dict[str, Any]]:
        """
        Get data formatted for dashboard gauge visualizations.

        Returns:
            Dictionary with gauge configurations and values
        """
        metrics = self._metrics

        return {
            "var_95": {
                "label": "VaR (95%)",
                "value": metrics.var_95_pct * 100,
                "unit": "%",
                "min": 0,
                "max": 5,
                "thresholds": {
                    "warning": 1.5,
                    "critical": 2.0,
                },
                "status": self._get_gauge_status(metrics.var_95_pct, 0.015, 0.02),
            },
            "drawdown": {
                "label": "Drawdown",
                "value": metrics.current_drawdown * 100,
                "unit": "%",
                "min": 0,
                "max": 15,
                "thresholds": {
                    "warning": 5,
                    "critical": 10,
                },
                "status": self._get_gauge_status(metrics.current_drawdown, 0.05, 0.10),
            },
            "leverage": {
                "label": "Leverage",
                "value": metrics.leverage,
                "unit": "x",
                "min": 0,
                "max": 3,
                "thresholds": {
                    "warning": 1.5,
                    "critical": 2.0,
                },
                "status": self._get_gauge_status(metrics.leverage, 1.5, 2.0),
            },
            "daily_pnl": {
                "label": "Daily P&L",
                "value": metrics.daily_pnl_pct * 100,
                "unit": "%",
                "min": -5,
                "max": 5,
                "thresholds": {
                    "warning": -2,
                    "critical": -3,
                },
                "status": self._get_pnl_gauge_status(metrics.daily_pnl_pct),
            },
            "concentration": {
                "label": "Position Concentration",
                "value": metrics.position_concentration * 100,
                "unit": "%",
                "min": 0,
                "max": 10,
                "thresholds": {
                    "warning": 4,
                    "critical": 5,
                },
                "status": self._get_gauge_status(metrics.position_concentration, 0.04, 0.05),
            },
        }

    def _get_gauge_status(self, value: float, warning: float, critical: float) -> str:
        """Determine gauge status based on thresholds."""
        if value >= critical:
            return "critical"
        elif value >= warning:
            return "warning"
        return "ok"

    def _get_pnl_gauge_status(self, pnl_pct: float) -> str:
        """Determine P&L gauge status (negative is bad)."""
        if pnl_pct <= -0.03:
            return "critical"
        elif pnl_pct <= -0.02:
            return "warning"
        elif pnl_pct >= 0:
            return "ok"
        return "ok"

    def get_chart_data(
        self,
        metric: str = "var_95_pct",
        hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get historical data formatted for chart visualization.

        Args:
            metric: Which metric to chart
            hours: Number of hours of history to include

        Returns:
            Dictionary with chart data points
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        data_points = []
        for snapshot in self._metrics_history:
            timestamp = datetime.fromisoformat(snapshot["timestamp"])
            if timestamp >= cutoff:
                value = snapshot.get(metric, 0.0)
                data_points.append({
                    "timestamp": snapshot["timestamp"],
                    "value": value,
                })

        return {
            "metric": metric,
            "hours": hours,
            "data_points": data_points,
            "summary": {
                "min": min(p["value"] for p in data_points) if data_points else 0,
                "max": max(p["value"] for p in data_points) if data_points else 0,
                "avg": sum(p["value"] for p in data_points) / len(data_points) if data_points else 0,
                "current": data_points[-1]["value"] if data_points else 0,
            },
        }

    def get_sector_exposure_chart(self) -> dict[str, Any]:
        """
        Get sector exposure data for pie/bar chart.

        Returns:
            Dictionary with sector exposure data
        """
        exposures = self._metrics.sector_exposure

        return {
            "sectors": [
                {
                    "sector": sector,
                    "exposure_pct": exposure * 100,
                    "status": "critical" if exposure > 0.20 else "warning" if exposure > 0.15 else "ok",
                }
                for sector, exposure in sorted(exposures.items(), key=lambda x: -x[1])
            ],
            "total_sectors": len(exposures),
            "max_exposure": max(exposures.values()) * 100 if exposures else 0,
            "hhi": self._metrics.hhi,
        }

    def get_limit_utilization_bars(self) -> list[dict[str, Any]]:
        """
        Get limit utilization data for bar chart visualization.

        Returns:
            List of limit utilization data points
        """
        bars = []
        for name, limit in self._limits.items():
            bars.append({
                "limit_name": name,
                "utilization_pct": limit.utilization_pct,
                "status": limit.status.value,
                "current_value": limit.current_value,
                "limit_value": limit.limit_value,
            })

        # Sort by utilization (highest first)
        bars.sort(key=lambda x: -x["utilization_pct"])
        return bars

    def get_status_summary(self) -> dict[str, Any]:
        """
        Get a quick status summary for health check endpoints.

        Returns:
            Dictionary with overall status
        """
        breached_limits = [
            name for name, limit in self._limits.items()
            if limit.status == LimitStatus.BREACH
        ]
        warning_limits = [
            name for name, limit in self._limits.items()
            if limit.status == LimitStatus.WARNING
        ]

        # Determine overall status
        if self._kill_switch_status.state == KillSwitchState.ACTIVE:
            overall_status = "halt"
        elif breached_limits:
            overall_status = "critical"
        elif warning_limits:
            overall_status = "warning"
        else:
            overall_status = "ok"

        return {
            "status": overall_status,
            "kill_switch_active": self._kill_switch_status.state == KillSwitchState.ACTIVE,
            "breached_limits": breached_limits,
            "warning_limits": warning_limits,
            "unacknowledged_alerts": sum(1 for a in self._alerts if not a.acknowledged),
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "portfolio_value": self._metrics.portfolio_value,
            "daily_pnl_pct": self._metrics.daily_pnl_pct,
        }

    def reset_daily_metrics(self) -> None:
        """Reset daily counters and metrics (call at start of trading day)."""
        for limit in self._limits.values():
            limit.breach_count_today = 0

        logger.info("RiskMonitor daily metrics reset")

"""
Momentum Crash Protection System
================================

Based on research from Daniel & Moskowitz (2016) on momentum crashes.
Detects conditions that historically precede momentum strategy losses.

Key indicators:
- VIX spike (current vs MA)
- Recent drawdown level
- Correlation spike (diversification breakdown)
- Winner/Loser reversal (Baermann-Wang indicator)
- Drawdown velocity (speed of decline)
- Momentum decay (factor performance deterioration)

Enhanced in PHASE 5 with:
- History tracking for all indicators
- Velocity-aware drawdown response
- Protection mode with decay
- More granular recommendations

PHASE 2 -> PHASE 5 Risk Management Enhancement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class CrashRiskLevel(Enum):
    """Crash risk classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CrashWarning:
    """
    Warning signal from crash protection system.

    Includes risk level, probability estimate, active indicators,
    and recommended action with leverage adjustment.
    """
    level: CrashRiskLevel
    probability: float  # 0-1 crash probability estimate
    indicators: list[str]  # Active warning indicators
    recommended_action: str  # Action description
    leverage_multiplier: float  # 1.0 = normal, 0.5 = half exposure
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "level": self.level.value,
            "probability": self.probability,
            "indicators": self.indicators,
            "recommended_action": self.recommended_action,
            "leverage_multiplier": self.leverage_multiplier,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CrashProtectionConfig:
    """Configuration for crash protection system."""
    # VIX thresholds
    vix_spike_threshold: float = 1.5  # Current/MA ratio to trigger
    vix_extreme_level: float = 40.0  # Absolute VIX level for extreme warning
    vix_ma_period: int = 20  # MA period for VIX comparison

    # Drawdown thresholds
    drawdown_warning_pct: float = 0.05  # 5%
    drawdown_critical_pct: float = 0.10  # 10%

    # Correlation thresholds
    correlation_spike_threshold: float = 0.20  # Change in avg correlation
    correlation_crisis_level: float = 0.70  # Absolute avg correlation

    # Reversal detection
    reversal_threshold: float = 0.05  # Losers outperforming winners by 5%+

    # Leverage adjustments
    min_leverage_multiplier: float = 0.20  # Minimum exposure (20%)

    # Lookback periods
    momentum_lookback_days: int = 252  # 1 year for winner/loser calculation
    short_lookback_days: int = 20  # For recent performance


class MomentumCrashProtection:
    """
    Momentum crash early warning and protection system.

    Based on research findings:
    - Momentum strategies vulnerable to sharp reversals
    - Crashes often preceded by: high volatility, correlation spikes,
      and loser stocks starting to outperform winners
    - VIX spike is strong leading indicator

    Usage:
        protection = MomentumCrashProtection(config)
        warning = protection.evaluate_crash_risk(
            recent_drawdown=-0.08,
            vix_current=35.0,
            vix_ma=18.0,
            correlation_increase=0.25,
            past_winners_return=-0.02,
            past_losers_return=0.05
        )
        if warning.level == CrashRiskLevel.CRITICAL:
            # Reduce exposure by warning.leverage_multiplier
    """

    def __init__(self, config: CrashProtectionConfig | None = None):
        self._config = config or CrashProtectionConfig()
        self._warning_history: list[CrashWarning] = []
        self._last_warning: CrashWarning | None = None

        logger.info(
            f"MomentumCrashProtection initialized: "
            f"vix_spike={self._config.vix_spike_threshold}, "
            f"dd_warning={self._config.drawdown_warning_pct:.0%}"
        )

    def evaluate_crash_risk(
        self,
        recent_drawdown: float,
        vix_current: float | None = None,
        vix_ma: float | None = None,
        correlation_increase: float | None = None,
        avg_correlation: float | None = None,
        past_winners_return: float | None = None,
        past_losers_return: float | None = None,
    ) -> CrashWarning:
        """
        Evaluate momentum crash risk based on multiple indicators.

        Each indicator contributes to a protection score (0-1).
        Higher score = higher crash risk = lower leverage recommended.

        Args:
            recent_drawdown: Current drawdown as negative decimal (e.g., -0.10 = -10%)
            vix_current: Current VIX level
            vix_ma: VIX moving average (typically 20-day)
            correlation_increase: Change in average portfolio correlation
            avg_correlation: Current average pairwise correlation
            past_winners_return: Recent return of past winner stocks
            past_losers_return: Recent return of past loser stocks

        Returns:
            CrashWarning with level, probability, and recommended action
        """
        indicators = []
        protection_score = 0.0

        # 1. VIX Spike Protection (max 0.35 contribution)
        if vix_current is not None:
            # Check absolute extreme level
            if vix_current >= self._config.vix_extreme_level:
                protection_score += 0.35
                indicators.append(f"VIX extreme: {vix_current:.1f} >= {self._config.vix_extreme_level}")
            elif vix_ma is not None and vix_ma > 0:
                # Check spike relative to MA
                vix_ratio = vix_current / vix_ma
                if vix_ratio >= self._config.vix_spike_threshold:
                    spike_severity = min(0.30, (vix_ratio - 1.0) * 0.2)
                    protection_score += spike_severity
                    indicators.append(
                        f"VIX spike: {vix_current:.1f} vs MA {vix_ma:.1f} "
                        f"(ratio={vix_ratio:.2f})"
                    )

        # 2. Drawdown Protection (max 0.30 contribution)
        dd_abs = abs(recent_drawdown)
        if dd_abs >= self._config.drawdown_critical_pct:
            protection_score += 0.30
            indicators.append(f"Critical drawdown: {dd_abs:.1%}")
        elif dd_abs >= self._config.drawdown_warning_pct:
            dd_severity = (dd_abs - self._config.drawdown_warning_pct) / \
                         (self._config.drawdown_critical_pct - self._config.drawdown_warning_pct)
            protection_score += 0.15 + (0.15 * dd_severity)
            indicators.append(f"Elevated drawdown: {dd_abs:.1%}")

        # 3. Correlation Spike Protection (max 0.20 contribution)
        if correlation_increase is not None:
            if correlation_increase > self._config.correlation_spike_threshold:
                corr_severity = min(0.20, correlation_increase * 0.5)
                protection_score += corr_severity
                indicators.append(f"Correlation spike: +{correlation_increase:.1%}")

        if avg_correlation is not None:
            if avg_correlation >= self._config.correlation_crisis_level:
                protection_score += 0.15
                indicators.append(
                    f"High correlation regime: {avg_correlation:.2f} >= "
                    f"{self._config.correlation_crisis_level}"
                )

        # 4. Winner/Loser Reversal Detection (max 0.20 contribution)
        # Baermann-Wang indicator: losers outperforming winners signals momentum crash
        if past_winners_return is not None and past_losers_return is not None:
            reversal = past_losers_return - past_winners_return
            if reversal > self._config.reversal_threshold:
                reversal_severity = min(0.20, reversal * 2.0)
                protection_score += reversal_severity
                indicators.append(
                    f"Momentum reversal: losers {past_losers_return:+.1%} vs "
                    f"winners {past_winners_return:+.1%} (diff={reversal:+.1%})"
                )

        # Cap protection score at 1.0
        protection_score = min(1.0, protection_score)

        # Determine warning level and leverage multiplier
        warning = self._create_warning(protection_score, indicators)

        # Store in history
        self._warning_history.append(warning)
        if len(self._warning_history) > 100:
            self._warning_history = self._warning_history[-100:]
        self._last_warning = warning

        # Log if significant
        if warning.level in (CrashRiskLevel.HIGH, CrashRiskLevel.CRITICAL):
            logger.warning(
                f"CRASH PROTECTION [{warning.level.value.upper()}]: "
                f"score={protection_score:.2f}, leverage={warning.leverage_multiplier:.0%}, "
                f"indicators={warning.indicators}"
            )

        return warning

    def _create_warning(
        self,
        protection_score: float,
        indicators: list[str]
    ) -> CrashWarning:
        """Create CrashWarning based on protection score."""
        if protection_score >= 0.80:
            level = CrashRiskLevel.CRITICAL
            leverage = self._config.min_leverage_multiplier
            action = "EMERGENCY_DELEVERAGE: Close momentum positions, hedge remaining"
        elif protection_score >= 0.50:
            level = CrashRiskLevel.HIGH
            leverage = 0.50
            action = "REDUCE_EXPOSURE_50PCT: Close weakest momentum positions"
        elif protection_score >= 0.30:
            level = CrashRiskLevel.MEDIUM
            leverage = 0.75
            action = "REDUCE_EXPOSURE_25PCT: Tighten stops, reduce new positions"
        else:
            level = CrashRiskLevel.LOW
            leverage = 1.0
            action = "MAINTAIN_NORMAL: Continue normal operations"

        return CrashWarning(
            level=level,
            probability=protection_score,
            indicators=indicators,
            recommended_action=action,
            leverage_multiplier=leverage,
        )

    def get_position_size_multiplier(self) -> float:
        """
        Get current position size multiplier based on last warning.

        Returns:
            Multiplier for position sizing (0.2 to 1.0)
        """
        if self._last_warning is None:
            return 1.0
        return self._last_warning.leverage_multiplier

    def should_close_momentum_positions(self) -> bool:
        """Check if we should close momentum positions (CRITICAL level)."""
        if self._last_warning is None:
            return False
        return self._last_warning.level == CrashRiskLevel.CRITICAL

    def should_halt_new_momentum_trades(self) -> bool:
        """Check if we should halt new momentum trades (HIGH+ level)."""
        if self._last_warning is None:
            return False
        return self._last_warning.level in (CrashRiskLevel.HIGH, CrashRiskLevel.CRITICAL)

    def get_warning_history(self, limit: int = 20) -> list[dict]:
        """Get recent warning history."""
        return [w.to_dict() for w in self._warning_history[-limit:]]

    def get_status(self) -> dict[str, Any]:
        """Get current protection status for monitoring."""
        return {
            "enabled": True,
            "last_warning": self._last_warning.to_dict() if self._last_warning else None,
            "current_level": self._last_warning.level.value if self._last_warning else "low",
            "leverage_multiplier": self.get_position_size_multiplier(),
            "halt_new_trades": self.should_halt_new_momentum_trades(),
            "close_positions": self.should_close_momentum_positions(),
            "config": {
                "vix_spike_threshold": self._config.vix_spike_threshold,
                "drawdown_warning_pct": self._config.drawdown_warning_pct,
                "correlation_spike_threshold": self._config.correlation_spike_threshold,
            }
        }


# =============================================================================
# VIX-Based Market Regime Detection
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification based on volatility."""
    LOW_VOL = "low_vol"       # VIX < 15
    NORMAL = "normal"         # VIX 15-20
    ELEVATED = "elevated"     # VIX 20-25
    HIGH_VOL = "high_vol"     # VIX 25-35
    CRISIS = "crisis"         # VIX > 35


def detect_market_regime(vix: float) -> MarketRegime:
    """
    Detect current market regime based on VIX level.

    VIX thresholds based on historical analysis:
    - < 15: Complacency (potential risk build-up)
    - 15-20: Normal market conditions
    - 20-25: Elevated uncertainty
    - 25-35: High volatility, increased hedging
    - > 35: Crisis conditions

    Args:
        vix: Current VIX level

    Returns:
        MarketRegime classification
    """
    if vix < 15:
        return MarketRegime.LOW_VOL
    elif vix < 20:
        return MarketRegime.NORMAL
    elif vix < 25:
        return MarketRegime.ELEVATED
    elif vix < 35:
        return MarketRegime.HIGH_VOL
    else:
        return MarketRegime.CRISIS


def get_regime_risk_parameters(regime: MarketRegime) -> dict[str, float]:
    """
    Get risk parameters adjusted for market regime.

    Returns:
        Dict with position_multiplier, var_confidence_boost, stop_multiplier
    """
    REGIME_PARAMS = {
        MarketRegime.LOW_VOL: {
            "position_multiplier": 1.0,
            "var_confidence_boost": 0.02,  # Use 97% VaR (risk building)
            "stop_multiplier": 0.8,  # Tighter stops
            "new_position_allowed": True,
        },
        MarketRegime.NORMAL: {
            "position_multiplier": 1.0,
            "var_confidence_boost": 0.0,  # Standard 95% VaR
            "stop_multiplier": 1.0,
            "new_position_allowed": True,
        },
        MarketRegime.ELEVATED: {
            "position_multiplier": 0.8,
            "var_confidence_boost": 0.01,  # 96% VaR
            "stop_multiplier": 1.2,  # Wider stops
            "new_position_allowed": True,
        },
        MarketRegime.HIGH_VOL: {
            "position_multiplier": 0.5,
            "var_confidence_boost": 0.02,  # 97% VaR
            "stop_multiplier": 1.5,  # Much wider stops
            "new_position_allowed": True,
        },
        MarketRegime.CRISIS: {
            "position_multiplier": 0.25,
            "var_confidence_boost": 0.04,  # 99% VaR
            "stop_multiplier": 2.0,  # Very wide stops
            "new_position_allowed": False,  # No new positions
        },
    }
    return REGIME_PARAMS.get(regime, REGIME_PARAMS[MarketRegime.NORMAL])


# =============================================================================
# PHASE 5: Enhanced Crash Protection with History Tracking
# =============================================================================

class DrawdownVelocity(Enum):
    """Drawdown velocity classification."""
    GRADUAL = "gradual"      # < 1% per day
    MODERATE = "moderate"    # 1-3% per day
    FAST = "fast"            # 3-5% per day
    CRASH = "crash"          # > 5% per day


@dataclass
class VelocityAwareWarning:
    """Enhanced warning with velocity awareness."""
    crash_warning: CrashWarning
    drawdown_velocity: DrawdownVelocity
    velocity_pct_per_day: float
    additional_position_reduction: float  # Extra reduction due to velocity
    immediate_action_required: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = self.crash_warning.to_dict()
        result.update({
            "drawdown_velocity": self.drawdown_velocity.value,
            "velocity_pct_per_day": self.velocity_pct_per_day,
            "additional_position_reduction": self.additional_position_reduction,
            "immediate_action_required": self.immediate_action_required,
        })
        return result


class EnhancedCrashProtection(MomentumCrashProtection):
    """
    Enhanced crash protection with history tracking and velocity awareness.

    PHASE 5 enhancement of MomentumCrashProtection:
    - Maintains historical data for indicators
    - Calculates drawdown velocity
    - Provides velocity-aware recommendations
    - Implements protection mode with decay
    """

    def __init__(
        self,
        config: CrashProtectionConfig | None = None,
        vix_history_size: int = 100,
        equity_history_size: int = 100,
        correlation_history_size: int = 100,
        protection_decay_days: int = 5,
    ):
        super().__init__(config)

        # History storage
        self._vix_history: list[tuple[datetime, float]] = []
        self._equity_history: list[tuple[datetime, float]] = []
        self._correlation_history: list[tuple[datetime, float]] = []
        self._momentum_returns_history: list[tuple[datetime, float]] = []

        # History size limits
        self._vix_history_size = vix_history_size
        self._equity_history_size = equity_history_size
        self._correlation_history_size = correlation_history_size

        # Protection mode state
        self._protection_decay_days = protection_decay_days
        self._is_in_protection_mode = False
        self._protection_start: datetime | None = None
        self._protection_peak_level = CrashRiskLevel.LOW

        logger.info(
            f"EnhancedCrashProtection initialized with history tracking: "
            f"vix={vix_history_size}, equity={equity_history_size}"
        )

    def record_vix(self, vix: float, timestamp: datetime | None = None) -> None:
        """Record VIX observation."""
        ts = timestamp or datetime.now(timezone.utc)
        self._vix_history.append((ts, vix))
        if len(self._vix_history) > self._vix_history_size:
            self._vix_history.pop(0)

    def record_equity(self, equity: float, timestamp: datetime | None = None) -> None:
        """Record portfolio equity."""
        ts = timestamp or datetime.now(timezone.utc)
        self._equity_history.append((ts, equity))
        if len(self._equity_history) > self._equity_history_size:
            self._equity_history.pop(0)

    def record_correlation(self, correlation: float, timestamp: datetime | None = None) -> None:
        """Record cross-asset correlation."""
        ts = timestamp or datetime.now(timezone.utc)
        self._correlation_history.append((ts, correlation))
        if len(self._correlation_history) > self._correlation_history_size:
            self._correlation_history.pop(0)

    def record_momentum_return(self, returns: float, timestamp: datetime | None = None) -> None:
        """Record momentum strategy return."""
        ts = timestamp or datetime.now(timezone.utc)
        self._momentum_returns_history.append((ts, returns))
        if len(self._momentum_returns_history) > 100:
            self._momentum_returns_history.pop(0)

    def _calculate_vix_ma(self, window: int | None = None) -> float | None:
        """Calculate VIX moving average from history."""
        window = window or self._config.vix_ma_period
        if len(self._vix_history) < window:
            return None

        recent_vix = [v for _, v in self._vix_history[-window:]]
        return float(np.mean(recent_vix))

    def _calculate_drawdown_and_velocity(self) -> tuple[float, float, DrawdownVelocity]:
        """
        Calculate current drawdown and velocity from equity history.

        Returns:
            (current_drawdown, velocity_per_day, velocity_level)
        """
        if len(self._equity_history) < 5:
            return 0.0, 0.0, DrawdownVelocity.GRADUAL

        equities = np.array([e for _, e in self._equity_history])
        timestamps = [t for t, _ in self._equity_history]

        # Calculate peak and current drawdown
        peak = np.maximum.accumulate(equities)
        drawdowns = (equities - peak) / peak
        current_dd = drawdowns[-1]

        # Calculate velocity (change in drawdown over last 5 observations)
        if len(drawdowns) >= 5:
            dd_5_ago = drawdowns[-5]
            dd_change = current_dd - dd_5_ago

            # Estimate days between observations
            time_delta = (timestamps[-1] - timestamps[-5]).total_seconds() / 86400
            if time_delta > 0:
                velocity_per_day = dd_change / time_delta * 100  # Convert to % per day
            else:
                velocity_per_day = 0.0
        else:
            velocity_per_day = 0.0

        # Classify velocity
        abs_velocity = abs(velocity_per_day)
        if abs_velocity < 1.0:
            velocity_level = DrawdownVelocity.GRADUAL
        elif abs_velocity < 3.0:
            velocity_level = DrawdownVelocity.MODERATE
        elif abs_velocity < 5.0:
            velocity_level = DrawdownVelocity.FAST
        else:
            velocity_level = DrawdownVelocity.CRASH

        return float(current_dd), velocity_per_day, velocity_level

    def _calculate_correlation_change(self) -> tuple[float, float]:
        """
        Calculate correlation level and recent change.

        Returns:
            (current_correlation, correlation_change)
        """
        if len(self._correlation_history) < 5:
            return 0.0, 0.0

        correlations = [c for _, c in self._correlation_history]
        current = correlations[-1]

        # Calculate change over last 5 observations
        if len(correlations) >= 5:
            change = current - correlations[-5]
        else:
            change = 0.0

        return current, change

    def _calculate_winner_loser_reversal(self) -> tuple[float, float]:
        """
        Calculate winner/loser returns from momentum history.

        Returns:
            (winners_return, losers_return) - simplified as prior vs recent performance
        """
        if len(self._momentum_returns_history) < 20:
            return 0.0, 0.0

        returns = [r for _, r in self._momentum_returns_history]

        # Prior period (20-5 days ago) represents "winners"
        prior_return = np.mean(returns[-20:-5])

        # Recent period (last 5 days) - if reversing, losers outperform
        recent_return = np.mean(returns[-5:])

        return prior_return, recent_return

    def evaluate_with_histories(self) -> VelocityAwareWarning:
        """
        Evaluate crash risk using recorded histories.

        Returns:
            VelocityAwareWarning with velocity-aware recommendations
        """
        # Get values from histories
        current_dd, velocity, velocity_level = self._calculate_drawdown_and_velocity()
        current_corr, corr_change = self._calculate_correlation_change()
        winners_ret, losers_ret = self._calculate_winner_loser_reversal()

        # VIX
        vix_current = self._vix_history[-1][1] if self._vix_history else None
        vix_ma = self._calculate_vix_ma()

        # Get base crash warning
        crash_warning = self.evaluate_crash_risk(
            recent_drawdown=current_dd,
            vix_current=vix_current,
            vix_ma=vix_ma,
            correlation_increase=corr_change if corr_change > 0 else None,
            avg_correlation=current_corr,
            past_winners_return=winners_ret,
            past_losers_return=losers_ret,
        )

        # Calculate velocity-based adjustments
        velocity_adjustments = {
            DrawdownVelocity.GRADUAL: 0.0,
            DrawdownVelocity.MODERATE: 0.20,  # 20% additional reduction
            DrawdownVelocity.FAST: 0.50,      # 50% additional reduction
            DrawdownVelocity.CRASH: 0.80,     # 80% additional reduction (close most positions)
        }

        additional_reduction = velocity_adjustments.get(velocity_level, 0.0)
        immediate_action = velocity_level in (DrawdownVelocity.FAST, DrawdownVelocity.CRASH)

        # Update protection mode
        if crash_warning.level in (CrashRiskLevel.HIGH, CrashRiskLevel.CRITICAL):
            if not self._is_in_protection_mode:
                self._is_in_protection_mode = True
                self._protection_start = datetime.now(timezone.utc)
                self._protection_peak_level = crash_warning.level
            elif crash_warning.level.value > self._protection_peak_level.value:
                self._protection_peak_level = crash_warning.level
        elif self._is_in_protection_mode:
            # Check if we should exit protection mode
            if self._protection_start:
                days_in_protection = (datetime.now(timezone.utc) - self._protection_start).days
                if days_in_protection >= self._protection_decay_days and \
                   crash_warning.level == CrashRiskLevel.LOW:
                    self._is_in_protection_mode = False
                    self._protection_start = None
                    self._protection_peak_level = CrashRiskLevel.LOW

        return VelocityAwareWarning(
            crash_warning=crash_warning,
            drawdown_velocity=velocity_level,
            velocity_pct_per_day=velocity,
            additional_position_reduction=additional_reduction,
            immediate_action_required=immediate_action,
        )

    def get_total_leverage_multiplier(self) -> float:
        """
        Get total leverage multiplier including velocity adjustment.

        Returns:
            Float between 0 and 1
        """
        base = self.get_position_size_multiplier()
        _, _, velocity_level = self._calculate_drawdown_and_velocity()

        velocity_multipliers = {
            DrawdownVelocity.GRADUAL: 1.0,
            DrawdownVelocity.MODERATE: 0.8,
            DrawdownVelocity.FAST: 0.5,
            DrawdownVelocity.CRASH: 0.2,
        }

        velocity_mult = velocity_multipliers.get(velocity_level, 1.0)

        # If in protection mode, maintain reduced leverage
        protection_mult = 0.7 if self._is_in_protection_mode else 1.0

        return base * velocity_mult * protection_mult

    def get_enhanced_status(self) -> dict[str, Any]:
        """Get enhanced status including velocity and protection mode."""
        base_status = self.get_status()

        current_dd, velocity, velocity_level = self._calculate_drawdown_and_velocity()
        current_corr, corr_change = self._calculate_correlation_change()

        base_status.update({
            "enhanced": True,
            "velocity": {
                "level": velocity_level.value,
                "pct_per_day": velocity,
            },
            "current_drawdown_pct": current_dd * 100,
            "correlation": {
                "current": current_corr,
                "change": corr_change,
            },
            "protection_mode": {
                "active": self._is_in_protection_mode,
                "start": self._protection_start.isoformat() if self._protection_start else None,
                "peak_level": self._protection_peak_level.value,
                "decay_days": self._protection_decay_days,
            },
            "history_sizes": {
                "vix": len(self._vix_history),
                "equity": len(self._equity_history),
                "correlation": len(self._correlation_history),
                "momentum": len(self._momentum_returns_history),
            },
            "total_leverage_multiplier": self.get_total_leverage_multiplier(),
        })

        return base_status


def create_crash_protection(
    config: dict[str, Any] | None = None,
    enhanced: bool = True,
) -> MomentumCrashProtection | EnhancedCrashProtection:
    """
    Factory function to create crash protection system.

    Args:
        config: Configuration dictionary with optional keys from CrashProtectionConfig
        enhanced: If True, create EnhancedCrashProtection with history tracking

    Returns:
        Configured crash protection instance
    """
    config = config or {}

    protection_config = CrashProtectionConfig(
        vix_spike_threshold=config.get("vix_spike_threshold", 1.5),
        vix_extreme_level=config.get("vix_extreme_level", 40.0),
        vix_ma_period=config.get("vix_ma_period", 20),
        drawdown_warning_pct=config.get("drawdown_warning_pct", 0.05),
        drawdown_critical_pct=config.get("drawdown_critical_pct", 0.10),
        correlation_spike_threshold=config.get("correlation_spike_threshold", 0.20),
        correlation_crisis_level=config.get("correlation_crisis_level", 0.70),
        reversal_threshold=config.get("reversal_threshold", 0.05),
        min_leverage_multiplier=config.get("min_leverage_multiplier", 0.20),
    )

    if enhanced:
        return EnhancedCrashProtection(
            config=protection_config,
            vix_history_size=config.get("vix_history_size", 100),
            equity_history_size=config.get("equity_history_size", 100),
            correlation_history_size=config.get("correlation_history_size", 100),
            protection_decay_days=config.get("protection_decay_days", 5),
        )
    else:
        return MomentumCrashProtection(config=protection_config)

"""
Slippage Estimation Module
==========================

Estimates expected slippage for signal generation (Issue #Q11).
Incorporates capacity constraints for strategy sizing (Issue #Q12).

Features:
- Market impact estimation
- Liquidity-based slippage
- Capacity constraints
- Size-adjusted signal strength
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LiquidityTier(str, Enum):
    """Asset liquidity classification."""
    ULTRA_LIQUID = "ultra_liquid"  # SPY, ES, major FX
    HIGHLY_LIQUID = "highly_liquid"  # Large caps, major futures
    LIQUID = "liquid"  # Mid caps, minor futures
    MODERATELY_LIQUID = "moderately_liquid"  # Small caps
    ILLIQUID = "illiquid"  # Micro caps, exotic


@dataclass
class SlippageEstimate:
    """Estimated slippage for a trade."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int

    # Slippage components (in basis points)
    spread_cost_bps: float  # Half spread crossing cost
    market_impact_bps: float  # Temporary impact
    permanent_impact_bps: float  # Information leakage
    volatility_cost_bps: float  # Execution risk from vol

    # Totals
    total_slippage_bps: float
    total_slippage_dollars: float

    # Confidence
    confidence: float  # 0-1, higher is more confident
    estimation_method: str

    # Timing recommendation
    recommended_urgency: str  # 'immediate', 'normal', 'patient'
    optimal_execution_horizon_minutes: float

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'spread_cost_bps': self.spread_cost_bps,
            'market_impact_bps': self.market_impact_bps,
            'permanent_impact_bps': self.permanent_impact_bps,
            'volatility_cost_bps': self.volatility_cost_bps,
            'total_slippage_bps': self.total_slippage_bps,
            'total_slippage_dollars': self.total_slippage_dollars,
            'confidence': self.confidence,
            'estimation_method': self.estimation_method,
            'recommended_urgency': self.recommended_urgency,
            'optimal_execution_horizon_minutes': self.optimal_execution_horizon_minutes,
        }


@dataclass
class CapacityConstraints:
    """Capacity limits for a strategy/symbol."""
    symbol: str

    # Position limits
    max_position_shares: int
    max_position_pct_adv: float  # % of average daily volume

    # Order limits
    max_single_order_shares: int
    max_single_order_pct_adv: float

    # Rate limits
    max_daily_volume_pct: float  # Max % of daily volume we can be
    max_hourly_volume_pct: float

    # Computed
    current_capacity_used_pct: float = 0.0
    remaining_capacity_shares: int = 0

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'max_position_shares': self.max_position_shares,
            'max_position_pct_adv': self.max_position_pct_adv,
            'max_single_order_shares': self.max_single_order_shares,
            'max_single_order_pct_adv': self.max_single_order_pct_adv,
            'max_daily_volume_pct': self.max_daily_volume_pct,
            'max_hourly_volume_pct': self.max_hourly_volume_pct,
            'current_capacity_used_pct': self.current_capacity_used_pct,
            'remaining_capacity_shares': self.remaining_capacity_shares,
        }


@dataclass
class LiquidityProfile:
    """Liquidity characteristics of an asset."""
    symbol: str
    average_daily_volume: int
    average_spread_bps: float
    average_depth_shares: int  # At best bid/ask
    volatility_daily_pct: float
    liquidity_tier: LiquidityTier

    # Time-of-day patterns
    volume_by_hour: dict[int, float] = field(default_factory=dict)  # Hour -> relative volume
    spread_by_hour: dict[int, float] = field(default_factory=dict)  # Hour -> relative spread

    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SlippageEstimator:
    """
    Estimates execution slippage for trading signals (#Q11).

    Uses multiple models:
    - Square-root market impact (Almgren-Chriss)
    - Linear spread model
    - Volatility adjustment
    """

    # Default liquidity profiles by tier
    DEFAULT_PROFILES = {
        LiquidityTier.ULTRA_LIQUID: {
            'spread_bps': 1.0,
            'depth_shares': 50000,
            'impact_coefficient': 0.05,
        },
        LiquidityTier.HIGHLY_LIQUID: {
            'spread_bps': 3.0,
            'depth_shares': 10000,
            'impact_coefficient': 0.10,
        },
        LiquidityTier.LIQUID: {
            'spread_bps': 8.0,
            'depth_shares': 3000,
            'impact_coefficient': 0.20,
        },
        LiquidityTier.MODERATELY_LIQUID: {
            'spread_bps': 20.0,
            'depth_shares': 500,
            'impact_coefficient': 0.40,
        },
        LiquidityTier.ILLIQUID: {
            'spread_bps': 50.0,
            'depth_shares': 100,
            'impact_coefficient': 0.80,
        },
    }

    def __init__(
        self,
        permanent_impact_fraction: float = 0.3,  # Fraction of impact that's permanent
        volatility_risk_premium: float = 0.5,  # Vol adjustment factor
    ):
        self.permanent_impact_fraction = permanent_impact_fraction
        self.volatility_risk_premium = volatility_risk_premium

        # Liquidity profiles by symbol
        self._profiles: dict[str, LiquidityProfile] = {}

        # Market prices for dollar calculations
        self._prices: dict[str, float] = {}

    def update_liquidity_profile(
        self,
        symbol: str,
        adv: int,
        spread_bps: float,
        depth_shares: int,
        volatility_pct: float,
        tier: LiquidityTier | None = None,
    ) -> LiquidityProfile:
        """Update liquidity profile for a symbol."""
        # Auto-classify tier if not provided
        if tier is None:
            tier = self._classify_liquidity_tier(adv, spread_bps)

        profile = LiquidityProfile(
            symbol=symbol,
            average_daily_volume=adv,
            average_spread_bps=spread_bps,
            average_depth_shares=depth_shares,
            volatility_daily_pct=volatility_pct,
            liquidity_tier=tier,
        )

        self._profiles[symbol] = profile
        return profile

    def update_price(self, symbol: str, price: float) -> None:
        """Update current price for a symbol."""
        self._prices[symbol] = price

    def _classify_liquidity_tier(
        self,
        adv: int,
        spread_bps: float,
    ) -> LiquidityTier:
        """Classify asset into liquidity tier."""
        if adv > 10_000_000 and spread_bps < 2:
            return LiquidityTier.ULTRA_LIQUID
        elif adv > 1_000_000 and spread_bps < 5:
            return LiquidityTier.HIGHLY_LIQUID
        elif adv > 100_000 and spread_bps < 15:
            return LiquidityTier.LIQUID
        elif adv > 10_000 and spread_bps < 30:
            return LiquidityTier.MODERATELY_LIQUID
        else:
            return LiquidityTier.ILLIQUID

    def estimate_slippage(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float | None = None,
        urgency: str = "normal",
        vix_level: float | None = None,
    ) -> SlippageEstimate:
        """
        Estimate slippage for a potential trade.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Number of shares/contracts
            price: Current price (uses cached if not provided)
            urgency: 'immediate', 'normal', or 'patient'
            vix_level: Optional VIX level for regime-based multiplier

        Returns:
            SlippageEstimate with breakdown
        """
        profile = self._profiles.get(symbol)
        current_price = price or self._prices.get(symbol, 100.0)

        if profile is None:
            # Use default tier assumptions
            return self._estimate_without_profile(
                symbol, side, quantity, current_price, urgency
            )

        # Spread cost (half spread to cross)
        spread_cost_bps = profile.average_spread_bps / 2

        # Urgency adjustment
        if urgency == "immediate":
            spread_cost_bps *= 1.5  # Full spread crossing
        elif urgency == "patient":
            spread_cost_bps *= 0.3  # Passive fills

        # Market impact using square-root model
        # Impact = σ * sqrt(Q/ADV) * coefficient
        adv = profile.average_daily_volume or 1_000_000
        participation_rate = quantity / adv

        tier_params = self.DEFAULT_PROFILES.get(
            profile.liquidity_tier,
            self.DEFAULT_PROFILES[LiquidityTier.LIQUID]
        )
        impact_coef = tier_params['impact_coefficient']

        # Square-root impact using Almgren-Chriss model
        # Impact = sigma * sqrt(Q/ADV) * coefficient * sqrt(price_normalizer)
        # Price normalization: use sqrt(price/100) to account for tick size effects
        vol_daily = profile.volatility_daily_pct / 100
        price_normalizer = math.sqrt(current_price / 100.0) if current_price > 0 else 1.0
        market_impact_bps = (
            vol_daily * math.sqrt(participation_rate) * impact_coef * price_normalizer * 10000
        )

        # MS004: Regime-based multiplier for crisis conditions
        # VIX > 40: crisis mode (10x), VIX > 25: stressed (5x), else normal (1x)
        if vix_level is not None:
            if vix_level > 40:
                regime_multiplier = 10.0
            elif vix_level > 25:
                regime_multiplier = 5.0
            else:
                regime_multiplier = 1.0
            market_impact_bps *= regime_multiplier

        # Urgency adjustment for impact
        if urgency == "immediate":
            market_impact_bps *= 1.8
        elif urgency == "patient":
            market_impact_bps *= 0.5

        # Permanent impact
        permanent_impact_bps = market_impact_bps * self.permanent_impact_fraction

        # Volatility risk (execution risk from price movement)
        # MS016 FIX: Correct time scaling - use vol_daily * sqrt(execution_hours / 6.5)
        # This avoids double-counting time by scaling daily vol directly
        optimal_horizon = self._calculate_optimal_horizon(
            quantity, adv, urgency
        )
        execution_hours = optimal_horizon / 60
        # Correct formula: volatility scales with sqrt(time fraction of trading day)
        volatility_cost_bps = (
            vol_daily * math.sqrt(execution_hours / 6.5) *
            self.volatility_risk_premium * 10000
        )

        # Total
        total_bps = (
            spread_cost_bps +
            market_impact_bps +
            permanent_impact_bps +
            volatility_cost_bps
        )

        # Dollar cost
        notional = quantity * current_price
        total_dollars = notional * total_bps / 10000

        # Confidence based on profile freshness and data quality
        age_hours = (datetime.now(timezone.utc) - profile.last_updated).total_seconds() / 3600
        confidence = max(0.3, min(1.0, 1.0 - age_hours / 24))

        return SlippageEstimate(
            symbol=symbol,
            side=side,
            quantity=quantity,
            spread_cost_bps=round(spread_cost_bps, 2),
            market_impact_bps=round(market_impact_bps, 2),
            permanent_impact_bps=round(permanent_impact_bps, 2),
            volatility_cost_bps=round(volatility_cost_bps, 2),
            total_slippage_bps=round(total_bps, 2),
            total_slippage_dollars=round(total_dollars, 2),
            confidence=round(confidence, 2),
            estimation_method="almgren_chriss",
            recommended_urgency=self._recommend_urgency(participation_rate),
            optimal_execution_horizon_minutes=round(optimal_horizon, 1),
        )

    def _estimate_without_profile(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        urgency: str,
    ) -> SlippageEstimate:
        """Estimate with default assumptions when no profile available."""
        # Assume moderately liquid
        tier_params = self.DEFAULT_PROFILES[LiquidityTier.LIQUID]

        spread_cost_bps = tier_params['spread_bps'] / 2

        # Conservative impact estimate
        assumed_adv = 500_000
        participation_rate = quantity / assumed_adv
        market_impact_bps = (
            0.02 * math.sqrt(participation_rate) *
            tier_params['impact_coefficient'] * 10000
        )

        permanent_impact_bps = market_impact_bps * self.permanent_impact_fraction
        volatility_cost_bps = 5.0  # Default assumption

        total_bps = spread_cost_bps + market_impact_bps + permanent_impact_bps + volatility_cost_bps
        total_dollars = quantity * price * total_bps / 10000

        return SlippageEstimate(
            symbol=symbol,
            side=side,
            quantity=quantity,
            spread_cost_bps=round(spread_cost_bps, 2),
            market_impact_bps=round(market_impact_bps, 2),
            permanent_impact_bps=round(permanent_impact_bps, 2),
            volatility_cost_bps=round(volatility_cost_bps, 2),
            total_slippage_bps=round(total_bps, 2),
            total_slippage_dollars=round(total_dollars, 2),
            confidence=0.4,  # Low confidence without profile
            estimation_method="default_assumptions",
            recommended_urgency=urgency,
            optimal_execution_horizon_minutes=30.0,
        )

    def _calculate_optimal_horizon(
        self,
        quantity: int,
        adv: int,
        urgency: str,
    ) -> float:
        """Calculate optimal execution horizon in minutes."""
        participation_rate = quantity / adv

        # Base horizon proportional to size
        base_minutes = participation_rate * 390  # 6.5 hours trading day in minutes

        # Urgency adjustment
        if urgency == "immediate":
            return max(1, base_minutes * 0.3)
        elif urgency == "patient":
            return min(390, base_minutes * 2.0)
        else:
            return max(5, min(120, base_minutes))

    def _recommend_urgency(self, participation_rate: float) -> str:
        """Recommend execution urgency based on participation rate."""
        if participation_rate < 0.01:
            return "immediate"
        elif participation_rate < 0.05:
            return "normal"
        else:
            return "patient"


class CapacityManager:
    """
    Manages capacity constraints for strategies (#Q12).

    Ensures strategies don't exceed sustainable position sizes.
    """

    def __init__(
        self,
        default_max_adv_pct: float = 5.0,  # Max 5% of ADV per position
        default_max_order_adv_pct: float = 1.0,  # Max 1% of ADV per order
        default_max_daily_volume_pct: float = 10.0,
    ):
        self.default_max_adv_pct = default_max_adv_pct
        self.default_max_order_adv_pct = default_max_order_adv_pct
        self.default_max_daily_volume_pct = default_max_daily_volume_pct

        # Custom constraints by symbol
        self._constraints: dict[str, CapacityConstraints] = {}

        # Volume data
        self._adv: dict[str, int] = {}

        # Current usage tracking
        self._current_positions: dict[str, int] = {}
        self._daily_volume: dict[str, int] = {}
        self._hourly_volume: dict[str, dict[int, int]] = {}

    def set_adv(self, symbol: str, adv: int) -> None:
        """Set average daily volume for a symbol."""
        self._adv[symbol] = adv
        self._update_constraints(symbol)

    def set_custom_constraints(
        self,
        symbol: str,
        max_position_pct_adv: float | None = None,
        max_order_pct_adv: float | None = None,
        max_daily_volume_pct: float | None = None,
    ) -> None:
        """Set custom capacity constraints for a symbol."""
        adv = self._adv.get(symbol, 1_000_000)

        constraints = CapacityConstraints(
            symbol=symbol,
            max_position_shares=int(adv * (max_position_pct_adv or self.default_max_adv_pct) / 100),
            max_position_pct_adv=max_position_pct_adv or self.default_max_adv_pct,
            max_single_order_shares=int(adv * (max_order_pct_adv or self.default_max_order_adv_pct) / 100),
            max_single_order_pct_adv=max_order_pct_adv or self.default_max_order_adv_pct,
            max_daily_volume_pct=max_daily_volume_pct or self.default_max_daily_volume_pct,
            max_hourly_volume_pct=(max_daily_volume_pct or self.default_max_daily_volume_pct) / 6.5,
        )

        self._constraints[symbol] = constraints

    def _update_constraints(self, symbol: str) -> None:
        """Update constraints when ADV changes."""
        adv = self._adv.get(symbol, 1_000_000)

        if symbol not in self._constraints:
            self._constraints[symbol] = CapacityConstraints(
                symbol=symbol,
                max_position_shares=int(adv * self.default_max_adv_pct / 100),
                max_position_pct_adv=self.default_max_adv_pct,
                max_single_order_shares=int(adv * self.default_max_order_adv_pct / 100),
                max_single_order_pct_adv=self.default_max_order_adv_pct,
                max_daily_volume_pct=self.default_max_daily_volume_pct,
                max_hourly_volume_pct=self.default_max_daily_volume_pct / 6.5,
            )
        else:
            # Update share limits based on new ADV
            c = self._constraints[symbol]
            c.max_position_shares = int(adv * c.max_position_pct_adv / 100)
            c.max_single_order_shares = int(adv * c.max_single_order_pct_adv / 100)

    def update_position(self, symbol: str, quantity: int) -> None:
        """Update current position for a symbol."""
        self._current_positions[symbol] = quantity
        self._update_capacity_used(symbol)

    def record_execution(self, symbol: str, quantity: int) -> None:
        """Record executed volume."""
        self._daily_volume[symbol] = self._daily_volume.get(symbol, 0) + abs(quantity)

        hour = datetime.now(timezone.utc).hour
        if symbol not in self._hourly_volume:
            self._hourly_volume[symbol] = {}
        self._hourly_volume[symbol][hour] = self._hourly_volume[symbol].get(hour, 0) + abs(quantity)

    def _update_capacity_used(self, symbol: str) -> None:
        """Update capacity usage metrics."""
        if symbol not in self._constraints:
            return

        c = self._constraints[symbol]
        current_pos = abs(self._current_positions.get(symbol, 0))

        c.current_capacity_used_pct = (current_pos / c.max_position_shares * 100) if c.max_position_shares > 0 else 0
        c.remaining_capacity_shares = max(0, c.max_position_shares - current_pos)

    def get_constraints(self, symbol: str) -> CapacityConstraints:
        """Get capacity constraints for a symbol."""
        if symbol not in self._constraints:
            self._update_constraints(symbol)
        return self._constraints[symbol]

    def check_capacity(
        self,
        symbol: str,
        proposed_quantity: int,
    ) -> dict:
        """
        Check if proposed trade fits within capacity constraints.

        Returns:
            Dict with 'allowed', 'max_allowed', and 'reasons'
        """
        constraints = self.get_constraints(symbol)
        adv = self._adv.get(symbol, 1_000_000)

        issues = []
        max_allowed = proposed_quantity

        # Check single order limit
        if abs(proposed_quantity) > constraints.max_single_order_shares:
            issues.append(f"Exceeds max order size ({constraints.max_single_order_shares} shares)")
            max_allowed = min(max_allowed, constraints.max_single_order_shares)

        # Check position limit
        current_pos = self._current_positions.get(symbol, 0)
        new_pos = current_pos + proposed_quantity
        if abs(new_pos) > constraints.max_position_shares:
            issues.append(f"Would exceed max position ({constraints.max_position_shares} shares)")
            remaining = constraints.max_position_shares - abs(current_pos)
            max_allowed = min(max_allowed, remaining)

        # Check daily volume limit
        daily_vol = self._daily_volume.get(symbol, 0)
        max_daily = int(adv * constraints.max_daily_volume_pct / 100)
        if daily_vol + abs(proposed_quantity) > max_daily:
            issues.append(f"Would exceed daily volume limit ({constraints.max_daily_volume_pct}% of ADV)")
            remaining = max_daily - daily_vol
            max_allowed = min(max_allowed, remaining)

        # Check hourly volume limit
        hour = datetime.now(timezone.utc).hour
        hourly_vol = self._hourly_volume.get(symbol, {}).get(hour, 0)
        max_hourly = int(adv * constraints.max_hourly_volume_pct / 100)
        if hourly_vol + abs(proposed_quantity) > max_hourly:
            issues.append(f"Would exceed hourly volume limit ({constraints.max_hourly_volume_pct:.1f}% of ADV)")
            remaining = max_hourly - hourly_vol
            max_allowed = min(max_allowed, remaining)

        return {
            'allowed': len(issues) == 0,
            'proposed_quantity': proposed_quantity,
            'max_allowed': max(0, max_allowed),
            'reasons': issues,
            'capacity_used_pct': constraints.current_capacity_used_pct,
        }

    def get_adjusted_signal_size(
        self,
        symbol: str,
        desired_quantity: int,
        signal_strength: float,
    ) -> dict:
        """
        Get capacity-adjusted position size.

        Scales down size if hitting capacity limits.
        """
        check = self.check_capacity(symbol, desired_quantity)

        if check['allowed']:
            return {
                'adjusted_quantity': desired_quantity,
                'scale_factor': 1.0,
                'capacity_limited': False,
                'signal_strength': signal_strength,
                'adjusted_signal_strength': signal_strength,
            }

        max_allowed = check['max_allowed']
        scale_factor = max_allowed / abs(desired_quantity) if desired_quantity != 0 else 0

        # Reduce signal strength if significantly capacity constrained
        adjusted_signal = signal_strength * min(1.0, scale_factor + 0.3)

        return {
            'adjusted_quantity': max_allowed if desired_quantity > 0 else -max_allowed,
            'scale_factor': scale_factor,
            'capacity_limited': True,
            'signal_strength': signal_strength,
            'adjusted_signal_strength': adjusted_signal,
            'limit_reasons': check['reasons'],
        }

    def reset_daily_volume(self) -> None:
        """Reset daily volume counters (call at EOD)."""
        self._daily_volume.clear()
        self._hourly_volume.clear()
        logger.info("Reset daily volume counters")

    def get_all_constraints(self) -> dict[str, dict]:
        """Get constraints for all tracked symbols."""
        return {
            symbol: c.to_dict()
            for symbol, c in self._constraints.items()
        }


class SignalSlippageAdjuster:
    """
    Adjusts signal strength based on expected slippage (#Q11).

    Reduces signal strength when slippage would consume expected alpha.
    """

    def __init__(
        self,
        slippage_estimator: SlippageEstimator,
        capacity_manager: CapacityManager,
        min_edge_after_slippage_bps: float = 5.0,  # Minimum alpha after costs
    ):
        self.slippage_estimator = slippage_estimator
        self.capacity_manager = capacity_manager
        self.min_edge_after_slippage_bps = min_edge_after_slippage_bps

    def adjust_signal(
        self,
        symbol: str,
        direction: str,
        raw_strength: float,
        expected_alpha_bps: float,
        desired_quantity: int,
        price: float,
        urgency: str = "normal",
    ) -> dict:
        """
        Adjust signal strength based on execution costs and capacity.

        Args:
            symbol: Trading symbol
            direction: 'LONG' or 'SHORT'
            raw_strength: Original signal strength (0-1)
            expected_alpha_bps: Expected alpha from the signal in bps
            desired_quantity: Desired position size
            price: Current price
            urgency: Execution urgency

        Returns:
            Adjusted signal with all components
        """
        # Estimate slippage
        side = "BUY" if direction == "LONG" else "SELL"
        slippage = self.slippage_estimator.estimate_slippage(
            symbol, side, abs(desired_quantity), price, urgency
        )

        # Check capacity
        capacity_adj = self.capacity_manager.get_adjusted_signal_size(
            symbol, desired_quantity, raw_strength
        )

        # Calculate net expected alpha
        net_alpha_bps = expected_alpha_bps - slippage.total_slippage_bps

        # Adjust signal strength based on cost impact
        if expected_alpha_bps > 0:
            alpha_retention = max(0, net_alpha_bps / expected_alpha_bps)
        else:
            alpha_retention = 0

        cost_adjusted_strength = raw_strength * alpha_retention

        # Combined adjustment (cost and capacity)
        final_strength = min(
            cost_adjusted_strength,
            capacity_adj['adjusted_signal_strength']
        )

        # Decision
        should_trade = (
            net_alpha_bps >= self.min_edge_after_slippage_bps and
            final_strength >= 0.1 and
            capacity_adj['adjusted_quantity'] > 0
        )

        return {
            'symbol': symbol,
            'direction': direction,
            'raw_strength': raw_strength,
            'cost_adjusted_strength': round(cost_adjusted_strength, 3),
            'capacity_adjusted_strength': round(capacity_adj['adjusted_signal_strength'], 3),
            'final_strength': round(final_strength, 3),
            'expected_alpha_bps': expected_alpha_bps,
            'slippage_bps': slippage.total_slippage_bps,
            'net_alpha_bps': round(net_alpha_bps, 2),
            'desired_quantity': desired_quantity,
            'adjusted_quantity': capacity_adj['adjusted_quantity'],
            'capacity_limited': capacity_adj['capacity_limited'],
            'should_trade': should_trade,
            'slippage_estimate': slippage.to_dict(),
            'execution_recommendation': {
                'urgency': slippage.recommended_urgency,
                'horizon_minutes': slippage.optimal_execution_horizon_minutes,
                'confidence': slippage.confidence,
            },
        }

    def batch_adjust_signals(
        self,
        signals: list[dict],
    ) -> list[dict]:
        """
        Adjust multiple signals considering portfolio-level capacity.

        Args:
            signals: List of dicts with symbol, direction, strength, alpha, quantity, price

        Returns:
            List of adjusted signals sorted by net alpha
        """
        adjusted = []

        for sig in signals:
            adj = self.adjust_signal(
                symbol=sig['symbol'],
                direction=sig['direction'],
                raw_strength=sig['strength'],
                expected_alpha_bps=sig['expected_alpha_bps'],
                desired_quantity=sig['quantity'],
                price=sig['price'],
                urgency=sig.get('urgency', 'normal'),
            )
            adjusted.append(adj)

        # Sort by net alpha (highest first)
        adjusted.sort(key=lambda x: x['net_alpha_bps'], reverse=True)

        return adjusted

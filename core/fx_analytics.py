"""
FX Analytics Module
===================

Advanced FX analytics including volatility smile, central bank detection, and carry trade.

Issues Addressed:
- #X10: Missing FX volatility smile data
- #X11: No central bank intervention detection
- #X12: FX fixing rates not tracked
- #X13: No carry trade optimization
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, time, timedelta, date
from enum import Enum
from typing import Any, Callable
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# =============================================================================
# FX VOLATILITY SMILE DATA (#X10)
# =============================================================================

@dataclass
class FXVolPoint:
    """Single point on FX volatility surface (#X10)."""
    delta: float  # 10, 25, ATM, -25, -10
    expiry_days: int
    implied_vol: float
    vol_type: str  # "call", "put", "straddle"


@dataclass
class FXVolSmile:
    """FX volatility smile at single expiry (#X10)."""
    pair: str
    expiry_days: int
    timestamp: datetime
    atm_vol: float
    rr_25d: float  # 25-delta risk reversal (call - put)
    rr_10d: float  # 10-delta risk reversal
    bf_25d: float  # 25-delta butterfly
    bf_10d: float  # 10-delta butterfly

    @property
    def skew(self) -> float:
        """Get smile skew (positive = call premium)."""
        return self.rr_25d

    @property
    def smile_curvature(self) -> float:
        """Get smile curvature (convexity)."""
        return self.bf_25d

    def get_vol_at_delta(self, delta: float) -> float:
        """
        Interpolate volatility at specific delta (#X10).

        Uses polynomial fitting through ATM, RR, and BF quotes.
        """
        # Standard delta points
        if abs(delta - 0.5) < 0.01:  # ATM
            return self.atm_vol

        # Approximate using RR and BF
        # Vol(delta) = ATM + 0.5*RR*(2*delta-1) + 2*BF*(delta-0.5)^2
        d = delta - 0.5
        vol = self.atm_vol + 0.5 * self.rr_25d * (2 * d) + 2 * self.bf_25d * (d ** 2) * 4

        return max(0.01, vol)

    def to_dict(self) -> dict:
        return {
            'pair': self.pair,
            'expiry_days': self.expiry_days,
            'timestamp': self.timestamp.isoformat(),
            'atm_vol': self.atm_vol,
            'rr_25d': self.rr_25d,
            'rr_10d': self.rr_10d,
            'bf_25d': self.bf_25d,
            'bf_10d': self.bf_10d,
            'skew': self.skew,
            'curvature': self.smile_curvature,
        }


@dataclass
class FXVolSurface:
    """Complete FX volatility surface (#X10)."""
    pair: str
    timestamp: datetime
    smiles: list[FXVolSmile]
    term_structure_atm: list[tuple[int, float]]  # (days, vol)

    def get_vol(self, delta: float, expiry_days: int) -> float:
        """
        Get interpolated volatility from surface (#X10).

        Args:
            delta: Delta (0-1 for calls)
            expiry_days: Days to expiry

        Returns:
            Interpolated implied volatility
        """
        # Find bracketing expiries
        sorted_smiles = sorted(self.smiles, key=lambda s: s.expiry_days)

        if not sorted_smiles:
            return 0.15  # Default

        # Find closest expiries
        lower = None
        upper = None
        for smile in sorted_smiles:
            if smile.expiry_days <= expiry_days:
                lower = smile
            if smile.expiry_days >= expiry_days and upper is None:
                upper = smile

        if lower is None:
            lower = sorted_smiles[0]
        if upper is None:
            upper = sorted_smiles[-1]

        # Get vols at delta for both expiries
        lower_vol = lower.get_vol_at_delta(delta)
        upper_vol = upper.get_vol_at_delta(delta)

        # Linear interpolation in time
        if lower.expiry_days == upper.expiry_days:
            return lower_vol

        weight = (expiry_days - lower.expiry_days) / (upper.expiry_days - lower.expiry_days)
        return lower_vol + weight * (upper_vol - lower_vol)

    def to_dict(self) -> dict:
        return {
            'pair': self.pair,
            'timestamp': self.timestamp.isoformat(),
            'smiles': [s.to_dict() for s in self.smiles],
            'term_structure_atm': self.term_structure_atm,
        }


class FXVolSmileManager:
    """
    FX volatility smile management (#X10).

    Tracks and manages vol surfaces for FX pairs with:
    - Standard tenor smiles (1W, 1M, 3M, 6M, 1Y)
    - Delta interpolation
    - Term structure
    """

    # Standard FX tenors
    STANDARD_TENORS = {
        '1W': 7,
        '2W': 14,
        '1M': 30,
        '2M': 60,
        '3M': 90,
        '6M': 180,
        '9M': 270,
        '1Y': 365,
    }

    def __init__(self):
        self._surfaces: dict[str, FXVolSurface] = {}
        self._smile_history: dict[str, list[FXVolSmile]] = defaultdict(list)

    def update_smile(self, smile: FXVolSmile) -> None:
        """Update smile for a pair/tenor."""
        key = f"{smile.pair}_{smile.expiry_days}"
        self._smile_history[key].append(smile)
        self._smile_history[key] = self._smile_history[key][-100:]

        # Rebuild surface
        self._rebuild_surface(smile.pair)

    def _rebuild_surface(self, pair: str) -> None:
        """Rebuild complete surface from smiles."""
        # Collect latest smile for each tenor
        smiles = []
        for tenor_name, days in self.STANDARD_TENORS.items():
            key = f"{pair}_{days}"
            history = self._smile_history.get(key, [])
            if history:
                smiles.append(history[-1])

        if not smiles:
            return

        # Build term structure
        term_structure = [(s.expiry_days, s.atm_vol) for s in sorted(smiles, key=lambda x: x.expiry_days)]

        self._surfaces[pair] = FXVolSurface(
            pair=pair,
            timestamp=datetime.now(timezone.utc),
            smiles=smiles,
            term_structure_atm=term_structure,
        )

    def get_surface(self, pair: str) -> FXVolSurface | None:
        """Get vol surface for pair."""
        return self._surfaces.get(pair)

    def get_vol(self, pair: str, delta: float, expiry_days: int) -> float | None:
        """Get interpolated vol from surface."""
        surface = self._surfaces.get(pair)
        if surface:
            return surface.get_vol(delta, expiry_days)
        return None

    def analyze_smile(self, pair: str, expiry_days: int) -> dict:
        """
        Analyze smile characteristics (#X10).

        Returns analysis of smile shape and signals.
        """
        surface = self._surfaces.get(pair)
        if not surface:
            return {'error': 'no_surface'}

        # Find closest smile
        smile = min(surface.smiles, key=lambda s: abs(s.expiry_days - expiry_days), default=None)
        if not smile:
            return {'error': 'no_smile'}

        # Analyze
        signals = []

        # Skew analysis
        if abs(smile.rr_25d) > 2.0:  # Significant skew
            direction = "calls_premium" if smile.rr_25d > 0 else "puts_premium"
            signals.append({
                'type': 'skew_extreme',
                'direction': direction,
                'value': smile.rr_25d,
            })

        # Curvature analysis
        if smile.bf_25d > 1.5:  # High curvature
            signals.append({
                'type': 'high_curvature',
                'value': smile.bf_25d,
                'interpretation': 'elevated_tail_risk',
            })

        # Historical comparison
        key = f"{pair}_{smile.expiry_days}"
        history = self._smile_history.get(key, [])
        if len(history) >= 20:
            hist_rrs = [s.rr_25d for s in history[-20:]]
            rr_percentile = sum(1 for r in hist_rrs if r < smile.rr_25d) / len(hist_rrs) * 100

            if rr_percentile > 90 or rr_percentile < 10:
                signals.append({
                    'type': 'rr_extreme',
                    'percentile': rr_percentile,
                })

        return {
            'pair': pair,
            'expiry_days': expiry_days,
            'smile': smile.to_dict(),
            'signals': signals,
        }


# =============================================================================
# CENTRAL BANK INTERVENTION DETECTION (#X11)
# =============================================================================

class InterventionType(str, Enum):
    """Type of central bank intervention."""
    VERBAL = "verbal"
    ACTUAL = "actual"
    COORDINATED = "coordinated"
    STERILIZED = "sterilized"
    UNSTERILIZED = "unsterilized"


@dataclass
class InterventionSignal:
    """Signal of potential central bank intervention (#X11)."""
    pair: str
    timestamp: datetime
    signal_type: str
    confidence: float
    indicators: dict[str, float]
    recommendation: str

    def to_dict(self) -> dict:
        return {
            'pair': self.pair,
            'timestamp': self.timestamp.isoformat(),
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'indicators': self.indicators,
            'recommendation': self.recommendation,
        }


@dataclass
class InterventionEvent:
    """Historical intervention event (#X11)."""
    pair: str
    date: date
    central_bank: str
    intervention_type: InterventionType
    direction: str  # "buy_domestic", "sell_domestic"
    estimated_size_usd: float | None
    market_impact_pct: float
    notes: str


class CentralBankInterventionDetector:
    """
    Central bank intervention detection (#X11).

    Detects potential intervention through:
    - Abnormal price movements
    - Volume spikes
    - Volatility patterns
    - Historical intervention levels
    """

    # Known intervention thresholds by pair
    INTERVENTION_LEVELS = {
        'USDJPY': {
            'verbal_warning': {'low': 145, 'high': 160},
            'likely_intervention': {'low': 150, 'high': 162},
            'central_bank': 'BOJ',
        },
        'USDCNY': {
            'verbal_warning': {'low': 7.0, 'high': 7.35},
            'likely_intervention': {'low': 7.1, 'high': 7.4},
            'central_bank': 'PBOC',
        },
        'EURCHF': {
            'verbal_warning': {'low': 0.92, 'high': None},
            'likely_intervention': {'low': 0.90, 'high': None},
            'central_bank': 'SNB',
        },
        'USDKRW': {
            'verbal_warning': {'low': 1300, 'high': 1400},
            'likely_intervention': {'low': 1350, 'high': 1450},
            'central_bank': 'BOK',
        },
    }

    def __init__(self):
        self._price_history: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        self._intervention_history: list[InterventionEvent] = []

    def update_price(self, pair: str, price: float) -> None:
        """Update price for intervention monitoring."""
        self._price_history[pair].append((datetime.now(timezone.utc), price))
        self._price_history[pair] = self._price_history[pair][-1000:]

    def check_intervention_risk(
        self,
        pair: str,
        current_price: float,
    ) -> InterventionSignal | None:
        """
        Check for intervention risk (#X11).

        Args:
            pair: Currency pair
            current_price: Current spot price

        Returns:
            InterventionSignal if risk detected
        """
        levels = self.INTERVENTION_LEVELS.get(pair)
        if not levels:
            return None

        indicators = {}
        signal_type = None
        confidence = 0.0
        recommendation = "monitor"

        verbal = levels.get('verbal_warning', {})
        intervention = levels.get('likely_intervention', {})

        # Check level-based signals
        if verbal.get('high') and current_price >= verbal['high']:
            signal_type = "verbal_warning_zone"
            confidence = 0.5
            indicators['price_vs_verbal_high'] = current_price - verbal['high']
            recommendation = "reduce_long_exposure"

            if intervention.get('high') and current_price >= intervention['high']:
                signal_type = "intervention_likely"
                confidence = 0.8
                indicators['price_vs_intervention'] = current_price - intervention['high']
                recommendation = "hedge_or_close_longs"

        elif verbal.get('low') and current_price <= verbal['low']:
            signal_type = "verbal_warning_zone"
            confidence = 0.5
            indicators['price_vs_verbal_low'] = verbal['low'] - current_price
            recommendation = "reduce_short_exposure"

            if intervention.get('low') and current_price <= intervention['low']:
                signal_type = "intervention_likely"
                confidence = 0.8
                indicators['price_vs_intervention'] = intervention['low'] - current_price
                recommendation = "hedge_or_close_shorts"

        # Check for abnormal moves
        history = self._price_history.get(pair, [])
        if len(history) >= 100:
            prices = [p for _, p in history[-100:]]
            returns = np.diff(np.log(prices))

            # Recent move
            if len(history) >= 20:
                recent_return = np.log(current_price / history[-20][1])
                std_return = np.std(returns)

                if abs(recent_return) > 3 * std_return:
                    indicators['move_z_score'] = recent_return / std_return
                    confidence = max(confidence, 0.6)
                    if signal_type is None:
                        signal_type = "abnormal_move"
                        recommendation = "await_confirmation"

        if signal_type is None:
            return None

        return InterventionSignal(
            pair=pair,
            timestamp=datetime.now(timezone.utc),
            signal_type=signal_type,
            confidence=confidence,
            indicators=indicators,
            recommendation=recommendation,
        )

    def record_intervention(self, event: InterventionEvent) -> None:
        """Record historical intervention event."""
        self._intervention_history.append(event)

    def get_intervention_history(self, pair: str) -> list[InterventionEvent]:
        """Get intervention history for pair."""
        return [e for e in self._intervention_history if e.pair == pair]


# =============================================================================
# FX FIXING RATES (#X12)
# =============================================================================

class FixingType(str, Enum):
    """Type of FX fixing rate."""
    WMR_LONDON_4PM = "wmr_london_4pm"  # WM/Reuters London 4pm
    ECB_DAILY = "ecb_daily"  # ECB daily fixing
    BOE_DAILY = "boe_daily"
    FED_NOON = "fed_noon"  # Fed noon rate
    BOJ_TOKYO = "boj_tokyo"
    PBOC_DAILY = "pboc_daily"
    CUSTOM = "custom"


@dataclass
class FXFixingRate:
    """FX fixing rate (#X12)."""
    pair: str
    fixing_type: FixingType
    fixing_date: date
    fixing_time: time
    rate: float
    source: str

    def to_dict(self) -> dict:
        return {
            'pair': self.pair,
            'fixing_type': self.fixing_type.value,
            'fixing_date': self.fixing_date.isoformat(),
            'fixing_time': self.fixing_time.isoformat(),
            'rate': self.rate,
            'source': self.source,
        }


class FXFixingManager:
    """
    FX fixing rate tracking (#X12).

    Tracks official fixing rates for:
    - Portfolio valuation
    - Benchmark comparison
    - Execution analysis
    """

    # Fixing schedules
    FIXING_SCHEDULES = {
        FixingType.WMR_LONDON_4PM: {
            'time': time(16, 0),
            'timezone': 'Europe/London',
            'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD'],
        },
        FixingType.ECB_DAILY: {
            'time': time(14, 15),
            'timezone': 'Europe/Frankfurt',
            'pairs': ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF'],
        },
        FixingType.PBOC_DAILY: {
            'time': time(9, 15),
            'timezone': 'Asia/Shanghai',
            'pairs': ['USDCNY'],
        },
    }

    def __init__(self):
        self._fixings: dict[str, list[FXFixingRate]] = defaultdict(list)
        self._latest: dict[tuple[str, FixingType], FXFixingRate] = {}

    def record_fixing(self, fixing: FXFixingRate) -> None:
        """Record fixing rate (#X12)."""
        key = f"{fixing.pair}_{fixing.fixing_type.value}"
        self._fixings[key].append(fixing)

        # Keep 2 years
        self._fixings[key] = self._fixings[key][-520:]

        # Update latest
        self._latest[(fixing.pair, fixing.fixing_type)] = fixing

    def get_latest_fixing(
        self,
        pair: str,
        fixing_type: FixingType = FixingType.WMR_LONDON_4PM,
    ) -> FXFixingRate | None:
        """Get latest fixing rate."""
        return self._latest.get((pair, fixing_type))

    def get_fixing_history(
        self,
        pair: str,
        fixing_type: FixingType,
        days: int = 30,
    ) -> list[FXFixingRate]:
        """Get fixing rate history."""
        key = f"{pair}_{fixing_type.value}"
        history = self._fixings.get(key, [])
        return history[-days:]

    def calculate_fixing_vs_spot(
        self,
        pair: str,
        spot_rate: float,
        fixing_type: FixingType = FixingType.WMR_LONDON_4PM,
    ) -> dict:
        """
        Compare current spot to last fixing (#X12).

        Returns analysis useful for fixing orders.
        """
        fixing = self.get_latest_fixing(pair, fixing_type)
        if not fixing:
            return {'error': 'no_fixing_available'}

        diff = spot_rate - fixing.rate
        diff_pips = diff * 10000  # Standard pip calculation

        # Historical deviation
        history = self.get_fixing_history(pair, fixing_type, 30)
        if len(history) >= 2:
            # Calculate typical fixing deviation pattern
            # (Would need intraday data for accurate analysis)
            pass

        return {
            'pair': pair,
            'spot_rate': spot_rate,
            'fixing_rate': fixing.rate,
            'fixing_date': fixing.fixing_date.isoformat(),
            'difference': diff,
            'difference_pips': diff_pips,
            'difference_pct': (diff / fixing.rate) * 100,
        }

    def get_upcoming_fixings(self) -> list[dict]:
        """Get upcoming fixing times."""
        now = datetime.now(timezone.utc)
        upcoming = []

        for fixing_type, schedule in self.FIXING_SCHEDULES.items():
            fixing_time = schedule['time']
            # Simplified - would need proper timezone handling
            upcoming.append({
                'fixing_type': fixing_type.value,
                'time': fixing_time.isoformat(),
                'timezone': schedule['timezone'],
                'pairs': schedule['pairs'],
            })

        return upcoming


# =============================================================================
# CARRY TRADE OPTIMIZATION (#X13)
# =============================================================================

@dataclass
class CarryTradeOpportunity:
    """Carry trade opportunity (#X13)."""
    long_currency: str
    short_currency: str
    pair: str
    long_rate: float
    short_rate: float
    carry_bps: float  # Annual carry in bps
    volatility: float
    sharpe_ratio: float
    correlation_to_risk: float  # Correlation to risk-off
    score: float

    def to_dict(self) -> dict:
        return {
            'long_currency': self.long_currency,
            'short_currency': self.short_currency,
            'pair': self.pair,
            'long_rate': self.long_rate,
            'short_rate': self.short_rate,
            'carry_bps': self.carry_bps,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'correlation_to_risk': self.correlation_to_risk,
            'score': self.score,
        }


@dataclass
class CarryPortfolio:
    """Optimized carry portfolio (#X13)."""
    positions: list[dict]  # [{pair, weight, direction}]
    total_carry_bps: float
    portfolio_volatility: float
    portfolio_sharpe: float
    max_drawdown_estimate: float
    diversification_ratio: float

    def to_dict(self) -> dict:
        return {
            'positions': self.positions,
            'total_carry_bps': self.total_carry_bps,
            'portfolio_volatility': self.portfolio_volatility,
            'portfolio_sharpe': self.portfolio_sharpe,
            'max_drawdown_estimate': self.max_drawdown_estimate,
            'diversification_ratio': self.diversification_ratio,
        }


class CarryTradeOptimizer:
    """
    Carry trade optimization (#X13).

    Optimizes carry trades considering:
    - Interest rate differentials
    - Currency volatility
    - Correlations
    - Risk-off sensitivity
    """

    def __init__(self):
        # Interest rates by currency
        self._rates: dict[str, float] = {}

        # Volatilities by pair
        self._volatilities: dict[str, float] = {}

        # Correlation matrix
        self._correlations: dict[tuple[str, str], float] = {}

        # Risk-off correlations
        self._risk_correlations: dict[str, float] = {}

    def update_rate(self, currency: str, rate: float) -> None:
        """Update interest rate for currency."""
        self._rates[currency] = rate

    def update_volatility(self, pair: str, vol: float) -> None:
        """Update volatility for pair."""
        self._volatilities[pair] = vol

    def update_correlation(self, pair1: str, pair2: str, corr: float) -> None:
        """Update correlation between pairs."""
        self._correlations[(pair1, pair2)] = corr
        self._correlations[(pair2, pair1)] = corr

    def update_risk_correlation(self, pair: str, corr: float) -> None:
        """Update correlation to risk-off (e.g., VIX or S&P)."""
        self._risk_correlations[pair] = corr

    def find_opportunities(
        self,
        min_carry_bps: float = 100,
        max_volatility: float = 0.15,
    ) -> list[CarryTradeOpportunity]:
        """
        Find carry trade opportunities (#X13).

        Args:
            min_carry_bps: Minimum annual carry in bps
            max_volatility: Maximum acceptable volatility

        Returns:
            List of opportunities sorted by score
        """
        opportunities = []
        currencies = list(self._rates.keys())

        for long_ccy in currencies:
            for short_ccy in currencies:
                if long_ccy == short_ccy:
                    continue

                long_rate = self._rates[long_ccy]
                short_rate = self._rates[short_ccy]

                # Only consider positive carry (long higher rate)
                if long_rate <= short_rate:
                    continue

                carry_bps = (long_rate - short_rate) * 10000

                if carry_bps < min_carry_bps:
                    continue

                # Get pair volatility
                pair = f"{long_ccy}{short_ccy}"
                reverse_pair = f"{short_ccy}{long_ccy}"
                vol = self._volatilities.get(pair) or self._volatilities.get(reverse_pair, 0.10)

                if vol > max_volatility:
                    continue

                # Calculate Sharpe ratio (annualized)
                carry_return = carry_bps / 10000
                sharpe = carry_return / vol if vol > 0 else 0

                # Get risk correlation
                risk_corr = self._risk_correlations.get(pair, 0)

                # Calculate score (higher is better)
                # Penalize high risk correlation (carry unwinds in risk-off)
                risk_penalty = max(0, risk_corr) * 0.5
                score = sharpe * (1 - risk_penalty)

                opportunities.append(CarryTradeOpportunity(
                    long_currency=long_ccy,
                    short_currency=short_ccy,
                    pair=pair,
                    long_rate=long_rate,
                    short_rate=short_rate,
                    carry_bps=carry_bps,
                    volatility=vol,
                    sharpe_ratio=sharpe,
                    correlation_to_risk=risk_corr,
                    score=score,
                ))

        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities

    def optimize_portfolio(
        self,
        opportunities: list[CarryTradeOpportunity],
        max_positions: int = 5,
        target_volatility: float = 0.10,
    ) -> CarryPortfolio:
        """
        Optimize carry portfolio (#X13).

        Uses simple diversification approach.
        """
        if not opportunities:
            return CarryPortfolio(
                positions=[],
                total_carry_bps=0,
                portfolio_volatility=0,
                portfolio_sharpe=0,
                max_drawdown_estimate=0,
                diversification_ratio=0,
            )

        # Select top opportunities
        selected = opportunities[:max_positions]

        # Simple equal-weight for now
        # Production would use mean-variance optimization
        weight = 1.0 / len(selected)

        positions = []
        total_carry = 0
        weighted_vol_sq = 0

        for opp in selected:
            positions.append({
                'pair': opp.pair,
                'weight': weight,
                'direction': 'long',
                'carry_bps': opp.carry_bps,
            })
            total_carry += weight * opp.carry_bps
            weighted_vol_sq += (weight * opp.volatility) ** 2

        # Add correlation effects (simplified)
        for i, opp1 in enumerate(selected):
            for j, opp2 in enumerate(selected):
                if i >= j:
                    continue
                corr = self._correlations.get((opp1.pair, opp2.pair), 0.5)
                weighted_vol_sq += 2 * weight * weight * opp1.volatility * opp2.volatility * corr

        portfolio_vol = np.sqrt(weighted_vol_sq)

        # Adjust to target volatility
        if portfolio_vol > 0:
            leverage = target_volatility / portfolio_vol
            for pos in positions:
                pos['weight'] *= leverage
            portfolio_vol = target_volatility
            total_carry *= leverage

        # Calculate metrics
        portfolio_sharpe = (total_carry / 10000) / portfolio_vol if portfolio_vol > 0 else 0

        # Estimate max drawdown (assuming normal distribution, ~3 sigma)
        max_dd_estimate = portfolio_vol * 3 * np.sqrt(30/252)  # 30-day horizon

        # Diversification ratio
        standalone_vols = sum(opp.volatility for opp in selected) / len(selected)
        div_ratio = standalone_vols / portfolio_vol if portfolio_vol > 0 else 1

        return CarryPortfolio(
            positions=positions,
            total_carry_bps=total_carry,
            portfolio_volatility=portfolio_vol,
            portfolio_sharpe=portfolio_sharpe,
            max_drawdown_estimate=max_dd_estimate,
            diversification_ratio=div_ratio,
        )

    def get_carry_dashboard(self) -> dict:
        """Get carry trade dashboard (#X13)."""
        opportunities = self.find_opportunities()
        portfolio = self.optimize_portfolio(opportunities)

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'rates': self._rates,
            'top_opportunities': [o.to_dict() for o in opportunities[:10]],
            'optimal_portfolio': portfolio.to_dict(),
            'market_conditions': {
                'avg_g10_rate': np.mean(list(self._rates.values())) if self._rates else 0,
                'rate_dispersion': np.std(list(self._rates.values())) if len(self._rates) > 1 else 0,
            },
        }

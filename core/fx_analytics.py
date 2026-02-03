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
# FX PIP CALCULATION HELPERS (P0-5 Fix)
# =============================================================================

def get_pip_multiplier(pair: str) -> int:
    """
    Get pip multiplier for FX pair.

    For JPY pairs (USDJPY, EURJPY, etc.): 1 pip = 0.01, so multiplier = 100
    For other pairs (EURUSD, GBPUSD, etc.): 1 pip = 0.0001, so multiplier = 10000

    Args:
        pair: Currency pair (e.g., "USDJPY", "EURUSD")

    Returns:
        Pip multiplier (100 for JPY pairs, 10000 for others)
    """
    if "JPY" in pair.upper():
        return 100
    return 10000


def get_pip_size(pair: str) -> float:
    """
    Get pip size for FX pair.

    For JPY pairs: 1 pip = 0.01
    For other pairs: 1 pip = 0.0001

    Args:
        pair: Currency pair (e.g., "USDJPY", "EURUSD")

    Returns:
        Pip size as decimal
    """
    if "JPY" in pair.upper():
        return 0.01
    return 0.0001


def calculate_pip_value(
    pair: str,
    lot_size: float,
    spot_rate: float,
    account_currency: str = "USD",
) -> float:
    """
    Calculate pip value for FX trade (FX-P0-3 Fix).

    For direct pairs (XXXUSD): pip_value = pip_size * lot_size
    For indirect pairs (USDXXX): pip_value = pip_size * lot_size / spot_rate
    For cross-rates (EUR/GBP, etc.): pip_value = pip_size * lot_size / spot_rate
        (NOT * spot_rate as incorrectly done in some implementations)

    Args:
        pair: Currency pair (e.g., "EURUSD", "GBPJPY", "EURGBP")
        lot_size: Trade size in base currency units
        spot_rate: Current spot rate
        account_currency: Account denomination currency

    Returns:
        Pip value in account currency
    """
    pair_upper = pair.upper()
    pip_size = get_pip_size(pair)

    # Extract quote currency (last 3 chars)
    quote_ccy = pair_upper[-3:]
    base_ccy = pair_upper[:3]

    # Guard against zero/invalid spot rate
    if spot_rate <= 0:
        return 0.0

    if quote_ccy == account_currency:
        # Direct pair: pip value in account currency directly
        # e.g., EURUSD with USD account: pip_value = 0.0001 * lot_size
        pip_value = pip_size * lot_size
    elif base_ccy == account_currency:
        # Indirect pair: need to convert
        # e.g., USDCAD with USD account: pip_value = 0.0001 * lot_size / spot_rate
        pip_value = pip_size * lot_size / spot_rate
    else:
        # Cross-rate (e.g., EURGBP with USD account)
        # FX-P0-3: For crosses, pip value = pip_size * lot_size / spot_rate
        # This is because we need to convert from quote currency to account currency
        # The spot_rate here represents quote_ccy/account_ccy conversion
        pip_value = pip_size * lot_size / spot_rate

    return pip_value


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


class DeltaConvention(str, Enum):
    """
    FX delta convention types (FX-P0-2 Fix).

    spot: Spot delta (most common in G10)
    forward: Forward delta (used in some EM pairs)
    spot_pa: Spot premium-adjusted delta (common in USDJPY options)
    forward_pa: Forward premium-adjusted delta
    """
    SPOT = "spot"
    FORWARD = "forward"
    SPOT_PA = "spot_pa"  # Premium-adjusted spot
    FORWARD_PA = "forward_pa"  # Premium-adjusted forward


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
    delta_convention: str = "spot"  # FX-P0-2: Delta convention parameter

    @property
    def skew(self) -> float:
        """Get smile skew (positive = call premium)."""
        return self.rr_25d

    @property
    def smile_curvature(self) -> float:
        """Get smile curvature (convexity)."""
        return self.bf_25d

    def get_vol_at_delta(
        self,
        delta: float,
        delta_convention: str | None = None,
    ) -> float:
        """
        Interpolate volatility at specific delta using FX market conventions (FX-P0-1 Fix).

        FX vol smiles are quoted in delta space, not moneyness.
        Standard delta points: 10D put, 25D put, ATM, 25D call, 10D call

        Uses the Malz formula for FX smile interpolation:
        sigma(delta) = ATM + 2*RR*(delta - 0.5) + 16*BF*(delta - 0.5)^2

        Args:
            delta: Delta value (0-1 for calls, where 0.5 is ATM)
            delta_convention: Override default delta convention

        Returns:
            Interpolated implied volatility
        """
        convention = delta_convention or self.delta_convention

        # ATM check
        if abs(delta - 0.5) < 0.01:
            return self.atm_vol

        # Determine which RR/BF to use based on delta
        # Use 10D quotes for extreme deltas, 25D for moderate
        if abs(delta - 0.5) > 0.35:  # Close to 10D or 90D
            rr = self.rr_10d
            bf = self.bf_10d
            # Scale factor for interpolation between 25D and 10D
            scale = (abs(delta - 0.5) - 0.25) / 0.15  # 0 at 25D, 1 at 10D
            scale = max(0, min(1, scale))
            rr = self.rr_25d * (1 - scale) + self.rr_10d * scale
            bf = self.bf_25d * (1 - scale) + self.bf_10d * scale
        else:
            rr = self.rr_25d
            bf = self.bf_25d

        # Malz formula for FX smile (delta-based, not moneyness-based)
        # sigma(delta) = ATM + 2*RR*(delta - 0.5) + 16*BF*(delta - 0.5)^2
        d = delta - 0.5
        vol = self.atm_vol + 2 * rr * d + 16 * bf * (d ** 2)

        # Apply convention adjustment if needed
        # Premium-adjusted delta shifts the smile slightly
        if convention in (DeltaConvention.SPOT_PA.value, DeltaConvention.FORWARD_PA.value):
            # Premium adjustment typically reduces wing vols slightly
            pa_adjustment = 1.0 - 0.02 * abs(d)  # Small reduction in wings
            vol = vol * pa_adjustment

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

        # Defensive assertion to guard against division by zero
        assert upper.expiry_days != lower.expiry_days, "Expiry days should differ for interpolation"
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
            # Guard against empty list (should not happen due to len check, but defensive)
            if len(hist_rrs) > 0:
                rr_percentile = sum(1 for r in hist_rrs if r < smile.rr_25d) / len(hist_rrs) * 100
            else:
                rr_percentile = 50.0  # Neutral default

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
                # Guard against division by zero in log
                prev_price = history[-20][1]
                if prev_price > 0 and current_price > 0:
                    recent_return = np.log(current_price / prev_price)
                    std_return = np.std(returns)

                    # Guard against zero std
                    if std_return > 1e-12 and abs(recent_return) > 3 * std_return:
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
        diff_pips = diff * get_pip_multiplier(pair)  # P0-5: Use proper multiplier for JPY pairs

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
            'difference_pct': (diff / fixing.rate) * 100 if fixing.rate != 0 else 0.0,
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


# =============================================================================
# FORWARD POINTS CALCULATION (P2)
# =============================================================================

@dataclass
class ForwardPointsResult:
    """Result of forward points calculation (P2)."""
    pair: str
    spot_rate: float
    forward_rate: float
    forward_points: float
    forward_points_pips: float
    tenor_days: int
    domestic_rate: float
    foreign_rate: float
    annualized_forward_premium_pct: float
    swap_points_per_day: float

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "spot_rate": self.spot_rate,
            "forward_rate": self.forward_rate,
            "forward_points": self.forward_points,
            "forward_points_pips": self.forward_points_pips,
            "tenor_days": self.tenor_days,
            "domestic_rate": self.domestic_rate,
            "foreign_rate": self.foreign_rate,
            "annualized_forward_premium_pct": self.annualized_forward_premium_pct,
            "swap_points_per_day": self.swap_points_per_day,
        }


class ForwardPointsCalculator:
    """
    FX Forward Points Calculator (P2).

    Calculates forward points based on interest rate differentials
    using covered interest rate parity:

    F = S * (1 + r_d * T) / (1 + r_f * T)

    Where:
    - F = Forward rate
    - S = Spot rate
    - r_d = Domestic (quote) currency interest rate
    - r_f = Foreign (base) currency interest rate
    - T = Time to maturity in years

    Forward points = F - S
    """

    # Standard FX tenors in days
    STANDARD_TENORS = {
        "ON": 1,  # Overnight
        "TN": 2,  # Tomorrow-next
        "SN": 3,  # Spot-next
        "1W": 7,
        "2W": 14,
        "1M": 30,
        "2M": 60,
        "3M": 90,
        "6M": 180,
        "9M": 270,
        "1Y": 365,
    }

    def __init__(self):
        # Interest rates by currency (annualized, decimal form)
        self._rates: dict[str, float] = {}

    def update_rate(self, currency: str, rate: float) -> None:
        """Update interest rate for currency (annualized, decimal)."""
        self._rates[currency] = rate

    def calculate_forward_points(
        self,
        pair: str,
        spot_rate: float,
        tenor_days: int | None = None,
        tenor_name: str | None = None,
        domestic_rate: float | None = None,
        foreign_rate: float | None = None,
    ) -> ForwardPointsResult:
        """
        Calculate forward points for FX pair (P2).

        Args:
            pair: Currency pair (e.g., "EURUSD" where EUR is base, USD is quote/domestic)
            spot_rate: Current spot rate
            tenor_days: Days to forward date (or use tenor_name)
            tenor_name: Standard tenor name (e.g., "1M", "3M")
            domestic_rate: Override domestic (quote) currency rate
            foreign_rate: Override foreign (base) currency rate

        Returns:
            ForwardPointsResult with forward rate and points
        """
        # Parse currencies from pair
        base_ccy = pair[:3].upper()  # Foreign currency
        quote_ccy = pair[3:].upper()  # Domestic currency

        # Get tenor in days
        if tenor_days is None:
            if tenor_name and tenor_name in self.STANDARD_TENORS:
                tenor_days = self.STANDARD_TENORS[tenor_name]
            else:
                tenor_days = 30  # Default to 1 month

        # Get interest rates
        r_d = domestic_rate if domestic_rate is not None else self._rates.get(quote_ccy, 0.05)
        r_f = foreign_rate if foreign_rate is not None else self._rates.get(base_ccy, 0.04)

        # Time to maturity in years
        T = tenor_days / 365.0

        # Forward rate using interest rate parity
        # F = S * (1 + r_d * T) / (1 + r_f * T)
        forward_rate = spot_rate * (1 + r_d * T) / (1 + r_f * T)

        # Forward points
        forward_points = forward_rate - spot_rate

        # Convert to pips
        pip_multiplier = get_pip_multiplier(pair)
        forward_points_pips = forward_points * pip_multiplier

        # Annualized forward premium/discount
        if spot_rate > 0 and T > 0:
            annualized_premium = (forward_rate / spot_rate - 1) / T * 100
        else:
            annualized_premium = 0.0

        # Swap points per day
        swap_points_per_day = forward_points_pips / tenor_days if tenor_days > 0 else 0

        return ForwardPointsResult(
            pair=pair,
            spot_rate=spot_rate,
            forward_rate=forward_rate,
            forward_points=forward_points,
            forward_points_pips=forward_points_pips,
            tenor_days=tenor_days,
            domestic_rate=r_d,
            foreign_rate=r_f,
            annualized_forward_premium_pct=annualized_premium,
            swap_points_per_day=swap_points_per_day,
        )

    def calculate_forward_curve(
        self,
        pair: str,
        spot_rate: float,
    ) -> dict[str, ForwardPointsResult]:
        """
        Calculate forward points for all standard tenors (P2).

        Args:
            pair: Currency pair
            spot_rate: Current spot rate

        Returns:
            Dictionary of tenor -> ForwardPointsResult
        """
        curve = {}
        for tenor_name, tenor_days in self.STANDARD_TENORS.items():
            result = self.calculate_forward_points(
                pair=pair,
                spot_rate=spot_rate,
                tenor_days=tenor_days,
            )
            curve[tenor_name] = result
        return curve

    def get_forward_curve_summary(self, pair: str, spot_rate: float) -> dict:
        """Get summary of forward curve for display."""
        curve = self.calculate_forward_curve(pair, spot_rate)
        return {
            "pair": pair,
            "spot": spot_rate,
            "forwards": {
                tenor: {
                    "rate": result.forward_rate,
                    "points_pips": result.forward_points_pips,
                    "premium_pct": result.annualized_forward_premium_pct,
                }
                for tenor, result in curve.items()
            },
        }


# =============================================================================
# INTEREST RATE PARITY CHECK (P2)
# =============================================================================

@dataclass
class IRPCheckResult:
    """Result of interest rate parity check (P2)."""
    pair: str
    spot_rate: float
    forward_rate: float
    theoretical_forward: float
    deviation: float
    deviation_pips: float
    deviation_pct: float
    domestic_rate: float
    foreign_rate: float
    tenor_days: int
    arbitrage_opportunity: bool
    arbitrage_type: str | None
    arbitrage_profit_bps: float
    is_covered_irp_holding: bool

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "spot_rate": self.spot_rate,
            "forward_rate": self.forward_rate,
            "theoretical_forward": self.theoretical_forward,
            "deviation": self.deviation,
            "deviation_pips": self.deviation_pips,
            "deviation_pct": self.deviation_pct,
            "domestic_rate": self.domestic_rate,
            "foreign_rate": self.foreign_rate,
            "tenor_days": self.tenor_days,
            "arbitrage_opportunity": self.arbitrage_opportunity,
            "arbitrage_type": self.arbitrage_type,
            "arbitrage_profit_bps": self.arbitrage_profit_bps,
            "is_covered_irp_holding": self.is_covered_irp_holding,
        }


class InterestRateParityChecker:
    """
    Interest Rate Parity (IRP) Checker (P2).

    Checks whether covered interest rate parity holds:
    F/S = (1 + r_d) / (1 + r_f)

    Deviations may indicate:
    - Arbitrage opportunities
    - Credit/liquidity risk premiums
    - Market stress
    - Data quality issues
    """

    # Typical transaction costs in bps for IRP arbitrage
    TRANSACTION_COST_BPS = 5.0  # Bid-ask spreads, etc.

    def __init__(self):
        self._rates: dict[str, float] = {}
        self._forward_calculator = ForwardPointsCalculator()

    def update_rate(self, currency: str, rate: float) -> None:
        """Update interest rate for currency."""
        self._rates[currency] = rate
        self._forward_calculator.update_rate(currency, rate)

    def check_covered_irp(
        self,
        pair: str,
        spot_rate: float,
        forward_rate: float,
        tenor_days: int = 90,
        domestic_rate: float | None = None,
        foreign_rate: float | None = None,
        arbitrage_threshold_bps: float = 10.0,
    ) -> IRPCheckResult:
        """
        Check if covered interest rate parity holds (P2).

        Args:
            pair: Currency pair
            spot_rate: Current spot rate
            forward_rate: Market quoted forward rate
            tenor_days: Forward tenor in days
            domestic_rate: Quote currency rate (override)
            foreign_rate: Base currency rate (override)
            arbitrage_threshold_bps: Minimum deviation for arbitrage signal

        Returns:
            IRPCheckResult with IRP analysis
        """
        base_ccy = pair[:3].upper()
        quote_ccy = pair[3:].upper()

        r_d = domestic_rate if domestic_rate is not None else self._rates.get(quote_ccy, 0.05)
        r_f = foreign_rate if foreign_rate is not None else self._rates.get(base_ccy, 0.04)

        T = tenor_days / 365.0

        # Calculate theoretical forward from IRP
        theoretical_forward = spot_rate * (1 + r_d * T) / (1 + r_f * T)

        # Calculate deviation
        deviation = forward_rate - theoretical_forward
        pip_multiplier = get_pip_multiplier(pair)
        deviation_pips = deviation * pip_multiplier
        deviation_pct = (deviation / spot_rate) * 100 if spot_rate > 0 else 0

        # Check for arbitrage opportunity
        deviation_bps = abs(deviation_pct) * 100  # Convert to basis points
        net_profit_bps = deviation_bps - self.TRANSACTION_COST_BPS

        arbitrage_opportunity = net_profit_bps > arbitrage_threshold_bps

        # Determine arbitrage type
        arbitrage_type = None
        if arbitrage_opportunity:
            if deviation > 0:
                # Forward is overpriced relative to IRP
                # Strategy: Sell forward, borrow domestic, buy spot, invest foreign
                arbitrage_type = "SELL_FORWARD_OVERPRICED"
            else:
                # Forward is underpriced relative to IRP
                # Strategy: Buy forward, borrow foreign, sell spot, invest domestic
                arbitrage_type = "BUY_FORWARD_UNDERPRICED"

        # Is IRP holding within reasonable tolerance?
        is_irp_holding = deviation_bps < arbitrage_threshold_bps

        return IRPCheckResult(
            pair=pair,
            spot_rate=spot_rate,
            forward_rate=forward_rate,
            theoretical_forward=theoretical_forward,
            deviation=deviation,
            deviation_pips=deviation_pips,
            deviation_pct=deviation_pct,
            domestic_rate=r_d,
            foreign_rate=r_f,
            tenor_days=tenor_days,
            arbitrage_opportunity=arbitrage_opportunity,
            arbitrage_type=arbitrage_type,
            arbitrage_profit_bps=max(0, net_profit_bps),
            is_covered_irp_holding=is_irp_holding,
        )

    def check_uncovered_irp(
        self,
        pair: str,
        spot_rate: float,
        expected_spot_rate: float,
        tenor_days: int = 90,
        domestic_rate: float | None = None,
        foreign_rate: float | None = None,
    ) -> dict:
        """
        Check uncovered interest rate parity (P2).

        Uncovered IRP: E[S_t+1] / S_t = (1 + r_d) / (1 + r_f)

        This is an equilibrium condition that may not hold due to
        risk premiums (forward premium puzzle).

        Args:
            pair: Currency pair
            spot_rate: Current spot rate
            expected_spot_rate: Expected future spot rate
            tenor_days: Horizon in days
            domestic_rate: Quote currency rate
            foreign_rate: Base currency rate

        Returns:
            Dictionary with uncovered IRP analysis
        """
        base_ccy = pair[:3].upper()
        quote_ccy = pair[3:].upper()

        r_d = domestic_rate if domestic_rate is not None else self._rates.get(quote_ccy, 0.05)
        r_f = foreign_rate if foreign_rate is not None else self._rates.get(base_ccy, 0.04)

        T = tenor_days / 365.0

        # Theoretical expected spot from uncovered IRP
        theoretical_expected = spot_rate * (1 + r_d * T) / (1 + r_f * T)

        # Expected appreciation
        expected_change_pct = (expected_spot_rate / spot_rate - 1) * 100 if spot_rate > 0 else 0
        irp_implied_change_pct = (theoretical_expected / spot_rate - 1) * 100 if spot_rate > 0 else 0

        # Risk premium (deviation from UIP)
        risk_premium_pct = expected_change_pct - irp_implied_change_pct

        return {
            "pair": pair,
            "spot_rate": spot_rate,
            "expected_spot_rate": expected_spot_rate,
            "irp_implied_spot": theoretical_expected,
            "expected_change_pct": expected_change_pct,
            "irp_implied_change_pct": irp_implied_change_pct,
            "risk_premium_pct": risk_premium_pct,
            "carry_trade_favorable": (r_f > r_d) and (expected_change_pct > -abs(r_f - r_d) * T * 100),
            "tenor_days": tenor_days,
        }

    def scan_irp_deviations(
        self,
        pairs: list[str],
        spot_rates: dict[str, float],
        forward_rates: dict[str, float],
        tenor_days: int = 90,
    ) -> dict:
        """
        Scan multiple pairs for IRP deviations (P2).

        Args:
            pairs: List of currency pairs
            spot_rates: Dictionary of pair -> spot rate
            forward_rates: Dictionary of pair -> forward rate
            tenor_days: Forward tenor

        Returns:
            Dictionary with scan results and opportunities
        """
        results = []
        opportunities = []

        for pair in pairs:
            spot = spot_rates.get(pair)
            forward = forward_rates.get(pair)

            if spot and forward:
                result = self.check_covered_irp(pair, spot, forward, tenor_days)
                results.append(result.to_dict())

                if result.arbitrage_opportunity:
                    opportunities.append({
                        "pair": pair,
                        "type": result.arbitrage_type,
                        "profit_bps": result.arbitrage_profit_bps,
                        "deviation_pips": result.deviation_pips,
                    })

        # Sort opportunities by profit
        opportunities.sort(key=lambda x: x["profit_bps"], reverse=True)

        return {
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "pairs_scanned": len(results),
            "tenor_days": tenor_days,
            "results": results,
            "opportunities": opportunities,
            "total_opportunities": len(opportunities),
        }


# =============================================================================
# ENHANCED CARRY TRADE ANALYTICS (P2)
# =============================================================================

@dataclass
class CarryTradeAnalysis:
    """Detailed carry trade analysis (P2)."""
    pair: str
    direction: str  # "long" or "short"
    carry_bps_annual: float
    carry_bps_daily: float
    break_even_move_pct: float
    holding_period_days: int
    expected_return_pct: float
    volatility: float
    sharpe_ratio: float
    max_adverse_move_1std: float
    risk_reward_ratio: float

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "direction": self.direction,
            "carry_bps_annual": self.carry_bps_annual,
            "carry_bps_daily": self.carry_bps_daily,
            "break_even_move_pct": self.break_even_move_pct,
            "holding_period_days": self.holding_period_days,
            "expected_return_pct": self.expected_return_pct,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_adverse_move_1std": self.max_adverse_move_1std,
            "risk_reward_ratio": self.risk_reward_ratio,
        }


class CarryTradeOptimizer:
    """
    Carry trade optimization (#X13).

    Optimizes carry trades considering:
    - Interest rate differentials
    - Currency volatility
    - Correlations
    - Risk-off sensitivity

    Enhanced with P2 features:
    - Forward points integration
    - IRP-based carry calculation
    - Break-even analysis
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

        # P2: Forward points and IRP checkers
        self._forward_calculator = ForwardPointsCalculator()
        self._irp_checker = InterestRateParityChecker()

    def update_rate(self, currency: str, rate: float) -> None:
        """Update interest rate for currency."""
        self._rates[currency] = rate
        # P2: Sync with forward calculator and IRP checker
        self._forward_calculator.update_rate(currency, rate)
        self._irp_checker.update_rate(currency, rate)

    def update_volatility(self, pair: str, vol: float) -> None:
        """Update volatility for pair."""
        self._volatilities[pair] = vol

    def update_correlation(self, pair1: str, pair2: str, corr: float) -> None:
        """
        Update correlation between pairs (FX-P1-1 Fix).

        Ensures symmetry by updating both (pair1, pair2) and (pair2, pair1).
        """
        # Clip correlation to valid range [-1, 1]
        corr = max(-1.0, min(1.0, corr))
        self._correlations[(pair1, pair2)] = corr
        self._correlations[(pair2, pair1)] = corr

    def get_correlation_matrix(self, pairs: list[str]) -> np.ndarray:
        """
        Get correlation matrix for specified pairs (FX-P1-1 Fix).

        Returns a guaranteed symmetric positive semi-definite correlation matrix.

        Args:
            pairs: List of pair names

        Returns:
            Symmetric correlation matrix as numpy array
        """
        n = len(pairs)
        corr_matrix = np.eye(n)  # Start with identity (diagonal = 1)

        for i in range(n):
            for j in range(i + 1, n):
                corr = self._correlations.get((pairs[i], pairs[j]), 0.0)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # Ensure symmetry

        # FX-P1-1: Force symmetry after construction (defensive)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2

        # Ensure positive semi-definiteness (eigenvalue adjustment if needed)
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        if np.min(eigenvalues) < 0:
            # Add small value to diagonal to make PSD
            adjustment = abs(np.min(eigenvalues)) + 1e-6
            corr_matrix += adjustment * np.eye(n)
            # Re-normalize diagonal to 1
            diag = np.sqrt(np.diag(corr_matrix))
            corr_matrix = corr_matrix / np.outer(diag, diag)

        return corr_matrix

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

    # =========================================================================
    # P2: ENHANCED CARRY TRADE ANALYTICS
    # =========================================================================

    def analyze_carry_trade(
        self,
        pair: str,
        spot_rate: float,
        holding_period_days: int = 30,
    ) -> CarryTradeAnalysis:
        """
        Detailed carry trade analysis with break-even calculation (P2).

        Args:
            pair: Currency pair
            spot_rate: Current spot rate
            holding_period_days: Expected holding period

        Returns:
            CarryTradeAnalysis with detailed metrics
        """
        base_ccy = pair[:3].upper()
        quote_ccy = pair[3:].upper()

        # Get rates
        base_rate = self._rates.get(base_ccy, 0.0)
        quote_rate = self._rates.get(quote_ccy, 0.0)

        # Determine direction (long high yield, short low yield)
        if base_rate > quote_rate:
            # Long the pair (long base currency which has higher rate)
            direction = "long"
            carry_bps = (base_rate - quote_rate) * 10000
        else:
            # Short the pair (short base currency which has lower rate)
            direction = "short"
            carry_bps = (quote_rate - base_rate) * 10000

        # Annualized and daily carry
        carry_bps_daily = carry_bps / 365

        # Holding period carry
        holding_carry_bps = carry_bps_daily * holding_period_days

        # Get volatility
        vol = self._volatilities.get(pair, 0.10)

        # Break-even move (how much spot can move against before losing carry)
        break_even_move_pct = holding_carry_bps / 100  # Convert bps to pct

        # Expected return (assuming no spot movement)
        expected_return_pct = holding_carry_bps / 100

        # Sharpe ratio (annualized)
        sharpe = (carry_bps / 10000) / vol if vol > 0 else 0

        # Maximum adverse move at 1 standard deviation
        holding_vol = vol * np.sqrt(holding_period_days / 365)
        max_adverse_1std = holding_vol * 100  # As percentage

        # Risk/reward ratio
        risk_reward = break_even_move_pct / max_adverse_1std if max_adverse_1std > 0 else 0

        return CarryTradeAnalysis(
            pair=pair,
            direction=direction,
            carry_bps_annual=carry_bps,
            carry_bps_daily=carry_bps_daily,
            break_even_move_pct=break_even_move_pct,
            holding_period_days=holding_period_days,
            expected_return_pct=expected_return_pct,
            volatility=vol,
            sharpe_ratio=sharpe,
            max_adverse_move_1std=max_adverse_1std,
            risk_reward_ratio=risk_reward,
        )

    def calculate_roll_cost(
        self,
        pair: str,
        spot_rate: float,
        position_size: float,
        holding_period_days: int = 30,
    ) -> dict:
        """
        Calculate roll cost/benefit for carry position (P2).

        Roll cost is incurred when rolling forward contracts.
        For spot positions, this represents the swap points paid/received.

        Args:
            pair: Currency pair
            spot_rate: Current spot rate
            position_size: Position size in base currency
            holding_period_days: Holding period

        Returns:
            Dictionary with roll cost analysis
        """
        # Get forward points
        forward_result = self._forward_calculator.calculate_forward_points(
            pair=pair,
            spot_rate=spot_rate,
            tenor_days=holding_period_days,
        )

        # Calculate roll P&L
        # Long position: pay forward premium (if positive), receive discount (if negative)
        # Short position: opposite
        pip_value = calculate_pip_value(pair, position_size, spot_rate)
        roll_pips = forward_result.forward_points_pips

        # Roll cost in account currency
        roll_cost = roll_pips * pip_value

        return {
            "pair": pair,
            "spot_rate": spot_rate,
            "forward_rate": forward_result.forward_rate,
            "forward_points_pips": roll_pips,
            "holding_period_days": holding_period_days,
            "position_size": position_size,
            "roll_cost_long": -roll_cost,  # Long pays positive points
            "roll_cost_short": roll_cost,  # Short receives positive points
            "roll_cost_per_day": roll_cost / holding_period_days if holding_period_days > 0 else 0,
            "annualized_cost_pct": forward_result.annualized_forward_premium_pct,
        }

    def get_carry_trade_summary(
        self,
        pairs: list[str],
        spot_rates: dict[str, float],
        holding_period_days: int = 30,
    ) -> dict:
        """
        Get comprehensive carry trade summary (P2).

        Args:
            pairs: List of currency pairs
            spot_rates: Dictionary of pair -> spot rate
            holding_period_days: Holding period

        Returns:
            Dictionary with carry trade analysis for all pairs
        """
        analyses = []

        for pair in pairs:
            spot = spot_rates.get(pair)
            if spot:
                analysis = self.analyze_carry_trade(pair, spot, holding_period_days)
                analyses.append(analysis.to_dict())

        # Sort by Sharpe ratio
        analyses.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

        # Find best opportunities
        best_long = [a for a in analyses if a["direction"] == "long"][:3]
        best_short = [a for a in analyses if a["direction"] == "short"][:3]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "holding_period_days": holding_period_days,
            "all_analyses": analyses,
            "best_long_carry": best_long,
            "best_short_carry": best_short,
            "summary_statistics": {
                "avg_carry_bps": np.mean([a["carry_bps_annual"] for a in analyses]) if analyses else 0,
                "max_carry_bps": max([a["carry_bps_annual"] for a in analyses]) if analyses else 0,
                "avg_sharpe": np.mean([a["sharpe_ratio"] for a in analyses]) if analyses else 0,
                "pairs_with_positive_carry": len([a for a in analyses if a["carry_bps_annual"] > 0]),
            },
        }

"""
Dollar Index (DXY) Analyzer
===========================

Analysis of the US Dollar Index for macro trading signals.

The DXY is a weighted geometric mean of the US dollar's value
relative to a basket of foreign currencies:
- Euro (EUR) - 57.6%
- Japanese Yen (JPY) - 13.6%
- British Pound (GBP) - 11.9%
- Canadian Dollar (CAD) - 9.1%
- Swedish Krona (SEK) - 4.2%
- Swiss Franc (CHF) - 3.6%

Key relationships:
- Strong DXY typically negative for: Gold, Oil, EM equities, Commodities
- Weak DXY typically positive for: Gold, Oil, EM equities, Commodities
- US equities: Mixed relationship (multinationals hurt by strong USD)

Historical levels (as of 2024):
- DXY ~100: Neutral zone
- DXY >105: Strong dollar
- DXY <95: Weak dollar
- DXY >110: Extreme strength
- DXY <90: Extreme weakness
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# NumPy for numerical operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


logger = logging.getLogger(__name__)


class DXYState(str, Enum):
    """
    Dollar Index regime classification.

    Based on DXY level and momentum.
    """
    EXTREME_STRONG = "extreme_strong"  # DXY > 110
    STRONG = "strong"                   # DXY 105-110
    NEUTRAL = "neutral"                 # DXY 95-105
    WEAK = "weak"                       # DXY 90-95
    EXTREME_WEAK = "extreme_weak"       # DXY < 90


class AssetDXYRelation(str, Enum):
    """
    Relationship type between asset and DXY.
    """
    NEGATIVE = "negative"      # Asset rises when DXY falls
    POSITIVE = "positive"      # Asset rises when DXY rises
    MIXED = "mixed"            # Complex relationship
    UNCORRELATED = "uncorrelated"


@dataclass
class DXYAnalysisResult:
    """Result of DXY analysis."""
    state: DXYState
    current_level: float
    trend_direction: str         # "up", "down", "sideways"
    trend_strength: float        # 0 to 1
    momentum_score: float        # -1 to 1
    ma_20: float | None
    ma_50: float | None
    distance_from_ma_pct: float  # Distance from 20-day MA
    is_extreme: bool
    signal_for_risk_assets: float  # -1 to 1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "current_level": self.current_level,
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "momentum_score": self.momentum_score,
            "ma_20": self.ma_20,
            "ma_50": self.ma_50,
            "distance_from_ma_pct": self.distance_from_ma_pct,
            "is_extreme": self.is_extreme,
            "signal_for_risk_assets": self.signal_for_risk_assets,
            "timestamp": self.timestamp.isoformat(),
        }


# Asset class relationships with DXY
ASSET_DXY_CORRELATIONS: dict[str, tuple[AssetDXYRelation, float]] = {
    # Commodities - generally negative correlation
    "GC": (AssetDXYRelation.NEGATIVE, -0.75),     # Gold
    "SI": (AssetDXYRelation.NEGATIVE, -0.65),     # Silver
    "CL": (AssetDXYRelation.NEGATIVE, -0.50),     # Crude Oil
    "NG": (AssetDXYRelation.NEGATIVE, -0.30),     # Natural Gas
    "GLD": (AssetDXYRelation.NEGATIVE, -0.75),    # Gold ETF
    "SLV": (AssetDXYRelation.NEGATIVE, -0.65),    # Silver ETF
    "USO": (AssetDXYRelation.NEGATIVE, -0.50),    # Oil ETF

    # Agricultural - moderate negative
    "ZC": (AssetDXYRelation.NEGATIVE, -0.35),     # Corn
    "ZW": (AssetDXYRelation.NEGATIVE, -0.35),     # Wheat
    "ZS": (AssetDXYRelation.NEGATIVE, -0.35),     # Soybeans

    # EM equities - negative correlation
    "EEM": (AssetDXYRelation.NEGATIVE, -0.60),    # EM ETF
    "VWO": (AssetDXYRelation.NEGATIVE, -0.60),    # Vanguard EM

    # US equities - mixed relationship
    "SPY": (AssetDXYRelation.MIXED, -0.15),       # S&P 500
    "QQQ": (AssetDXYRelation.MIXED, -0.10),       # Nasdaq
    "DIA": (AssetDXYRelation.MIXED, -0.20),       # Dow Jones

    # US small caps - slightly positive (domestic focus)
    "IWM": (AssetDXYRelation.POSITIVE, 0.15),     # Russell 2000

    # International developed - negative
    "EFA": (AssetDXYRelation.NEGATIVE, -0.45),    # Developed ex-US

    # Fixed income - complex relationship
    "TLT": (AssetDXYRelation.MIXED, 0.10),        # Long-term treasuries
    "HYG": (AssetDXYRelation.MIXED, -0.20),       # High yield bonds

    # Currency pairs (vs USD)
    "EURUSD": (AssetDXYRelation.NEGATIVE, -0.95), # Inverse of DXY dominant
    "GBPUSD": (AssetDXYRelation.NEGATIVE, -0.80),
    "AUDUSD": (AssetDXYRelation.NEGATIVE, -0.70),
    "USDJPY": (AssetDXYRelation.POSITIVE, 0.85),  # USD strength = USDJPY up
    "USDCAD": (AssetDXYRelation.POSITIVE, 0.75),
    "USDCHF": (AssetDXYRelation.POSITIVE, 0.80),
}


def calculate_dxy_trend(
    dxy_prices: list[float] | np.ndarray,
    period: int = 20,
) -> tuple[str, float]:
    """
    Calculate DXY trend direction and strength.

    Uses simple moving average comparison and price momentum.

    Args:
        dxy_prices: Historical DXY prices (most recent last)
        period: Lookback period for trend calculation

    Returns:
        (direction, strength) tuple
        - direction: "up", "down", or "sideways"
        - strength: 0 to 1 measure of trend strength
    """
    if not HAS_NUMPY:
        prices = list(dxy_prices)
    else:
        prices = list(np.asarray(dxy_prices).flatten())

    if len(prices) < period:
        return "sideways", 0.0

    recent_prices = prices[-period:]
    current = prices[-1]

    # Calculate SMA
    sma = sum(recent_prices) / len(recent_prices)

    # Price vs SMA
    pct_from_sma = (current - sma) / sma * 100

    # Calculate slope using linear regression
    if HAS_NUMPY:
        x = np.arange(len(recent_prices))
        coeffs = np.polyfit(x, recent_prices, 1)
        slope = coeffs[0]
        # Normalize slope by average price
        slope_normalized = slope / sma * 252  # Annualized
    else:
        # Simple slope calculation
        slope = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
        slope_normalized = slope / sma * 252

    # Determine direction
    if abs(pct_from_sma) < 0.5 and abs(slope_normalized) < 0.02:
        direction = "sideways"
        strength = 0.1
    elif pct_from_sma > 0 or slope_normalized > 0:
        direction = "up"
        strength = min(1.0, abs(pct_from_sma) / 3 + abs(slope_normalized) / 0.1)
    else:
        direction = "down"
        strength = min(1.0, abs(pct_from_sma) / 3 + abs(slope_normalized) / 0.1)

    return direction, strength


def calculate_dxy_momentum(
    dxy_prices: list[float] | np.ndarray,
    short_period: int = 10,
    long_period: int = 30,
) -> float:
    """
    Calculate DXY momentum score.

    Uses rate of change and moving average crossover.

    Args:
        dxy_prices: Historical DXY prices
        short_period: Short lookback for momentum
        long_period: Long lookback for momentum

    Returns:
        Momentum score from -1 (bearish) to +1 (bullish)
    """
    if not HAS_NUMPY:
        prices = list(dxy_prices)
    else:
        prices = list(np.asarray(dxy_prices).flatten())

    if len(prices) < long_period:
        return 0.0

    current = prices[-1]

    # Rate of change
    short_ago = prices[-short_period] if len(prices) >= short_period else prices[0]
    long_ago = prices[-long_period] if len(prices) >= long_period else prices[0]

    roc_short = (current - short_ago) / short_ago * 100 if short_ago > 0 else 0
    roc_long = (current - long_ago) / long_ago * 100 if long_ago > 0 else 0

    # Moving average comparison
    sma_short = sum(prices[-short_period:]) / short_period
    sma_long = sum(prices[-long_period:]) / long_period

    ma_diff_pct = (sma_short - sma_long) / sma_long * 100 if sma_long > 0 else 0

    # Combine signals
    momentum = (
        roc_short * 0.4 +
        roc_long * 0.3 +
        ma_diff_pct * 0.3
    )

    # Normalize to [-1, 1]
    momentum = max(-1.0, min(1.0, momentum / 5.0))

    return momentum


def get_dxy_regime(
    dxy_price: float,
    dxy_ma: float | None = None,
) -> DXYState:
    """
    Get current DXY regime based on level.

    Args:
        dxy_price: Current DXY level
        dxy_ma: Optional moving average for context

    Returns:
        DXYState classification
    """
    # Historical level-based classification
    if dxy_price > 110:
        return DXYState.EXTREME_STRONG
    elif dxy_price > 105:
        return DXYState.STRONG
    elif dxy_price > 95:
        return DXYState.NEUTRAL
    elif dxy_price > 90:
        return DXYState.WEAK
    else:
        return DXYState.EXTREME_WEAK


def get_asset_correlation(
    asset: str,
    dxy_prices: list[float] | None = None,
    asset_prices: list[float] | None = None,
) -> tuple[AssetDXYRelation, float]:
    """
    Get correlation between asset and DXY.

    If prices are provided, calculates actual correlation.
    Otherwise returns historical average from lookup table.

    Args:
        asset: Asset symbol
        dxy_prices: Optional DXY price history
        asset_prices: Optional asset price history

    Returns:
        (relationship_type, correlation) tuple
    """
    # Try to calculate actual correlation if data provided
    if dxy_prices is not None and asset_prices is not None:
        if HAS_NUMPY and len(dxy_prices) >= 20 and len(asset_prices) >= 20:
            # Use returns for correlation
            dxy_arr = np.asarray(dxy_prices[-252:])
            asset_arr = np.asarray(asset_prices[-252:])

            min_len = min(len(dxy_arr), len(asset_arr))
            dxy_arr = dxy_arr[-min_len:]
            asset_arr = asset_arr[-min_len:]

            if len(dxy_arr) >= 20:
                dxy_returns = np.diff(dxy_arr) / dxy_arr[:-1]
                asset_returns = np.diff(asset_arr) / asset_arr[:-1]

                if len(dxy_returns) > 0 and len(asset_returns) > 0:
                    corr_matrix = np.corrcoef(dxy_returns, asset_returns)
                    corr = float(corr_matrix[0, 1])

                    if np.isfinite(corr):
                        if corr < -0.3:
                            relation = AssetDXYRelation.NEGATIVE
                        elif corr > 0.3:
                            relation = AssetDXYRelation.POSITIVE
                        elif abs(corr) < 0.15:
                            relation = AssetDXYRelation.UNCORRELATED
                        else:
                            relation = AssetDXYRelation.MIXED

                        return relation, corr

    # Fall back to lookup table
    asset_upper = asset.upper()
    if asset_upper in ASSET_DXY_CORRELATIONS:
        return ASSET_DXY_CORRELATIONS[asset_upper]

    # Default for unknown assets
    return AssetDXYRelation.UNCORRELATED, 0.0


def get_dxy_signal_for_asset(
    asset_type: str,
    dxy_state: DXYState,
    dxy_momentum: float,
) -> tuple[float, str]:
    """
    Get DXY-based signal adjustment for an asset type.

    Args:
        asset_type: Type of asset ("gold", "oil", "em_equity", "us_equity", etc.)
        dxy_state: Current DXY regime
        dxy_momentum: DXY momentum score (-1 to 1)

    Returns:
        (signal_adjustment, rationale) tuple
        signal_adjustment: -1 to +1 adjustment for the asset
    """
    # DXY state to signal mapping for negatively correlated assets
    dxy_state_signals = {
        DXYState.EXTREME_STRONG: -0.8,
        DXYState.STRONG: -0.4,
        DXYState.NEUTRAL: 0.0,
        DXYState.WEAK: 0.4,
        DXYState.EXTREME_WEAK: 0.8,
    }

    base_signal = dxy_state_signals.get(dxy_state, 0.0)

    # Adjust for momentum
    momentum_adj = -dxy_momentum * 0.3  # Negative because we want inverse

    # Asset-specific adjustments
    asset_multipliers = {
        "gold": 1.0,          # Strong inverse relationship
        "silver": 0.9,
        "oil": 0.7,
        "commodities": 0.6,
        "em_equity": 0.8,     # EM hurt by strong dollar
        "us_equity": 0.2,     # Mixed relationship
        "us_small_cap": -0.1, # Slightly benefits from strong USD
        "developed_intl": 0.5,
        "forex_em": 0.9,
    }

    multiplier = asset_multipliers.get(asset_type.lower(), 0.3)

    signal = (base_signal + momentum_adj) * multiplier
    signal = max(-1.0, min(1.0, signal))

    # Generate rationale
    dxy_desc = f"DXY {dxy_state.value}"
    if dxy_momentum > 0.3:
        dxy_desc += " with bullish momentum"
    elif dxy_momentum < -0.3:
        dxy_desc += " with bearish momentum"

    if signal > 0.3:
        rationale = f"{dxy_desc} - favorable for {asset_type}"
    elif signal < -0.3:
        rationale = f"{dxy_desc} - unfavorable for {asset_type}"
    else:
        rationale = f"{dxy_desc} - neutral impact on {asset_type}"

    return signal, rationale


class DXYAnalyzer:
    """
    Comprehensive DXY analyzer for macro trading signals.

    Tracks DXY levels, momentum, and regime for signal generation.
    Provides asset-specific signal adjustments based on DXY.
    """

    def __init__(
        self,
        lookback_period: int = 50,
        extreme_threshold_high: float = 110.0,
        extreme_threshold_low: float = 90.0,
    ):
        """
        Initialize DXY analyzer.

        Args:
            lookback_period: Periods for MA and trend calculation
            extreme_threshold_high: DXY level for extreme strength
            extreme_threshold_low: DXY level for extreme weakness
        """
        self.lookback_period = lookback_period
        self.extreme_threshold_high = extreme_threshold_high
        self.extreme_threshold_low = extreme_threshold_low

        # Price history
        self._dxy_history: list[float] = []
        self._asset_histories: dict[str, list[float]] = {}

        # Cached calculations
        self._last_analysis: DXYAnalysisResult | None = None

    def update(self, dxy_price: float) -> None:
        """
        Update with new DXY price.

        Args:
            dxy_price: Latest DXY level
        """
        self._dxy_history.append(dxy_price)

        # Maintain lookback window
        max_history = max(self.lookback_period * 2, 252)
        if len(self._dxy_history) > max_history:
            self._dxy_history = self._dxy_history[-max_history:]

        # Clear cached analysis
        self._last_analysis = None

    def update_asset(self, symbol: str, price: float) -> None:
        """
        Update asset price for correlation tracking.

        Args:
            symbol: Asset symbol
            price: Latest price
        """
        if symbol not in self._asset_histories:
            self._asset_histories[symbol] = []

        self._asset_histories[symbol].append(price)

        # Maintain history
        if len(self._asset_histories[symbol]) > 300:
            self._asset_histories[symbol] = self._asset_histories[symbol][-252:]

    def analyze(self) -> DXYAnalysisResult:
        """
        Perform full DXY analysis.

        Returns:
            DXYAnalysisResult with complete analysis
        """
        if not self._dxy_history:
            raise ValueError("No DXY data available")

        current = self._dxy_history[-1]

        # Calculate trend
        trend_dir, trend_strength = calculate_dxy_trend(
            self._dxy_history, self.lookback_period
        )

        # Calculate momentum
        momentum = calculate_dxy_momentum(self._dxy_history)

        # Get regime
        state = get_dxy_regime(current)

        # Calculate moving averages
        ma_20 = None
        ma_50 = None
        if len(self._dxy_history) >= 20:
            ma_20 = sum(self._dxy_history[-20:]) / 20
        if len(self._dxy_history) >= 50:
            ma_50 = sum(self._dxy_history[-50:]) / 50

        # Distance from MA
        distance_from_ma = 0.0
        if ma_20 is not None:
            distance_from_ma = (current - ma_20) / ma_20 * 100

        # Check extremes
        is_extreme = current > self.extreme_threshold_high or current < self.extreme_threshold_low

        # Signal for risk assets (inversely related to DXY strength)
        risk_signal = -momentum * 0.5
        if state == DXYState.EXTREME_STRONG:
            risk_signal -= 0.3
        elif state == DXYState.STRONG:
            risk_signal -= 0.15
        elif state == DXYState.WEAK:
            risk_signal += 0.15
        elif state == DXYState.EXTREME_WEAK:
            risk_signal += 0.3

        risk_signal = max(-1.0, min(1.0, risk_signal))

        result = DXYAnalysisResult(
            state=state,
            current_level=current,
            trend_direction=trend_dir,
            trend_strength=trend_strength,
            momentum_score=momentum,
            ma_20=ma_20,
            ma_50=ma_50,
            distance_from_ma_pct=distance_from_ma,
            is_extreme=is_extreme,
            signal_for_risk_assets=risk_signal,
        )

        self._last_analysis = result
        return result

    def get_signal_for_symbol(
        self,
        symbol: str,
        asset_type: str | None = None,
    ) -> tuple[float, str]:
        """
        Get DXY-based signal for a specific symbol.

        Args:
            symbol: Trading symbol
            asset_type: Optional asset type override

        Returns:
            (signal, rationale) tuple
        """
        if self._last_analysis is None:
            self.analyze()

        analysis = self._last_analysis

        # Get correlation for this symbol
        relation, corr = get_asset_correlation(
            symbol,
            self._dxy_history if len(self._dxy_history) >= 20 else None,
            self._asset_histories.get(symbol),
        )

        # Base signal from DXY momentum and state
        dxy_signal = -analysis.momentum_score  # Inverse for most assets

        # Adjust by correlation
        adjusted_signal = dxy_signal * abs(corr)

        # If positively correlated, flip the signal
        if relation == AssetDXYRelation.POSITIVE:
            adjusted_signal = -adjusted_signal

        # Add state-based adjustment
        state_adj = {
            DXYState.EXTREME_STRONG: -0.2,
            DXYState.STRONG: -0.1,
            DXYState.NEUTRAL: 0.0,
            DXYState.WEAK: 0.1,
            DXYState.EXTREME_WEAK: 0.2,
        }

        if relation == AssetDXYRelation.NEGATIVE:
            adjusted_signal += state_adj.get(analysis.state, 0.0)
        elif relation == AssetDXYRelation.POSITIVE:
            adjusted_signal -= state_adj.get(analysis.state, 0.0)

        adjusted_signal = max(-1.0, min(1.0, adjusted_signal))

        # Generate rationale
        rationale = (
            f"DXY {analysis.state.value} ({analysis.current_level:.1f}), "
            f"momentum {analysis.momentum_score:+.2f}, "
            f"{symbol} correlation: {corr:+.2f}"
        )

        return adjusted_signal, rationale

    def is_favorable_for_gold(self) -> bool:
        """Check if DXY conditions favor gold."""
        if self._last_analysis is None:
            self.analyze()

        return (
            self._last_analysis.state in (DXYState.WEAK, DXYState.EXTREME_WEAK)
            or self._last_analysis.momentum_score < -0.3
        )

    def is_favorable_for_em(self) -> bool:
        """Check if DXY conditions favor emerging markets."""
        if self._last_analysis is None:
            self.analyze()

        return (
            self._last_analysis.state in (DXYState.WEAK, DXYState.EXTREME_WEAK, DXYState.NEUTRAL)
            and self._last_analysis.momentum_score <= 0
        )

    def is_headwind_for_commodities(self) -> bool:
        """Check if strong DXY is headwind for commodities."""
        if self._last_analysis is None:
            self.analyze()

        return (
            self._last_analysis.state in (DXYState.STRONG, DXYState.EXTREME_STRONG)
            or self._last_analysis.momentum_score > 0.3
        )

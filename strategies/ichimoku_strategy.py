"""
Ichimoku Cloud Strategy
=======================

Full implementation of the Ichimoku Kinko Hyo trading system.

MATURITY: BETA
--------------
Status: Complete implementation with all 5 components
- [x] Tenkan-sen (Conversion Line)
- [x] Kijun-sen (Base Line)
- [x] Senkou Span A (Leading Span A)
- [x] Senkou Span B (Leading Span B)
- [x] Chikou Span (Lagging Span)
- [x] Cloud color determination
- [x] TK cross signals
- [x] Price/cloud position analysis
- [x] Chikou confirmation
- [x] Combined signal generation

Production Readiness:
- Unit tests: Comprehensive coverage
- Backtesting: Not yet performed
- Live testing: Not yet performed

Use in production: WITH CAUTION - verify signals manually

References:
- Developed by Goichi Hosoda in late 1930s
- Published in 1968 after 30+ years of testing
- Standard settings: 9, 26, 52 (based on 6-day trading week in Japan)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================


class CloudColor(Enum):
    """Ichimoku cloud color (bullish/bearish)."""
    GREEN = "green"  # Bullish cloud (Senkou A > Senkou B)
    RED = "red"      # Bearish cloud (Senkou A < Senkou B)
    NEUTRAL = "neutral"  # Spans are equal


class PricePosition(Enum):
    """Price position relative to cloud."""
    ABOVE = "above"    # Bullish - price above cloud
    BELOW = "below"    # Bearish - price below cloud
    INSIDE = "inside"  # Neutral - price within cloud


class CrossType(Enum):
    """Type of line cross signal."""
    BULLISH = "bullish"  # Fast line crosses above slow line
    BEARISH = "bearish"  # Fast line crosses below slow line
    NONE = "none"        # No cross


class SignalStrength(Enum):
    """Ichimoku signal strength levels."""
    STRONG = "strong"    # All elements align
    MEDIUM = "medium"    # Most elements align
    WEAK = "weak"        # Few elements align
    NONE = "none"        # No clear signal


@dataclass
class IchimokuCloud:
    """
    Complete Ichimoku Cloud values at a point in time.

    Attributes:
        tenkan_sen: Conversion Line (9-period midpoint)
        kijun_sen: Base Line (26-period midpoint)
        senkou_span_a: Leading Span A ((tenkan + kijun) / 2, displaced 26 forward)
        senkou_span_b: Leading Span B (52-period midpoint, displaced 26 forward)
        chikou_span: Lagging Span (close, displaced 26 backward)
        current_price: Current closing price for context
        timestamp: When these values were calculated
    """
    tenkan_sen: float
    kijun_sen: float
    senkou_span_a: float
    senkou_span_b: float
    chikou_span: float
    current_price: float = 0.0
    timestamp: datetime | None = None

    @property
    def cloud_top(self) -> float:
        """Upper boundary of the cloud."""
        return max(self.senkou_span_a, self.senkou_span_b)

    @property
    def cloud_bottom(self) -> float:
        """Lower boundary of the cloud."""
        return min(self.senkou_span_a, self.senkou_span_b)

    @property
    def cloud_thickness(self) -> float:
        """Thickness of the cloud (momentum indicator)."""
        return abs(self.senkou_span_a - self.senkou_span_b)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenkan_sen": self.tenkan_sen,
            "kijun_sen": self.kijun_sen,
            "senkou_span_a": self.senkou_span_a,
            "senkou_span_b": self.senkou_span_b,
            "chikou_span": self.chikou_span,
            "current_price": self.current_price,
            "cloud_top": self.cloud_top,
            "cloud_bottom": self.cloud_bottom,
            "cloud_thickness": self.cloud_thickness,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class IchimokuSignal:
    """
    Signal output from Ichimoku strategy analysis.
    """
    symbol: str
    direction: str  # "long", "short", "flat"
    strength: SignalStrength
    confidence: float  # 0 to 1

    # Cloud analysis
    cloud: IchimokuCloud
    cloud_color: CloudColor
    price_position: PricePosition

    # Component signals
    tk_cross: CrossType  # Tenkan/Kijun cross
    price_kijun_cross: CrossType  # Price/Kijun cross
    chikou_confirmation: bool  # Chikou above/below price 26 periods ago

    # Signal details
    bullish_signals: int  # Count of bullish conditions
    bearish_signals: int  # Count of bearish conditions
    rationale: str

    # Risk management
    stop_loss_price: float | None = None
    take_profit_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "cloud": self.cloud.to_dict(),
            "cloud_color": self.cloud_color.value,
            "price_position": self.price_position.value,
            "tk_cross": self.tk_cross.value,
            "price_kijun_cross": self.price_kijun_cross.value,
            "chikou_confirmation": self.chikou_confirmation,
            "bullish_signals": self.bullish_signals,
            "bearish_signals": self.bearish_signals,
            "rationale": self.rationale,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
        }


# =============================================================================
# ICHIMOKU COMPONENT CALCULATIONS
# =============================================================================


def calculate_tenkan_sen(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 9
) -> float:
    """
    Calculate Tenkan-sen (Conversion Line).

    The Tenkan-sen is the midpoint of the highest high and lowest low
    over the last 9 periods. It represents short-term equilibrium.

    Formula: (highest high + lowest low) / 2

    Args:
        highs: Array of high prices
        lows: Array of low prices
        period: Lookback period (default 9)

    Returns:
        Tenkan-sen value
    """
    if len(highs) < period or len(lows) < period:
        if len(highs) > 0 and len(lows) > 0:
            return (highs[-1] + lows[-1]) / 2
        return 0.0

    highest_high = np.max(highs[-period:])
    lowest_low = np.min(lows[-period:])

    return float((highest_high + lowest_low) / 2)


def calculate_kijun_sen(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 26
) -> float:
    """
    Calculate Kijun-sen (Base Line).

    The Kijun-sen is the midpoint of the highest high and lowest low
    over the last 26 periods. It represents medium-term equilibrium
    and acts as support/resistance.

    Formula: (highest high + lowest low) / 2

    Args:
        highs: Array of high prices
        lows: Array of low prices
        period: Lookback period (default 26)

    Returns:
        Kijun-sen value
    """
    if len(highs) < period or len(lows) < period:
        effective_period = min(len(highs), len(lows))
        if effective_period > 0:
            highest_high = np.max(highs[-effective_period:])
            lowest_low = np.min(lows[-effective_period:])
            return float((highest_high + lowest_low) / 2)
        return 0.0

    highest_high = np.max(highs[-period:])
    lowest_low = np.min(lows[-period:])

    return float((highest_high + lowest_low) / 2)


def calculate_senkou_span_a(
    tenkan: float,
    kijun: float
) -> float:
    """
    Calculate Senkou Span A (Leading Span A).

    Senkou Span A is the average of Tenkan-sen and Kijun-sen,
    plotted 26 periods ahead. It forms one boundary of the cloud.

    Formula: (Tenkan-sen + Kijun-sen) / 2

    Note: The displacement (plotting ahead) is handled separately.

    Args:
        tenkan: Current Tenkan-sen value
        kijun: Current Kijun-sen value

    Returns:
        Senkou Span A value
    """
    return float((tenkan + kijun) / 2)


def calculate_senkou_span_b(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 52
) -> float:
    """
    Calculate Senkou Span B (Leading Span B).

    Senkou Span B is the midpoint of the highest high and lowest low
    over the last 52 periods, plotted 26 periods ahead.
    It forms the other boundary of the cloud.

    Formula: (highest high + lowest low) / 2

    Args:
        highs: Array of high prices
        lows: Array of low prices
        period: Lookback period (default 52)

    Returns:
        Senkou Span B value
    """
    if len(highs) < period or len(lows) < period:
        effective_period = min(len(highs), len(lows))
        if effective_period > 0:
            highest_high = np.max(highs[-effective_period:])
            lowest_low = np.min(lows[-effective_period:])
            return float((highest_high + lowest_low) / 2)
        return 0.0

    highest_high = np.max(highs[-period:])
    lowest_low = np.min(lows[-period:])

    return float((highest_high + lowest_low) / 2)


def calculate_chikou_span(
    closes: np.ndarray,
    displacement: int = 26
) -> float:
    """
    Calculate Chikou Span (Lagging Span).

    The Chikou Span is simply the current close plotted 26 periods
    back. It's used for confirmation by comparing to past price action.

    Formula: Current close (plotted back 26 periods)

    Note: The displacement (plotting back) is handled separately.
    This returns the current close, which will be compared to
    price 26 periods ago.

    Args:
        closes: Array of closing prices
        displacement: How far back to reference (default 26)

    Returns:
        Current close (to be used as Chikou Span)
    """
    if len(closes) == 0:
        return 0.0

    return float(closes[-1])


def calculate_ichimoku_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26
) -> dict[str, np.ndarray]:
    """
    Calculate complete Ichimoku indicator series.

    Returns arrays for all 5 Ichimoku components, properly aligned
    for charting with displacements.

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        tenkan_period: Tenkan-sen period (default 9)
        kijun_period: Kijun-sen period (default 26)
        senkou_b_period: Senkou Span B period (default 52)
        displacement: Forward/backward displacement (default 26)

    Returns:
        Dictionary with arrays for each component
    """
    n = len(closes)

    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    senkou_a = np.full(n + displacement, np.nan)  # Extended for forward projection
    senkou_b = np.full(n + displacement, np.nan)  # Extended for forward projection
    chikou = np.full(n, np.nan)

    # Calculate Tenkan-sen
    for i in range(tenkan_period - 1, n):
        highest = np.max(highs[i - tenkan_period + 1:i + 1])
        lowest = np.min(lows[i - tenkan_period + 1:i + 1])
        tenkan[i] = (highest + lowest) / 2

    # Calculate Kijun-sen
    for i in range(kijun_period - 1, n):
        highest = np.max(highs[i - kijun_period + 1:i + 1])
        lowest = np.min(lows[i - kijun_period + 1:i + 1])
        kijun[i] = (highest + lowest) / 2

    # Calculate Senkou Span A (displaced forward)
    for i in range(kijun_period - 1, n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i + displacement] = (tenkan[i] + kijun[i]) / 2

    # Calculate Senkou Span B (displaced forward)
    for i in range(senkou_b_period - 1, n):
        highest = np.max(highs[i - senkou_b_period + 1:i + 1])
        lowest = np.min(lows[i - senkou_b_period + 1:i + 1])
        senkou_b[i + displacement] = (highest + lowest) / 2

    # Chikou Span (current close, for comparison with past)
    chikou = closes.copy()

    return {
        "tenkan_sen": tenkan,
        "kijun_sen": kijun,
        "senkou_span_a": senkou_a,
        "senkou_span_b": senkou_b,
        "chikou_span": chikou,
    }


# =============================================================================
# ICHIMOKU STRATEGY CLASS
# =============================================================================


class IchimokuStrategy:
    """
    Ichimoku Cloud Trading Strategy.

    Implements the complete Ichimoku Kinko Hyo system for trend
    identification and signal generation.

    Key Signals:
    1. TK Cross: Tenkan crosses Kijun (momentum)
    2. Price/Cloud: Above = bullish, Below = bearish
    3. Cloud Color: Green = bullish, Red = bearish
    4. Chikou: Above past price = bullish confirmation

    Strong signals occur when all elements align in one direction.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Ichimoku strategy.

        Args:
            config: Configuration dictionary with optional parameters:
                - tenkan_period: Conversion line period (default 9)
                - kijun_period: Base line period (default 26)
                - senkou_b_period: Leading Span B period (default 52)
                - displacement: Cloud displacement (default 26)
                - stop_loss_atr_multiplier: ATR multiple for stops (default 2.0)
                - take_profit_atr_multiplier: ATR multiple for TP (default 3.0)
        """
        config = config or {}

        # Ichimoku periods (standard: 9, 26, 52)
        self._tenkan_period = config.get("tenkan_period", 9)
        self._kijun_period = config.get("kijun_period", 26)
        self._senkou_b_period = config.get("senkou_b_period", 52)
        self._displacement = config.get("displacement", 26)

        # Risk management
        self._stop_loss_atr_mult = config.get("stop_loss_atr_multiplier", 2.0)
        self._take_profit_atr_mult = config.get("take_profit_atr_multiplier", 3.0)
        self._atr_period = config.get("atr_period", 14)

        # Signal thresholds
        self._require_chikou_confirmation = config.get(
            "require_chikou_confirmation", True
        )
        self._min_cloud_thickness_pct = config.get("min_cloud_thickness_pct", 0.005)

    def calculate_cloud(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamp: datetime | None = None
    ) -> IchimokuCloud:
        """
        Calculate current Ichimoku Cloud values.

        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of closing prices
            timestamp: Optional timestamp

        Returns:
            IchimokuCloud with all component values
        """
        # Calculate all components
        tenkan = calculate_tenkan_sen(highs, lows, self._tenkan_period)
        kijun = calculate_kijun_sen(highs, lows, self._kijun_period)
        senkou_a = calculate_senkou_span_a(tenkan, kijun)
        senkou_b = calculate_senkou_span_b(highs, lows, self._senkou_b_period)
        chikou = calculate_chikou_span(closes, self._displacement)

        current_price = closes[-1] if len(closes) > 0 else 0.0

        return IchimokuCloud(
            tenkan_sen=tenkan,
            kijun_sen=kijun,
            senkou_span_a=senkou_a,
            senkou_span_b=senkou_b,
            chikou_span=chikou,
            current_price=current_price,
            timestamp=timestamp or datetime.now(timezone.utc),
        )

    def get_cloud_color(self, cloud: IchimokuCloud) -> CloudColor:
        """
        Determine cloud color (bullish/bearish).

        Green (bullish): Senkou Span A > Senkou Span B
        Red (bearish): Senkou Span A < Senkou Span B

        Args:
            cloud: IchimokuCloud values

        Returns:
            CloudColor enum
        """
        if cloud.senkou_span_a > cloud.senkou_span_b:
            return CloudColor.GREEN
        elif cloud.senkou_span_a < cloud.senkou_span_b:
            return CloudColor.RED
        else:
            return CloudColor.NEUTRAL

    def get_price_cloud_position(
        self,
        price: float,
        cloud: IchimokuCloud
    ) -> PricePosition:
        """
        Determine price position relative to the cloud.

        Above cloud: Bullish - look for long entries
        Below cloud: Bearish - look for short entries
        Inside cloud: Neutral/consolidation - wait for breakout

        Args:
            price: Current price
            cloud: IchimokuCloud values

        Returns:
            PricePosition enum
        """
        cloud_top = cloud.cloud_top
        cloud_bottom = cloud.cloud_bottom

        if price > cloud_top:
            return PricePosition.ABOVE
        elif price < cloud_bottom:
            return PricePosition.BELOW
        else:
            return PricePosition.INSIDE

    def get_tk_cross_signal(
        self,
        current_tenkan: float,
        current_kijun: float,
        prev_tenkan: float,
        prev_kijun: float
    ) -> CrossType:
        """
        Detect Tenkan-sen/Kijun-sen cross signal.

        Bullish Cross: Tenkan crosses above Kijun
        Bearish Cross: Tenkan crosses below Kijun

        The TK cross is a momentum signal, strongest when:
        - Bullish: Above cloud, green cloud
        - Bearish: Below cloud, red cloud

        Args:
            current_tenkan: Current Tenkan-sen value
            current_kijun: Current Kijun-sen value
            prev_tenkan: Previous Tenkan-sen value
            prev_kijun: Previous Kijun-sen value

        Returns:
            CrossType enum
        """
        # Check for bullish cross
        if prev_tenkan <= prev_kijun and current_tenkan > current_kijun:
            return CrossType.BULLISH

        # Check for bearish cross
        if prev_tenkan >= prev_kijun and current_tenkan < current_kijun:
            return CrossType.BEARISH

        return CrossType.NONE

    def get_price_kijun_cross(
        self,
        current_price: float,
        prev_price: float,
        kijun: float
    ) -> CrossType:
        """
        Detect price crossing Kijun-sen (base line).

        The Kijun-sen acts as dynamic support/resistance.
        Price crossing it signals potential trend change.

        Args:
            current_price: Current close
            prev_price: Previous close
            kijun: Current Kijun-sen

        Returns:
            CrossType enum
        """
        if prev_price <= kijun and current_price > kijun:
            return CrossType.BULLISH
        elif prev_price >= kijun and current_price < kijun:
            return CrossType.BEARISH

        return CrossType.NONE

    def get_chikou_confirmation(
        self,
        closes: np.ndarray,
        displacement: int | None = None
    ) -> tuple[bool, str]:
        """
        Check Chikou Span confirmation.

        Chikou Span is the current close plotted 26 periods back.
        For confirmation, we compare current close to price 26 periods ago:
        - Bullish: Current close > close 26 periods ago
        - Bearish: Current close < close 26 periods ago

        Args:
            closes: Array of closing prices
            displacement: Lookback periods (default from config)

        Returns:
            Tuple of (is_bullish_confirmation, reason_string)
        """
        if displacement is None:
            displacement = self._displacement

        if len(closes) < displacement + 1:
            return False, "insufficient_data"

        current_close = closes[-1]
        past_close = closes[-displacement - 1]

        if current_close > past_close:
            return True, "chikou_above_past_price"
        else:
            return False, "chikou_below_past_price"

    def calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int | None = None
    ) -> float:
        """
        Calculate Average True Range for stop placement.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            period: ATR period

        Returns:
            ATR value
        """
        if period is None:
            period = self._atr_period

        if len(closes) < period + 1:
            return 0.0

        # Calculate True Range
        tr_values = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            tr_values.append(max(high_low, high_close, low_close))

        if len(tr_values) < period:
            return np.mean(tr_values) if tr_values else 0.0

        # Simple average for ATR
        return float(np.mean(tr_values[-period:]))

    def generate_signal(
        self,
        symbol: str,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamp: datetime | None = None
    ) -> IchimokuSignal:
        """
        Generate complete Ichimoku signal with all confirmations.

        Signal Strength Levels:
        - Strong: All 5 elements align (TK cross, cloud color, price position,
                  cloud direction, chikou confirmation)
        - Medium: 3-4 elements align
        - Weak: 1-2 elements align
        - None: Conflicting signals

        Args:
            symbol: Trading symbol
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of closing prices
            timestamp: Optional timestamp

        Returns:
            IchimokuSignal with complete analysis
        """
        # Validate input
        min_required = max(self._senkou_b_period, self._displacement) + 1
        if len(closes) < min_required:
            return self._create_flat_signal(
                symbol,
                "insufficient_data",
                timestamp
            )

        # Calculate cloud
        cloud = self.calculate_cloud(highs, lows, closes, timestamp)

        # Get cloud color
        cloud_color = self.get_cloud_color(cloud)

        # Get price position
        current_price = closes[-1]
        price_position = self.get_price_cloud_position(current_price, cloud)

        # Calculate previous cloud for TK cross detection
        if len(closes) >= 2:
            prev_tenkan = calculate_tenkan_sen(
                highs[:-1], lows[:-1], self._tenkan_period
            )
            prev_kijun = calculate_kijun_sen(
                highs[:-1], lows[:-1], self._kijun_period
            )
            prev_price = closes[-2]
        else:
            prev_tenkan = cloud.tenkan_sen
            prev_kijun = cloud.kijun_sen
            prev_price = current_price

        # Get TK cross signal
        tk_cross = self.get_tk_cross_signal(
            cloud.tenkan_sen, cloud.kijun_sen,
            prev_tenkan, prev_kijun
        )

        # Get price/Kijun cross
        price_kijun_cross = self.get_price_kijun_cross(
            current_price, prev_price, cloud.kijun_sen
        )

        # Get Chikou confirmation
        chikou_bullish, chikou_reason = self.get_chikou_confirmation(closes)

        # Count bullish/bearish signals
        bullish_signals = 0
        bearish_signals = 0

        # 1. Cloud color
        if cloud_color == CloudColor.GREEN:
            bullish_signals += 1
        elif cloud_color == CloudColor.RED:
            bearish_signals += 1

        # 2. Price position
        if price_position == PricePosition.ABOVE:
            bullish_signals += 1
        elif price_position == PricePosition.BELOW:
            bearish_signals += 1

        # 3. TK cross
        if tk_cross == CrossType.BULLISH:
            bullish_signals += 1
        elif tk_cross == CrossType.BEARISH:
            bearish_signals += 1

        # 4. Price above/below Kijun
        if current_price > cloud.kijun_sen:
            bullish_signals += 1
        elif current_price < cloud.kijun_sen:
            bearish_signals += 1

        # 5. Chikou confirmation
        if chikou_bullish:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # Determine direction and strength
        direction, strength, confidence, rationale = self._evaluate_signals(
            bullish_signals,
            bearish_signals,
            cloud_color,
            price_position,
            tk_cross,
            chikou_bullish,
        )

        # Calculate stop loss and take profit if directional signal
        stop_loss = None
        take_profit = None

        if direction != "flat":
            atr = self.calculate_atr(highs, lows, closes)
            if atr > 0:
                if direction == "long":
                    stop_loss = current_price - (atr * self._stop_loss_atr_mult)
                    take_profit = current_price + (atr * self._take_profit_atr_mult)
                else:  # short
                    stop_loss = current_price + (atr * self._stop_loss_atr_mult)
                    take_profit = current_price - (atr * self._take_profit_atr_mult)

        return IchimokuSignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            cloud=cloud,
            cloud_color=cloud_color,
            price_position=price_position,
            tk_cross=tk_cross,
            price_kijun_cross=price_kijun_cross,
            chikou_confirmation=chikou_bullish,
            bullish_signals=bullish_signals,
            bearish_signals=bearish_signals,
            rationale=rationale,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
        )

    def _evaluate_signals(
        self,
        bullish: int,
        bearish: int,
        cloud_color: CloudColor,
        price_position: PricePosition,
        tk_cross: CrossType,
        chikou_bullish: bool
    ) -> tuple[str, SignalStrength, float, str]:
        """
        Evaluate combined signals to determine direction and strength.

        Returns:
            Tuple of (direction, strength, confidence, rationale)
        """
        total_signals = 5  # Maximum possible signals

        # Check for strong alignment
        if bullish >= 4:
            if tk_cross == CrossType.BULLISH:
                return (
                    "long",
                    SignalStrength.STRONG,
                    bullish / total_signals,
                    f"Strong bullish: {bullish}/5 signals align with TK bullish cross"
                )
            elif price_position == PricePosition.ABOVE and cloud_color == CloudColor.GREEN:
                return (
                    "long",
                    SignalStrength.STRONG,
                    bullish / total_signals,
                    f"Strong bullish: {bullish}/5 signals, price above green cloud"
                )
            return (
                "long",
                SignalStrength.MEDIUM,
                bullish / total_signals,
                f"Medium bullish: {bullish}/5 signals align"
            )

        elif bearish >= 4:
            if tk_cross == CrossType.BEARISH:
                return (
                    "short",
                    SignalStrength.STRONG,
                    bearish / total_signals,
                    f"Strong bearish: {bearish}/5 signals align with TK bearish cross"
                )
            elif price_position == PricePosition.BELOW and cloud_color == CloudColor.RED:
                return (
                    "short",
                    SignalStrength.STRONG,
                    bearish / total_signals,
                    f"Strong bearish: {bearish}/5 signals, price below red cloud"
                )
            return (
                "short",
                SignalStrength.MEDIUM,
                bearish / total_signals,
                f"Medium bearish: {bearish}/5 signals align"
            )

        elif bullish >= 3 and bearish <= 2:
            if self._require_chikou_confirmation and not chikou_bullish:
                return (
                    "flat",
                    SignalStrength.WEAK,
                    0.3,
                    f"Weak bullish: {bullish}/5 signals but no chikou confirmation"
                )
            return (
                "long",
                SignalStrength.WEAK,
                bullish / total_signals * 0.8,
                f"Weak bullish: {bullish}/5 signals"
            )

        elif bearish >= 3 and bullish <= 2:
            if self._require_chikou_confirmation and chikou_bullish:
                return (
                    "flat",
                    SignalStrength.WEAK,
                    0.3,
                    f"Weak bearish: {bearish}/5 signals but chikou contradicts"
                )
            return (
                "short",
                SignalStrength.WEAK,
                bearish / total_signals * 0.8,
                f"Weak bearish: {bearish}/5 signals"
            )

        # Conflicting or neutral
        return (
            "flat",
            SignalStrength.NONE,
            0.0,
            f"No clear signal: {bullish} bullish vs {bearish} bearish"
        )

    def _create_flat_signal(
        self,
        symbol: str,
        reason: str,
        timestamp: datetime | None
    ) -> IchimokuSignal:
        """Create a flat/neutral signal."""
        empty_cloud = IchimokuCloud(
            tenkan_sen=0.0,
            kijun_sen=0.0,
            senkou_span_a=0.0,
            senkou_span_b=0.0,
            chikou_span=0.0,
            current_price=0.0,
            timestamp=timestamp,
        )

        return IchimokuSignal(
            symbol=symbol,
            direction="flat",
            strength=SignalStrength.NONE,
            confidence=0.0,
            cloud=empty_cloud,
            cloud_color=CloudColor.NEUTRAL,
            price_position=PricePosition.INSIDE,
            tk_cross=CrossType.NONE,
            price_kijun_cross=CrossType.NONE,
            chikou_confirmation=False,
            bullish_signals=0,
            bearish_signals=0,
            rationale=reason,
            stop_loss_price=None,
            take_profit_price=None,
        )

    def analyze(
        self,
        symbol: str,
        ohlc_data: dict[str, np.ndarray],
        timestamp: datetime | None = None
    ) -> IchimokuSignal:
        """
        Analyze OHLC data and generate Ichimoku signal.

        Convenience method that accepts OHLC dictionary.

        Args:
            symbol: Trading symbol
            ohlc_data: Dictionary with 'high', 'low', 'close' arrays
            timestamp: Optional timestamp

        Returns:
            IchimokuSignal with complete analysis
        """
        highs = ohlc_data.get("high", ohlc_data.get("highs", np.array([])))
        lows = ohlc_data.get("low", ohlc_data.get("lows", np.array([])))
        closes = ohlc_data.get("close", ohlc_data.get("closes", np.array([])))

        return self.generate_signal(symbol, highs, lows, closes, timestamp)

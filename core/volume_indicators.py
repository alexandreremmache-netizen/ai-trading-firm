"""
Volume-Weighted Indicators
==========================

Comprehensive volume analysis tools including:
- VWMA (Volume-Weighted Moving Average)
- VWAP (Volume-Weighted Average Price) with session reset
- Volume Profile with POC and Value Area
- OBV (On-Balance Volume)
- Volume RSI
- Climax detection

Designed for institutional-grade technical analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class VolumeProfileResult:
    """Result of volume profile calculation."""
    price_levels: np.ndarray  # Price bin centers
    volume_at_price: np.ndarray  # Volume at each price level
    poc_price: float  # Point of Control (highest volume price)
    poc_volume: float  # Volume at POC
    value_area_high: float  # Upper bound of value area
    value_area_low: float  # Lower bound of value area
    value_area_volume_pct: float  # Percentage of volume in value area
    total_volume: float
    n_bins: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "poc_price": self.poc_price,
            "poc_volume": self.poc_volume,
            "value_area_high": self.value_area_high,
            "value_area_low": self.value_area_low,
            "value_area_volume_pct": self.value_area_volume_pct,
            "total_volume": self.total_volume,
            "n_bins": self.n_bins,
        }


@dataclass
class VWAPResult:
    """Result of VWAP calculation."""
    vwap: float  # Current VWAP
    upper_band: float  # VWAP + n standard deviations
    lower_band: float  # VWAP - n standard deviations
    std_dev: float  # Standard deviation of price from VWAP
    cumulative_volume: float
    cumulative_pv: float  # Cumulative price * volume
    session_date: date | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vwap": self.vwap,
            "upper_band": self.upper_band,
            "lower_band": self.lower_band,
            "std_dev": self.std_dev,
            "cumulative_volume": self.cumulative_volume,
            "session_date": self.session_date.isoformat() if self.session_date else None,
        }


@dataclass
class VolumeClimaxResult:
    """Result of volume climax detection."""
    is_climax: bool
    climax_type: str | None  # "buying_climax", "selling_climax", None
    volume_ratio: float  # Current volume / average volume
    price_range_ratio: float  # Current range / average range
    close_position: float  # Position of close within bar range (0-1)
    strength: float  # Climax strength (0-1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_climax": self.is_climax,
            "climax_type": self.climax_type,
            "volume_ratio": self.volume_ratio,
            "price_range_ratio": self.price_range_ratio,
            "close_position": self.close_position,
            "strength": self.strength,
        }


# =============================================================================
# VWMA - Volume-Weighted Moving Average
# =============================================================================


def calculate_vwma(
    prices: np.ndarray,
    volumes: np.ndarray,
    period: int
) -> float:
    """
    Calculate Volume-Weighted Moving Average (VWMA).

    VWMA gives more weight to periods with higher volume, making it
    more responsive to price moves that occur on high volume.

    Formula: VWMA = sum(price * volume) / sum(volume)

    Args:
        prices: Array of closing prices
        volumes: Array of volumes (same length as prices)
        period: Lookback period for the moving average

    Returns:
        VWMA value for the most recent period

    Raises:
        ValueError: If prices and volumes have different lengths
    """
    if len(prices) != len(volumes):
        raise ValueError(
            f"Prices and volumes must have same length: "
            f"{len(prices)} vs {len(volumes)}"
        )

    if len(prices) < period:
        # Return simple average if insufficient data
        if len(prices) == 0:
            return 0.0
        return float(np.mean(prices))

    # Get last 'period' values
    recent_prices = prices[-period:]
    recent_volumes = volumes[-period:]

    # Handle zero total volume
    total_volume = np.sum(recent_volumes)
    if total_volume <= 0:
        return float(np.mean(recent_prices))

    # Calculate VWMA
    vwma = np.sum(recent_prices * recent_volumes) / total_volume

    return float(vwma)


def calculate_vwma_series(
    prices: np.ndarray,
    volumes: np.ndarray,
    period: int
) -> np.ndarray:
    """
    Calculate VWMA series for entire price array.

    Args:
        prices: Array of closing prices
        volumes: Array of volumes
        period: Lookback period

    Returns:
        Array of VWMA values (NaN for first period-1 values)
    """
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have same length")

    n = len(prices)
    vwma_series = np.full(n, np.nan)

    for i in range(period - 1, n):
        start_idx = i - period + 1
        window_prices = prices[start_idx:i + 1]
        window_volumes = volumes[start_idx:i + 1]

        total_vol = np.sum(window_volumes)
        if total_vol > 0:
            vwma_series[i] = np.sum(window_prices * window_volumes) / total_vol
        else:
            vwma_series[i] = np.mean(window_prices)

    return vwma_series


# =============================================================================
# VWAP - Volume-Weighted Average Price
# =============================================================================


def calculate_vwap(
    prices: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray | list[datetime] | None = None,
    std_dev_bands: float = 2.0
) -> VWAPResult:
    """
    Calculate VWAP (Volume-Weighted Average Price) with optional session reset.

    VWAP is calculated from the start of the trading session and resets
    each new day. It's the benchmark used by institutions for execution quality.

    Formula: VWAP = cumsum(typical_price * volume) / cumsum(volume)
    where typical_price = (high + low + close) / 3

    For simplicity, when only close prices provided:
    VWAP = cumsum(close * volume) / cumsum(volume)

    Args:
        prices: Array of prices (close or typical price)
        volumes: Array of volumes
        timestamps: Optional array of timestamps for session reset
        std_dev_bands: Number of standard deviations for bands

    Returns:
        VWAPResult with VWAP and bands
    """
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have same length")

    if len(prices) == 0:
        return VWAPResult(
            vwap=0.0,
            upper_band=0.0,
            lower_band=0.0,
            std_dev=0.0,
            cumulative_volume=0.0,
            cumulative_pv=0.0,
            session_date=None,
        )

    # Handle session reset if timestamps provided
    session_start_idx = 0
    session_date = None

    if timestamps is not None and len(timestamps) > 0:
        # Find session start (last date change)
        if isinstance(timestamps[-1], datetime):
            current_date = timestamps[-1].date()
            session_date = current_date
            for i in range(len(timestamps) - 1, -1, -1):
                if isinstance(timestamps[i], datetime):
                    if timestamps[i].date() != current_date:
                        session_start_idx = i + 1
                        break

    # Use only data from session start
    session_prices = prices[session_start_idx:]
    session_volumes = volumes[session_start_idx:]

    # Calculate cumulative values
    cumulative_pv = np.cumsum(session_prices * session_volumes)
    cumulative_volume = np.cumsum(session_volumes)

    # Avoid division by zero
    if cumulative_volume[-1] <= 0:
        mean_price = float(np.mean(session_prices))
        return VWAPResult(
            vwap=mean_price,
            upper_band=mean_price,
            lower_band=mean_price,
            std_dev=0.0,
            cumulative_volume=0.0,
            cumulative_pv=0.0,
            session_date=session_date,
        )

    # Calculate VWAP
    vwap = cumulative_pv[-1] / cumulative_volume[-1]

    # Calculate standard deviation from VWAP
    # Weighted deviation: sqrt(sum(volume * (price - vwap)^2) / sum(volume))
    squared_deviations = (session_prices - vwap) ** 2
    weighted_var = np.sum(session_volumes * squared_deviations) / cumulative_volume[-1]
    std_dev = np.sqrt(weighted_var)

    # Calculate bands
    upper_band = vwap + (std_dev_bands * std_dev)
    lower_band = vwap - (std_dev_bands * std_dev)

    return VWAPResult(
        vwap=float(vwap),
        upper_band=float(upper_band),
        lower_band=float(lower_band),
        std_dev=float(std_dev),
        cumulative_volume=float(cumulative_volume[-1]),
        cumulative_pv=float(cumulative_pv[-1]),
        session_date=session_date,
    )


# =============================================================================
# VOLUME PROFILE
# =============================================================================


def calculate_volume_profile(
    prices: np.ndarray,
    volumes: np.ndarray,
    n_bins: int = 20
) -> VolumeProfileResult:
    """
    Calculate Volume Profile - distribution of volume across price levels.

    Volume Profile shows where the most trading activity occurred,
    helping identify support/resistance levels and fair value areas.

    Args:
        prices: Array of prices
        volumes: Array of volumes
        n_bins: Number of price bins/levels

    Returns:
        VolumeProfileResult with POC and Value Area
    """
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have same length")

    if len(prices) == 0 or n_bins < 1:
        return VolumeProfileResult(
            price_levels=np.array([]),
            volume_at_price=np.array([]),
            poc_price=0.0,
            poc_volume=0.0,
            value_area_high=0.0,
            value_area_low=0.0,
            value_area_volume_pct=0.0,
            total_volume=0.0,
            n_bins=0,
        )

    # Create price bins
    price_min = np.min(prices)
    price_max = np.max(prices)

    # Handle case where all prices are the same
    if price_max == price_min:
        total_vol = float(np.sum(volumes))
        return VolumeProfileResult(
            price_levels=np.array([price_min]),
            volume_at_price=np.array([total_vol]),
            poc_price=float(price_min),
            poc_volume=total_vol,
            value_area_high=float(price_max),
            value_area_low=float(price_min),
            value_area_volume_pct=100.0,
            total_volume=total_vol,
            n_bins=1,
        )

    # Create bins
    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Assign each price to a bin and accumulate volume
    volume_at_price = np.zeros(n_bins)
    bin_indices = np.digitize(prices, bin_edges[1:-1])  # 0 to n_bins-1

    for i, vol in enumerate(volumes):
        bin_idx = min(bin_indices[i], n_bins - 1)
        volume_at_price[bin_idx] += vol

    # Find POC (Point of Control) - bin with highest volume
    poc_idx = np.argmax(volume_at_price)
    poc_price = float(bin_centers[poc_idx])
    poc_volume = float(volume_at_price[poc_idx])

    # Calculate total volume
    total_volume = float(np.sum(volume_at_price))

    # Calculate Value Area (default 70% of volume)
    value_area_high, value_area_low = get_value_area(
        VolumeProfileResult(
            price_levels=bin_centers,
            volume_at_price=volume_at_price,
            poc_price=poc_price,
            poc_volume=poc_volume,
            value_area_high=0.0,
            value_area_low=0.0,
            value_area_volume_pct=0.0,
            total_volume=total_volume,
            n_bins=n_bins,
        ),
        pct=0.70
    )

    # Calculate actual percentage in value area
    va_volume = 0.0
    for i, (center, vol) in enumerate(zip(bin_centers, volume_at_price)):
        if value_area_low <= center <= value_area_high:
            va_volume += vol
    va_volume_pct = (va_volume / total_volume * 100) if total_volume > 0 else 0.0

    return VolumeProfileResult(
        price_levels=bin_centers,
        volume_at_price=volume_at_price,
        poc_price=poc_price,
        poc_volume=poc_volume,
        value_area_high=float(value_area_high),
        value_area_low=float(value_area_low),
        value_area_volume_pct=float(va_volume_pct),
        total_volume=total_volume,
        n_bins=n_bins,
    )


def get_poc(volume_profile: VolumeProfileResult) -> tuple[float, float]:
    """
    Get Point of Control (POC) from volume profile.

    POC is the price level with the highest traded volume.
    It often acts as support/resistance.

    Args:
        volume_profile: VolumeProfileResult from calculate_volume_profile

    Returns:
        Tuple of (poc_price, poc_volume)
    """
    return volume_profile.poc_price, volume_profile.poc_volume


def get_value_area(
    volume_profile: VolumeProfileResult,
    pct: float = 0.70
) -> tuple[float, float]:
    """
    Calculate Value Area High and Low from volume profile.

    Value Area contains the specified percentage of volume, centered
    around the POC. It represents the "fair value" range.

    Args:
        volume_profile: VolumeProfileResult from calculate_volume_profile
        pct: Percentage of volume to include (default 70%)

    Returns:
        Tuple of (value_area_high, value_area_low)
    """
    if len(volume_profile.price_levels) == 0:
        return 0.0, 0.0

    if volume_profile.total_volume <= 0:
        return (
            float(np.max(volume_profile.price_levels)),
            float(np.min(volume_profile.price_levels))
        )

    # Start from POC and expand outward
    poc_idx = np.argmax(volume_profile.volume_at_price)

    target_volume = volume_profile.total_volume * pct
    current_volume = volume_profile.volume_at_price[poc_idx]

    low_idx = poc_idx
    high_idx = poc_idx

    # Expand alternating up and down until target volume reached
    while current_volume < target_volume:
        # Check if we can expand
        can_go_lower = low_idx > 0
        can_go_higher = high_idx < len(volume_profile.volume_at_price) - 1

        if not can_go_lower and not can_go_higher:
            break

        # Compare which side has more volume
        lower_vol = (
            volume_profile.volume_at_price[low_idx - 1]
            if can_go_lower else 0
        )
        higher_vol = (
            volume_profile.volume_at_price[high_idx + 1]
            if can_go_higher else 0
        )

        if lower_vol >= higher_vol and can_go_lower:
            low_idx -= 1
            current_volume += volume_profile.volume_at_price[low_idx]
        elif can_go_higher:
            high_idx += 1
            current_volume += volume_profile.volume_at_price[high_idx]
        elif can_go_lower:
            low_idx -= 1
            current_volume += volume_profile.volume_at_price[low_idx]

    # Get price levels from indices
    # Account for bin width to get actual range boundaries
    price_levels = volume_profile.price_levels
    n = len(price_levels)

    if n > 1:
        bin_width = (price_levels[1] - price_levels[0]) / 2
    else:
        bin_width = 0

    value_area_low = float(price_levels[low_idx] - bin_width)
    value_area_high = float(price_levels[high_idx] + bin_width)

    return value_area_high, value_area_low


# =============================================================================
# OBV - On-Balance Volume
# =============================================================================


def calculate_obv(
    prices: np.ndarray,
    volumes: np.ndarray
) -> np.ndarray:
    """
    Calculate On-Balance Volume (OBV).

    OBV is a cumulative indicator that adds volume on up days
    and subtracts volume on down days. It helps identify
    accumulation/distribution.

    Formula:
    - If close > previous close: OBV = previous OBV + volume
    - If close < previous close: OBV = previous OBV - volume
    - If close = previous close: OBV = previous OBV

    Args:
        prices: Array of closing prices
        volumes: Array of volumes

    Returns:
        Array of OBV values
    """
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have same length")

    if len(prices) == 0:
        return np.array([])

    if len(prices) == 1:
        return np.array([0.0])

    obv = np.zeros(len(prices))
    obv[0] = 0  # Start at zero

    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif prices[i] < prices[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]

    return obv


def calculate_obv_ma(
    prices: np.ndarray,
    volumes: np.ndarray,
    period: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate OBV with moving average signal line.

    Args:
        prices: Array of closing prices
        volumes: Array of volumes
        period: Period for OBV moving average

    Returns:
        Tuple of (obv_values, obv_ma_values)
    """
    obv = calculate_obv(prices, volumes)

    # Calculate moving average of OBV
    obv_ma = np.full(len(obv), np.nan)

    for i in range(period - 1, len(obv)):
        obv_ma[i] = np.mean(obv[i - period + 1:i + 1])

    return obv, obv_ma


# =============================================================================
# VOLUME RSI
# =============================================================================


def calculate_volume_rsi(
    volumes: np.ndarray,
    period: int = 14
) -> float:
    """
    Calculate Volume RSI (RSI applied to volume changes).

    Volume RSI measures the strength of volume changes, helping
    identify volume exhaustion or continuation.

    Formula: Same as price RSI but applied to volume changes

    Args:
        volumes: Array of volumes
        period: RSI period (default 14)

    Returns:
        Volume RSI value (0-100)
    """
    if len(volumes) < period + 1:
        return 50.0  # Neutral default

    # Calculate volume changes
    volume_changes = np.diff(volumes)

    # Separate gains and losses
    gains = np.where(volume_changes > 0, volume_changes, 0)
    losses = np.where(volume_changes < 0, -volume_changes, 0)

    if len(gains) < period:
        return 50.0

    # Use Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / period

    # Initialize with simple average of first 'period' values
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Apply Wilder's smoothing for remaining values
    for i in range(period, len(gains)):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

    # Calculate RSI
    if avg_loss < 1e-10:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi)


def calculate_volume_rsi_series(
    volumes: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    Calculate Volume RSI series.

    Args:
        volumes: Array of volumes
        period: RSI period

    Returns:
        Array of Volume RSI values
    """
    if len(volumes) < period + 2:
        return np.full(len(volumes), 50.0)

    rsi_series = np.full(len(volumes), np.nan)

    for i in range(period + 1, len(volumes) + 1):
        rsi_series[i - 1] = calculate_volume_rsi(volumes[:i], period)

    # Fill initial values with first calculated RSI
    first_valid = period + 1
    if first_valid < len(rsi_series):
        rsi_series[:first_valid] = rsi_series[first_valid]

    return rsi_series


# =============================================================================
# VOLUME CLIMAX DETECTION
# =============================================================================


def detect_volume_climax(
    prices: np.ndarray,
    volumes: np.ndarray,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    volume_threshold: float = 2.0,
    lookback: int = 20
) -> VolumeClimaxResult:
    """
    Detect volume climax conditions (buying or selling exhaustion).

    A climax occurs when:
    - Volume is significantly above average (threshold multiple)
    - Price range is expanded
    - Close is at extreme of range (suggests exhaustion)

    Buying Climax: High volume, wide range, close near high (bearish)
    Selling Climax: High volume, wide range, close near low (bullish)

    Args:
        prices: Array of closing prices
        volumes: Array of volumes
        highs: Optional array of high prices (estimated if not provided)
        lows: Optional array of low prices (estimated if not provided)
        volume_threshold: Multiple of average volume to qualify (default 2.0)
        lookback: Period for calculating averages (default 20)

    Returns:
        VolumeClimaxResult with climax analysis
    """
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have same length")

    if len(prices) < lookback + 1:
        return VolumeClimaxResult(
            is_climax=False,
            climax_type=None,
            volume_ratio=1.0,
            price_range_ratio=1.0,
            close_position=0.5,
            strength=0.0,
        )

    # Estimate highs/lows if not provided
    if highs is None or lows is None:
        # Use rolling max/min as proxy
        highs = np.zeros(len(prices))
        lows = np.zeros(len(prices))

        for i in range(len(prices)):
            if i == 0:
                highs[i] = prices[i] * 1.001
                lows[i] = prices[i] * 0.999
            else:
                # Estimate based on price change
                change = abs(prices[i] - prices[i - 1])
                highs[i] = max(prices[i], prices[i - 1]) + change * 0.1
                lows[i] = min(prices[i], prices[i - 1]) - change * 0.1

    # Current bar values
    current_vol = volumes[-1]
    current_high = highs[-1]
    current_low = lows[-1]
    current_close = prices[-1]
    current_range = current_high - current_low

    # Calculate averages (excluding current bar)
    avg_volume = np.mean(volumes[-lookback - 1:-1])

    # Calculate average true range
    ranges = highs[-lookback - 1:-1] - lows[-lookback - 1:-1]
    avg_range = np.mean(ranges) if len(ranges) > 0 else current_range

    # Calculate ratios
    volume_ratio = current_vol / avg_volume if avg_volume > 0 else 1.0
    range_ratio = current_range / avg_range if avg_range > 0 else 1.0

    # Calculate close position within range (0 = low, 1 = high)
    if current_range > 0:
        close_position = (current_close - current_low) / current_range
    else:
        close_position = 0.5

    # Determine if climax
    is_climax = False
    climax_type = None
    strength = 0.0

    if volume_ratio >= volume_threshold and range_ratio >= 1.2:
        # High volume and expanded range - potential climax

        if close_position >= 0.8:
            # Buying climax: close near high with exhaustion
            is_climax = True
            climax_type = "buying_climax"
            # Strength based on how extreme the conditions are
            strength = min(1.0, (volume_ratio / volume_threshold) *
                          close_position * 0.5)

        elif close_position <= 0.2:
            # Selling climax: close near low with exhaustion
            is_climax = True
            climax_type = "selling_climax"
            strength = min(1.0, (volume_ratio / volume_threshold) *
                          (1 - close_position) * 0.5)

    return VolumeClimaxResult(
        is_climax=is_climax,
        climax_type=climax_type,
        volume_ratio=float(volume_ratio),
        price_range_ratio=float(range_ratio),
        close_position=float(close_position),
        strength=float(strength),
    )


# =============================================================================
# ADDITIONAL VOLUME ANALYSIS
# =============================================================================


def calculate_volume_price_trend(
    prices: np.ndarray,
    volumes: np.ndarray
) -> np.ndarray:
    """
    Calculate Volume Price Trend (VPT).

    VPT is similar to OBV but incorporates the percentage change
    in price, making it more sensitive to proportional moves.

    Formula: VPT = VPT_prev + volume * ((close - close_prev) / close_prev)

    Args:
        prices: Array of closing prices
        volumes: Array of volumes

    Returns:
        Array of VPT values
    """
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have same length")

    if len(prices) < 2:
        return np.zeros(len(prices))

    vpt = np.zeros(len(prices))
    vpt[0] = 0

    for i in range(1, len(prices)):
        if prices[i - 1] != 0:
            price_change_pct = (prices[i] - prices[i - 1]) / prices[i - 1]
            vpt[i] = vpt[i - 1] + volumes[i] * price_change_pct
        else:
            vpt[i] = vpt[i - 1]

    return vpt


def calculate_force_index(
    prices: np.ndarray,
    volumes: np.ndarray,
    period: int = 13
) -> np.ndarray:
    """
    Calculate Force Index (Elder's Force Index).

    Force Index measures the force behind price movements using
    both price change and volume.

    Formula:
    - Raw Force = (close - close_prev) * volume
    - Force Index = EMA(Raw Force, period)

    Args:
        prices: Array of closing prices
        volumes: Array of volumes
        period: EMA period for smoothing (default 13)

    Returns:
        Array of Force Index values
    """
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have same length")

    if len(prices) < 2:
        return np.zeros(len(prices))

    # Calculate raw force
    price_changes = np.diff(prices)
    raw_force = np.zeros(len(prices))
    raw_force[1:] = price_changes * volumes[1:]

    # Apply EMA smoothing
    if period < 1:
        return raw_force

    alpha = 2 / (period + 1)
    force_index = np.zeros(len(prices))
    force_index[0] = raw_force[0]

    for i in range(1, len(prices)):
        force_index[i] = alpha * raw_force[i] + (1 - alpha) * force_index[i - 1]

    return force_index


def analyze_volume_trend(
    volumes: np.ndarray,
    period: int = 20
) -> dict[str, Any]:
    """
    Analyze volume trend direction and strength.

    Args:
        volumes: Array of volumes
        period: Analysis period

    Returns:
        Dictionary with volume trend analysis
    """
    if len(volumes) < period:
        return {
            "trend": "unknown",
            "strength": 0.0,
            "avg_volume": 0.0,
            "current_vs_avg": 1.0,
            "is_increasing": False,
        }

    recent_volumes = volumes[-period:]
    avg_volume = float(np.mean(recent_volumes))

    # Calculate trend using linear regression slope
    x = np.arange(len(recent_volumes))
    if len(x) > 1:
        slope = np.polyfit(x, recent_volumes, 1)[0]
        slope_pct = slope / avg_volume if avg_volume > 0 else 0
    else:
        slope_pct = 0

    # Determine trend
    if slope_pct > 0.02:  # 2% increase per period
        trend = "increasing"
        is_increasing = True
    elif slope_pct < -0.02:
        trend = "decreasing"
        is_increasing = False
    else:
        trend = "flat"
        is_increasing = False

    # Calculate strength
    strength = min(1.0, abs(slope_pct) * 10)  # Normalize to 0-1

    # Current vs average
    current_vs_avg = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

    return {
        "trend": trend,
        "strength": float(strength),
        "avg_volume": avg_volume,
        "current_vs_avg": float(current_vs_avg),
        "is_increasing": is_increasing,
        "slope_pct": float(slope_pct),
    }

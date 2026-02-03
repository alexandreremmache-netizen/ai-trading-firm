"""
Demand/Supply Zone Detection
============================

Identifies key price levels where significant buying or selling pressure exists.
Based on price action and volume analysis.

This module is used by signal agents to identify support/resistance levels
and potential reversal zones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PriceZone:
    """Represents a demand or supply zone."""
    zone_type: str  # "demand" or "supply"
    price_low: float
    price_high: float
    strength: float  # 0.0 to 1.0 (based on touches and volume)
    first_touch: datetime
    last_touch: datetime
    touch_count: int = 1
    volume_at_zone: float = 0.0
    broken: bool = False
    broken_at: datetime | None = None

    @property
    def mid_price(self) -> float:
        """Get the middle of the zone."""
        return (self.price_low + self.price_high) / 2

    @property
    def zone_width(self) -> float:
        """Get the width of the zone."""
        return self.price_high - self.price_low

    @property
    def age_hours(self) -> float:
        """Get the age of the zone in hours."""
        delta = datetime.now(timezone.utc) - self.first_touch
        return delta.total_seconds() / 3600

    def contains_price(self, price: float, buffer_pct: float = 0.1) -> bool:
        """Check if price is within the zone (with optional buffer)."""
        buffer = self.zone_width * buffer_pct
        return (self.price_low - buffer) <= price <= (self.price_high + buffer)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "zone_type": self.zone_type,
            "price_low": self.price_low,
            "price_high": self.price_high,
            "mid_price": self.mid_price,
            "strength": self.strength,
            "touch_count": self.touch_count,
            "volume_at_zone": self.volume_at_zone,
            "age_hours": self.age_hours,
            "broken": self.broken,
        }


@dataclass
class CandleStick:
    """Simple candlestick representation."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

    @property
    def body_size(self) -> float:
        """Size of the candle body."""
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        """Size of upper wick."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Size of lower wick."""
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish."""
        return self.close < self.open

    @property
    def total_range(self) -> float:
        """Total high-low range."""
        return self.high - self.low


class DemandZoneDetector:
    """
    Detects demand (support) and supply (resistance) zones.

    Methodology:
    1. Identify swing highs and swing lows
    2. Look for strong moves away from price levels (impulse moves)
    3. Mark the base of the impulse as a zone
    4. Track zone touches and breaks

    Usage:
        detector = DemandZoneDetector()
        detector.add_candle(candle)
        zones = detector.get_active_zones(current_price)
    """

    def __init__(
        self,
        lookback: int = 50,
        zone_touch_threshold: float = 0.02,  # 2% zone width
        min_impulse_size: float = 0.01,  # 1% minimum impulse move
        min_zone_strength: float = 0.3,
        max_zones: int = 10,
    ):
        self._lookback = lookback
        self._zone_touch_threshold = zone_touch_threshold
        self._min_impulse_size = min_impulse_size
        self._min_zone_strength = min_zone_strength
        self._max_zones = max_zones

        self._candles: deque[CandleStick] = deque(maxlen=lookback * 2)
        self._demand_zones: list[PriceZone] = []
        self._supply_zones: list[PriceZone] = []

    def add_candle(self, candle: CandleStick) -> None:
        """Add a new candle and update zones."""
        self._candles.append(candle)

        if len(self._candles) < 5:
            return

        # Detect new zones
        self._detect_zones()

        # Update existing zones (check for touches and breaks)
        self._update_zones(candle)

        # Cleanup old/broken zones
        self._cleanup_zones()

    def add_price_data(
        self,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: int = 0,
    ) -> None:
        """Add price data as individual values."""
        candle = CandleStick(
            timestamp=timestamp,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        self.add_candle(candle)

    def _detect_zones(self) -> None:
        """Detect new demand and supply zones from recent price action."""
        if len(self._candles) < 10:
            return

        candles = list(self._candles)

        # Look for demand zones (strong bullish moves from a base)
        self._detect_demand_zones(candles)

        # Look for supply zones (strong bearish moves from a base)
        self._detect_supply_zones(candles)

    def _detect_demand_zones(self, candles: list[CandleStick]) -> None:
        """
        Detect demand zones (buying pressure areas).

        Pattern: Consolidation/base followed by strong bullish impulse.
        The base becomes the demand zone.
        """
        n = len(candles)

        for i in range(5, n - 2):
            # Check for bullish impulse (strong up move)
            impulse_candle = candles[i]

            # Need a significant bullish candle
            if not impulse_candle.is_bullish:
                continue

            if impulse_candle.body_size < impulse_candle.total_range * 0.5:
                continue

            # Check impulse size relative to recent range
            recent_range = max(c.high for c in candles[i-5:i]) - min(c.low for c in candles[i-5:i])
            if recent_range <= 0:
                continue

            impulse_pct = impulse_candle.body_size / recent_range
            if impulse_pct < self._min_impulse_size * 10:  # Require meaningful impulse
                continue

            # The base before the impulse is the demand zone
            base_candles = candles[max(0, i-3):i]
            if not base_candles:
                continue

            zone_low = min(c.low for c in base_candles)
            zone_high = min(c.high for c in base_candles)  # Use lower highs for tighter zone

            # Check if zone already exists
            if self._zone_exists(zone_low, zone_high, "demand"):
                continue

            # Calculate strength based on impulse size and volume
            volume_factor = impulse_candle.volume / max(1, sum(c.volume for c in base_candles) / len(base_candles))
            strength = min(1.0, impulse_pct * 0.5 + min(1.0, volume_factor) * 0.5)

            if strength < self._min_zone_strength:
                continue

            zone = PriceZone(
                zone_type="demand",
                price_low=zone_low,
                price_high=zone_high,
                strength=strength,
                first_touch=base_candles[0].timestamp,
                last_touch=base_candles[-1].timestamp,
                volume_at_zone=sum(c.volume for c in base_candles),
            )

            self._demand_zones.append(zone)
            logger.debug(f"Detected demand zone: {zone_low:.2f}-{zone_high:.2f} (strength={strength:.2f})")

    def _detect_supply_zones(self, candles: list[CandleStick]) -> None:
        """
        Detect supply zones (selling pressure areas).

        Pattern: Consolidation/base followed by strong bearish impulse.
        The base becomes the supply zone.
        """
        n = len(candles)

        for i in range(5, n - 2):
            # Check for bearish impulse (strong down move)
            impulse_candle = candles[i]

            # Need a significant bearish candle
            if not impulse_candle.is_bearish:
                continue

            if impulse_candle.body_size < impulse_candle.total_range * 0.5:
                continue

            # Check impulse size relative to recent range
            recent_range = max(c.high for c in candles[i-5:i]) - min(c.low for c in candles[i-5:i])
            if recent_range <= 0:
                continue

            impulse_pct = impulse_candle.body_size / recent_range
            if impulse_pct < self._min_impulse_size * 10:
                continue

            # The base before the impulse is the supply zone
            base_candles = candles[max(0, i-3):i]
            if not base_candles:
                continue

            zone_low = max(c.low for c in base_candles)  # Use higher lows for tighter zone
            zone_high = max(c.high for c in base_candles)

            # Check if zone already exists
            if self._zone_exists(zone_low, zone_high, "supply"):
                continue

            # Calculate strength
            volume_factor = impulse_candle.volume / max(1, sum(c.volume for c in base_candles) / len(base_candles))
            strength = min(1.0, impulse_pct * 0.5 + min(1.0, volume_factor) * 0.5)

            if strength < self._min_zone_strength:
                continue

            zone = PriceZone(
                zone_type="supply",
                price_low=zone_low,
                price_high=zone_high,
                strength=strength,
                first_touch=base_candles[0].timestamp,
                last_touch=base_candles[-1].timestamp,
                volume_at_zone=sum(c.volume for c in base_candles),
            )

            self._supply_zones.append(zone)
            logger.debug(f"Detected supply zone: {zone_low:.2f}-{zone_high:.2f} (strength={strength:.2f})")

    def _zone_exists(self, low: float, high: float, zone_type: str) -> bool:
        """Check if a similar zone already exists."""
        zones = self._demand_zones if zone_type == "demand" else self._supply_zones
        threshold = (high - low) * 0.5  # 50% overlap

        for zone in zones:
            if zone.broken:
                continue
            # Check for significant overlap
            overlap_low = max(low, zone.price_low)
            overlap_high = min(high, zone.price_high)
            if overlap_high > overlap_low:
                overlap = overlap_high - overlap_low
                if overlap > threshold:
                    return True

        return False

    def _update_zones(self, candle: CandleStick) -> None:
        """Update zones based on new price action."""
        current_price = candle.close

        # Update demand zones
        for zone in self._demand_zones:
            if zone.broken:
                continue

            # Check for touch (price enters zone)
            if zone.contains_price(candle.low):
                zone.touch_count += 1
                zone.last_touch = candle.timestamp
                zone.volume_at_zone += candle.volume
                # Increase strength on successful holds
                zone.strength = min(1.0, zone.strength + 0.05)

            # Check for break (price closes below zone)
            if candle.close < zone.price_low:
                zone.broken = True
                zone.broken_at = candle.timestamp
                logger.debug(f"Demand zone broken: {zone.price_low:.2f}-{zone.price_high:.2f}")

        # Update supply zones
        for zone in self._supply_zones:
            if zone.broken:
                continue

            # Check for touch
            if zone.contains_price(candle.high):
                zone.touch_count += 1
                zone.last_touch = candle.timestamp
                zone.volume_at_zone += candle.volume
                zone.strength = min(1.0, zone.strength + 0.05)

            # Check for break (price closes above zone)
            if candle.close > zone.price_high:
                zone.broken = True
                zone.broken_at = candle.timestamp
                logger.debug(f"Supply zone broken: {zone.price_low:.2f}-{zone.price_high:.2f}")

    def _cleanup_zones(self) -> None:
        """Remove old or weak zones."""
        # Remove broken zones older than 24 hours
        cutoff = datetime.now(timezone.utc)

        self._demand_zones = [
            z for z in self._demand_zones
            if not z.broken or (z.broken_at and (cutoff - z.broken_at).total_seconds() < 86400)
        ]

        self._supply_zones = [
            z for z in self._supply_zones
            if not z.broken or (z.broken_at and (cutoff - z.broken_at).total_seconds() < 86400)
        ]

        # Keep only top zones by strength
        self._demand_zones = sorted(
            self._demand_zones, key=lambda z: z.strength, reverse=True
        )[:self._max_zones]

        self._supply_zones = sorted(
            self._supply_zones, key=lambda z: z.strength, reverse=True
        )[:self._max_zones]

    def get_active_zones(
        self,
        current_price: float | None = None,
        include_broken: bool = False,
    ) -> dict[str, list[PriceZone]]:
        """
        Get active demand and supply zones.

        Args:
            current_price: Optional price to filter nearby zones
            include_broken: Whether to include broken zones

        Returns:
            Dictionary with "demand" and "supply" zone lists
        """
        demand = [z for z in self._demand_zones if include_broken or not z.broken]
        supply = [z for z in self._supply_zones if include_broken or not z.broken]

        # Sort by proximity to current price if provided
        if current_price is not None:
            demand = sorted(demand, key=lambda z: abs(z.mid_price - current_price))
            supply = sorted(supply, key=lambda z: abs(z.mid_price - current_price))

        return {
            "demand": demand,
            "supply": supply,
        }

    def get_nearest_zone(
        self,
        current_price: float,
        zone_type: str = "any",
    ) -> PriceZone | None:
        """
        Get the nearest active zone to the current price.

        Args:
            current_price: Current price
            zone_type: "demand", "supply", or "any"

        Returns:
            Nearest zone or None
        """
        zones = self.get_active_zones(current_price)

        candidates = []
        if zone_type in ("demand", "any"):
            candidates.extend(zones["demand"])
        if zone_type in ("supply", "any"):
            candidates.extend(zones["supply"])

        if not candidates:
            return None

        return min(candidates, key=lambda z: abs(z.mid_price - current_price))

    def is_at_demand_zone(self, price: float, buffer_pct: float = 0.5) -> tuple[bool, PriceZone | None]:
        """Check if price is at a demand zone."""
        for zone in self._demand_zones:
            if zone.broken:
                continue
            if zone.contains_price(price, buffer_pct):
                return True, zone
        return False, None

    def is_at_supply_zone(self, price: float, buffer_pct: float = 0.5) -> tuple[bool, PriceZone | None]:
        """Check if price is at a supply zone."""
        for zone in self._supply_zones:
            if zone.broken:
                continue
            if zone.contains_price(price, buffer_pct):
                return True, zone
        return False, None

    def get_zone_analysis(self, current_price: float) -> dict[str, Any]:
        """
        Get comprehensive zone analysis for current price.

        Returns analysis including:
        - Nearest zones
        - Zone bias (more demand below = bullish, more supply above = bearish)
        - Suggested action
        """
        zones = self.get_active_zones(current_price)

        # Count zones above and below price
        demand_below = [z for z in zones["demand"] if z.mid_price < current_price]
        supply_above = [z for z in zones["supply"] if z.mid_price > current_price]
        demand_above = [z for z in zones["demand"] if z.mid_price >= current_price]
        supply_below = [z for z in zones["supply"] if z.mid_price <= current_price]

        # Calculate weighted support/resistance strength
        support_strength = sum(z.strength for z in demand_below)
        resistance_strength = sum(z.strength for z in supply_above)

        # Determine bias
        if support_strength > resistance_strength * 1.5:
            bias = "bullish"
        elif resistance_strength > support_strength * 1.5:
            bias = "bearish"
        else:
            bias = "neutral"

        # Check if at a zone
        at_demand, demand_zone = self.is_at_demand_zone(current_price)
        at_supply, supply_zone = self.is_at_supply_zone(current_price)

        return {
            "current_price": current_price,
            "bias": bias,
            "support_strength": support_strength,
            "resistance_strength": resistance_strength,
            "demand_zones_below": len(demand_below),
            "supply_zones_above": len(supply_above),
            "at_demand_zone": at_demand,
            "at_supply_zone": at_supply,
            "nearest_demand": demand_below[0].to_dict() if demand_below else None,
            "nearest_supply": supply_above[0].to_dict() if supply_above else None,
            "demand_zone_detail": demand_zone.to_dict() if demand_zone else None,
            "supply_zone_detail": supply_zone.to_dict() if supply_zone else None,
        }

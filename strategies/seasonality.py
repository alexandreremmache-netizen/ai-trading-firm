"""
Seasonality Strategy
====================

Implements seasonal patterns for commodity and equity trading.
Based on historical calendar effects and production/consumption cycles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class SeasonalPattern(Enum):
    """Types of seasonal patterns."""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    EVENT_BASED = "event_based"


class SeasonalStrength(Enum):
    """Strength of seasonal signal."""
    STRONG = "strong"  # Historical win rate > 70%
    MODERATE = "moderate"  # 60-70%
    WEAK = "weak"  # 50-60%
    NEUTRAL = "neutral"  # < 50%


@dataclass
class SeasonalWindow:
    """Defines a seasonal trading window."""
    name: str
    symbol: str
    pattern_type: SeasonalPattern
    start_month: int
    start_day: int
    end_month: int
    end_day: int
    direction: str  # "long" or "short"
    historical_win_rate: float
    avg_return: float
    description: str
    tags: list[str] = field(default_factory=list)

    @property
    def strength(self) -> SeasonalStrength:
        """Determine signal strength from win rate."""
        if self.historical_win_rate >= 0.70:
            return SeasonalStrength.STRONG
        elif self.historical_win_rate >= 0.60:
            return SeasonalStrength.MODERATE
        elif self.historical_win_rate >= 0.50:
            return SeasonalStrength.WEAK
        else:
            return SeasonalStrength.NEUTRAL

    def is_active(self, check_date: date) -> bool:
        """Check if this seasonal window is active on given date."""
        month = check_date.month
        day = check_date.day

        # Handle year wrap (e.g., Nov-Feb pattern)
        if self.start_month <= self.end_month:
            # Normal case: within same year
            if month < self.start_month or month > self.end_month:
                return False
            if month == self.start_month and day < self.start_day:
                return False
            if month == self.end_month and day > self.end_day:
                return False
            return True
        else:
            # Wrapping case: crosses year boundary
            if month >= self.start_month:
                if month == self.start_month and day < self.start_day:
                    return False
                return True
            elif month <= self.end_month:
                if month == self.end_month and day > self.end_day:
                    return False
                return True
            return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "pattern_type": self.pattern_type.value,
            "start": f"{self.start_month:02d}-{self.start_day:02d}",
            "end": f"{self.end_month:02d}-{self.end_day:02d}",
            "direction": self.direction,
            "historical_win_rate": self.historical_win_rate,
            "avg_return": self.avg_return,
            "strength": self.strength.value,
            "description": self.description,
        }


# =============================================================================
# PREDEFINED SEASONAL PATTERNS
# =============================================================================

SEASONAL_PATTERNS: dict[str, list[SeasonalWindow]] = {
    # =========================================================================
    # NATURAL GAS (NG) - Strong seasonal patterns
    # =========================================================================
    "NG": [
        SeasonalWindow(
            name="NG Winter Demand",
            symbol="NG",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=10,  # October
            start_day=1,
            end_month=2,  # February
            end_day=28,
            direction="long",
            historical_win_rate=0.68,
            avg_return=0.12,
            description="Long NG ahead of winter heating demand peak",
            tags=["heating", "winter", "demand"],
        ),
        SeasonalWindow(
            name="NG Summer Weakness",
            symbol="NG",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=4,  # April
            start_day=1,
            end_month=6,  # June
            end_day=30,
            direction="short",
            historical_win_rate=0.62,
            avg_return=0.08,
            description="Short NG during spring shoulder season, post-winter inventory builds",
            tags=["storage", "shoulder", "supply"],
        ),
        SeasonalWindow(
            name="NG Injection Season",
            symbol="NG",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=3,  # March
            start_day=15,
            end_month=10,  # October
            end_day=31,
            direction="short",
            historical_win_rate=0.58,
            avg_return=0.06,
            description="Short bias during storage injection season",
            tags=["storage", "injection"],
        ),
    ],

    # =========================================================================
    # CORN (ZC) - Agricultural seasonal patterns
    # =========================================================================
    "ZC": [
        SeasonalWindow(
            name="ZC Planting Concerns",
            symbol="ZC",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=4,  # April
            start_day=1,
            end_month=5,  # May
            end_day=31,
            direction="long",
            historical_win_rate=0.65,
            avg_return=0.07,
            description="Long corn during planting season uncertainty",
            tags=["planting", "weather", "uncertainty"],
        ),
        SeasonalWindow(
            name="ZC Harvest Pressure",
            symbol="ZC",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=9,  # September
            start_day=15,
            end_month=11,  # November
            end_day=15,
            direction="short",
            historical_win_rate=0.63,
            avg_return=0.06,
            description="Short corn during harvest supply pressure",
            tags=["harvest", "supply", "pressure"],
        ),
        SeasonalWindow(
            name="ZC Pre-Report Rally",
            symbol="ZC",
            pattern_type=SeasonalPattern.MONTHLY,
            start_month=6,  # June (USDA reports)
            start_day=1,
            end_month=6,
            end_day=15,
            direction="long",
            historical_win_rate=0.58,
            avg_return=0.04,
            description="Long corn ahead of June USDA acreage report",
            tags=["usda", "report", "acreage"],
        ),
    ],

    # =========================================================================
    # WHEAT (ZW) - Similar to corn but different timing
    # =========================================================================
    "ZW": [
        SeasonalWindow(
            name="ZW Winter Wheat Concerns",
            symbol="ZW",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=2,  # February
            start_day=1,
            end_month=4,  # April
            end_day=30,
            direction="long",
            historical_win_rate=0.62,
            avg_return=0.08,
            description="Long wheat on winter kill/crop condition concerns",
            tags=["winter", "crop", "condition"],
        ),
        SeasonalWindow(
            name="ZW Harvest",
            symbol="ZW",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=6,  # June
            start_day=1,
            end_month=7,  # July
            end_day=31,
            direction="short",
            historical_win_rate=0.60,
            avg_return=0.05,
            description="Short wheat during US winter wheat harvest",
            tags=["harvest", "supply"],
        ),
    ],

    # =========================================================================
    # SOYBEANS (ZS) - Export and South America patterns
    # =========================================================================
    "ZS": [
        SeasonalWindow(
            name="ZS South America Weather",
            symbol="ZS",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=12,  # December
            start_day=1,
            end_month=2,  # February
            end_day=28,
            direction="long",
            historical_win_rate=0.64,
            avg_return=0.09,
            description="Long soybeans on South America growing season weather concerns",
            tags=["weather", "brazil", "argentina"],
        ),
        SeasonalWindow(
            name="ZS US Harvest Pressure",
            symbol="ZS",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=9,  # September
            start_day=15,
            end_month=10,  # October
            end_day=31,
            direction="short",
            historical_win_rate=0.61,
            avg_return=0.05,
            description="Short soybeans during US harvest pressure",
            tags=["harvest", "supply", "us"],
        ),
    ],

    # =========================================================================
    # GOLD (GC) - Jewelry demand and safe haven
    # =========================================================================
    "GC": [
        SeasonalWindow(
            name="GC Indian Wedding Season",
            symbol="GC",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=9,  # September
            start_day=1,
            end_month=11,  # November
            end_day=30,
            direction="long",
            historical_win_rate=0.64,
            avg_return=0.05,
            description="Long gold ahead of Indian wedding/festival season demand",
            tags=["jewelry", "india", "demand"],
        ),
        SeasonalWindow(
            name="GC Chinese New Year",
            symbol="GC",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=1,  # January
            start_day=1,
            end_month=2,  # February
            end_day=15,
            direction="long",
            historical_win_rate=0.60,
            avg_return=0.04,
            description="Long gold for Chinese New Year gifting demand",
            tags=["jewelry", "china", "demand"],
        ),
        SeasonalWindow(
            name="GC Summer Doldrums",
            symbol="GC",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=6,  # June
            start_day=15,
            end_month=7,  # July
            end_day=31,
            direction="short",
            historical_win_rate=0.58,
            avg_return=0.03,
            description="Short gold during low-demand summer period",
            tags=["low_demand", "summer"],
        ),
    ],

    # =========================================================================
    # CRUDE OIL (CL) - Driving season and refinery patterns
    # =========================================================================
    "CL": [
        SeasonalWindow(
            name="CL Driving Season Buildup",
            symbol="CL",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=2,  # February
            start_day=1,
            end_month=4,  # April
            end_day=30,
            direction="long",
            historical_win_rate=0.62,
            avg_return=0.08,
            description="Long crude ahead of summer driving season",
            tags=["driving", "gasoline", "demand"],
        ),
        SeasonalWindow(
            name="CL Refinery Turnaround",
            symbol="CL",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=9,  # September
            start_day=15,
            end_month=10,  # October
            end_day=31,
            direction="short",
            historical_win_rate=0.59,
            avg_return=0.04,
            description="Short crude during fall refinery maintenance",
            tags=["refinery", "maintenance", "turnaround"],
        ),
    ],

    # =========================================================================
    # GASOLINE (RB) - Strong driving season pattern
    # =========================================================================
    "RB": [
        SeasonalWindow(
            name="RB Pre-Summer Rally",
            symbol="RB",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=2,  # February
            start_day=1,
            end_month=5,  # May
            end_day=31,
            direction="long",
            historical_win_rate=0.68,
            avg_return=0.12,
            description="Long gasoline ahead of summer driving season",
            tags=["driving", "summer", "demand"],
        ),
        SeasonalWindow(
            name="RB Post-Labor Day",
            symbol="RB",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=9,  # September
            start_day=1,
            end_month=10,  # October
            end_day=31,
            direction="short",
            historical_win_rate=0.65,
            avg_return=0.07,
            description="Short gasoline after summer driving season ends",
            tags=["post_summer", "demand_drop"],
        ),
    ],

    # =========================================================================
    # EQUITY INDEX (ES/SPY) - Calendar effects
    # =========================================================================
    "ES": [
        SeasonalWindow(
            name="ES Santa Rally",
            symbol="ES",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=12,  # December
            start_day=15,
            end_month=1,  # January
            end_day=3,
            direction="long",
            historical_win_rate=0.72,
            avg_return=0.02,
            description="Long equities during Santa Claus rally period",
            tags=["santa", "year_end", "rally"],
        ),
        SeasonalWindow(
            name="ES Sell in May",
            symbol="ES",
            pattern_type=SeasonalPattern.ANNUAL,
            start_month=5,  # May
            start_day=1,
            end_month=10,  # October
            end_day=31,
            direction="short",
            historical_win_rate=0.55,
            avg_return=0.01,
            description="Reduced equity exposure May-October",
            tags=["sell_in_may", "summer_weakness"],
        ),
        SeasonalWindow(
            name="ES September Weakness",
            symbol="ES",
            pattern_type=SeasonalPattern.MONTHLY,
            start_month=9,  # September
            start_day=1,
            end_month=9,
            end_day=30,
            direction="short",
            historical_win_rate=0.58,
            avg_return=0.02,
            description="Short equities during historically weak September",
            tags=["september", "weakness"],
        ),
    ],
}


@dataclass
class SeasonalSignal:
    """Signal generated from seasonal analysis."""
    symbol: str
    timestamp: datetime
    direction: str
    strength: SeasonalStrength
    confidence: float
    active_patterns: list[SeasonalWindow]
    combined_win_rate: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "active_patterns": [p.name for p in self.active_patterns],
            "combined_win_rate": self.combined_win_rate,
            "rationale": self.rationale,
        }


class SeasonalityStrategy:
    """
    Seasonality-based trading strategy.

    Generates signals based on historical seasonal patterns
    in commodities and equity markets.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize seasonality strategy.

        Args:
            config: Configuration with:
                - min_win_rate: Minimum win rate to generate signal (default: 0.55)
                - enabled_symbols: List of symbols to analyze (default: all)
                - min_patterns: Minimum confirming patterns (default: 1)
        """
        self._config = config or {}
        self._min_win_rate = self._config.get("min_win_rate", 0.55)
        self._enabled_symbols = self._config.get("enabled_symbols", list(SEASONAL_PATTERNS.keys()))
        self._min_patterns = self._config.get("min_patterns", 1)

        # Pattern storage
        self._patterns = dict(SEASONAL_PATTERNS)

        # Custom patterns can be added
        custom = self._config.get("custom_patterns", [])
        for pattern in custom:
            symbol = pattern.get("symbol")
            if symbol:
                if symbol not in self._patterns:
                    self._patterns[symbol] = []
                self._patterns[symbol].append(self._dict_to_window(pattern))

        logger.info(
            f"SeasonalityStrategy initialized: "
            f"{len(self._patterns)} symbols, min_win_rate={self._min_win_rate}"
        )

    def _dict_to_window(self, d: dict[str, Any]) -> SeasonalWindow:
        """Convert dictionary to SeasonalWindow."""
        return SeasonalWindow(
            name=d["name"],
            symbol=d["symbol"],
            pattern_type=SeasonalPattern[d.get("pattern_type", "ANNUAL").upper()],
            start_month=d["start_month"],
            start_day=d["start_day"],
            end_month=d["end_month"],
            end_day=d["end_day"],
            direction=d["direction"],
            historical_win_rate=d["historical_win_rate"],
            avg_return=d["avg_return"],
            description=d.get("description", ""),
            tags=d.get("tags", []),
        )

    def get_active_patterns(
        self,
        symbol: str,
        check_date: date | None = None
    ) -> list[SeasonalWindow]:
        """
        Get active seasonal patterns for a symbol.

        Args:
            symbol: Instrument symbol
            check_date: Date to check (default: today)

        Returns:
            List of active SeasonalWindows
        """
        if check_date is None:
            check_date = date.today()

        if symbol not in self._patterns:
            return []

        active = []
        for pattern in self._patterns[symbol]:
            if pattern.is_active(check_date):
                if pattern.historical_win_rate >= self._min_win_rate:
                    active.append(pattern)

        return active

    def generate_signal(
        self,
        symbol: str,
        check_date: date | None = None
    ) -> SeasonalSignal | None:
        """
        Generate seasonal signal for a symbol.

        Args:
            symbol: Instrument symbol
            check_date: Date to check

        Returns:
            SeasonalSignal if patterns are active, None otherwise
        """
        if symbol not in self._enabled_symbols:
            return None

        active = self.get_active_patterns(symbol, check_date)

        if len(active) < self._min_patterns:
            return None

        # Combine signals
        long_patterns = [p for p in active if p.direction == "long"]
        short_patterns = [p for p in active if p.direction == "short"]

        # Determine direction based on pattern strength
        long_score = sum(p.historical_win_rate * p.avg_return for p in long_patterns)
        short_score = sum(p.historical_win_rate * p.avg_return for p in short_patterns)

        if long_score > short_score and long_patterns:
            direction = "long"
            patterns = long_patterns
        elif short_patterns:
            direction = "short"
            patterns = short_patterns
        else:
            direction = "long"
            patterns = long_patterns

        if not patterns:
            return None

        # Calculate combined metrics
        combined_win_rate = np.mean([p.historical_win_rate for p in patterns])
        avg_strength = np.mean([p.avg_return for p in patterns])

        # Determine strength
        if combined_win_rate >= 0.70:
            strength = SeasonalStrength.STRONG
        elif combined_win_rate >= 0.60:
            strength = SeasonalStrength.MODERATE
        else:
            strength = SeasonalStrength.WEAK

        # Confidence based on number of confirming patterns
        confidence = min(0.9, 0.5 + 0.1 * len(patterns))

        # Build rationale
        pattern_names = ", ".join(p.name for p in patterns)
        rationale = f"Active patterns: {pattern_names}. Combined win rate: {combined_win_rate:.1%}"

        return SeasonalSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            direction=direction,
            strength=strength,
            confidence=confidence,
            active_patterns=patterns,
            combined_win_rate=combined_win_rate,
            rationale=rationale,
        )

    def scan_all_symbols(
        self,
        check_date: date | None = None
    ) -> list[SeasonalSignal]:
        """
        Scan all symbols for active seasonal signals.

        Args:
            check_date: Date to check

        Returns:
            List of SeasonalSignals
        """
        signals = []

        for symbol in self._enabled_symbols:
            signal = self.generate_signal(symbol, check_date)
            if signal:
                signals.append(signal)

        # Sort by confidence descending
        signals.sort(key=lambda s: s.confidence, reverse=True)

        return signals

    def get_upcoming_patterns(
        self,
        days_ahead: int = 30
    ) -> list[tuple[date, SeasonalWindow]]:
        """
        Get patterns starting within the next N days.

        Args:
            days_ahead: Days to look ahead

        Returns:
            List of (start_date, pattern) tuples
        """
        upcoming = []
        today = date.today()
        current_year = today.year

        for symbol, patterns in self._patterns.items():
            if symbol not in self._enabled_symbols:
                continue

            for pattern in patterns:
                # Try this year and next year
                for year in [current_year, current_year + 1]:
                    try:
                        start_date = date(year, pattern.start_month, pattern.start_day)
                    except ValueError:
                        continue

                    days_until = (start_date - today).days

                    if 0 < days_until <= days_ahead:
                        upcoming.append((start_date, pattern))

        # Sort by date
        upcoming.sort(key=lambda x: x[0])

        return upcoming

    def add_pattern(self, window: SeasonalWindow) -> None:
        """Add a custom seasonal pattern."""
        if window.symbol not in self._patterns:
            self._patterns[window.symbol] = []
        self._patterns[window.symbol].append(window)

    def get_all_patterns(self, symbol: str | None = None) -> list[SeasonalWindow]:
        """Get all patterns, optionally filtered by symbol."""
        if symbol:
            return list(self._patterns.get(symbol, []))

        all_patterns = []
        for patterns in self._patterns.values():
            all_patterns.extend(patterns)
        return all_patterns

    def get_status(self) -> dict[str, Any]:
        """Get strategy status."""
        total_patterns = sum(len(p) for p in self._patterns.values())

        return {
            "symbols_tracked": len(self._patterns),
            "enabled_symbols": len(self._enabled_symbols),
            "total_patterns": total_patterns,
            "min_win_rate": self._min_win_rate,
            "min_patterns": self._min_patterns,
            "active_signals": len(self.scan_all_symbols()),
        }

    # =========================================================================
    # PATTERN VALIDATION AND HISTORICAL ANALYSIS
    # =========================================================================

    def validate_pattern(
        self,
        pattern: SeasonalWindow,
        price_data: np.ndarray,
        dates: np.ndarray,
        lookback_years: int = 10,
    ) -> dict[str, Any]:
        """
        Validate a seasonal pattern against historical price data.

        Calculates actual win rate and returns based on historical performance.

        Args:
            pattern: SeasonalWindow to validate
            price_data: Historical prices (close prices)
            dates: Corresponding dates as numpy datetime64 array
            lookback_years: Years of history to analyze

        Returns:
            Dictionary with validation metrics
        """
        if len(price_data) != len(dates):
            return {"error": "Price data and dates must have same length"}

        # Convert to pandas-like date handling
        wins = 0
        total_trades = 0
        returns = []

        # Convert dates to year/month/day
        for year in range(datetime.now().year - lookback_years, datetime.now().year):
            try:
                # Find pattern window for this year
                start_date = date(year, pattern.start_month, pattern.start_day)
                if pattern.start_month <= pattern.end_month:
                    end_date = date(year, pattern.end_month, pattern.end_day)
                else:
                    end_date = date(year + 1, pattern.end_month, pattern.end_day)

                # Find indices for start and end dates
                start_idx = None
                end_idx = None

                for i, d in enumerate(dates):
                    dt = np.datetime64(d).astype('datetime64[D]').astype(date)
                    if dt >= start_date and start_idx is None:
                        start_idx = i
                    if dt >= end_date and end_idx is None:
                        end_idx = i
                        break

                if start_idx is None or end_idx is None or start_idx >= end_idx:
                    continue

                # Calculate return for this window
                entry_price = price_data[start_idx]
                exit_price = price_data[end_idx]

                if entry_price <= 0:
                    continue

                period_return = (exit_price - entry_price) / entry_price

                # Adjust for direction
                if pattern.direction == "short":
                    period_return = -period_return

                returns.append(period_return)
                total_trades += 1

                if period_return > 0:
                    wins += 1

            except (ValueError, IndexError):
                continue

        if total_trades == 0:
            return {
                "pattern_name": pattern.name,
                "trades": 0,
                "actual_win_rate": None,
                "avg_return": None,
                "error": "No valid trades found in historical data",
            }

        actual_win_rate = wins / total_trades
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if len(returns) > 1 else 0

        return {
            "pattern_name": pattern.name,
            "symbol": pattern.symbol,
            "trades": total_trades,
            "wins": wins,
            "actual_win_rate": actual_win_rate,
            "expected_win_rate": pattern.historical_win_rate,
            "win_rate_diff": actual_win_rate - pattern.historical_win_rate,
            "avg_return": avg_return,
            "expected_avg_return": pattern.avg_return,
            "std_return": std_return,
            "sharpe_ratio": avg_return / std_return if std_return > 0 else 0,
            "returns": returns,
        }

    def update_pattern_from_history(
        self,
        symbol: str,
        pattern_name: str,
        price_data: np.ndarray,
        dates: np.ndarray,
        lookback_years: int = 10,
    ) -> bool:
        """
        Update pattern's win rate and avg return based on historical data.

        Args:
            symbol: Symbol to update
            pattern_name: Name of pattern to update
            price_data: Historical prices
            dates: Corresponding dates
            lookback_years: Years to analyze

        Returns:
            True if pattern was updated, False otherwise
        """
        if symbol not in self._patterns:
            return False

        for i, pattern in enumerate(self._patterns[symbol]):
            if pattern.name == pattern_name:
                validation = self.validate_pattern(
                    pattern, price_data, dates, lookback_years
                )

                if validation.get("actual_win_rate") is not None:
                    # Create updated pattern (dataclass is frozen-like)
                    updated = SeasonalWindow(
                        name=pattern.name,
                        symbol=pattern.symbol,
                        pattern_type=pattern.pattern_type,
                        start_month=pattern.start_month,
                        start_day=pattern.start_day,
                        end_month=pattern.end_month,
                        end_day=pattern.end_day,
                        direction=pattern.direction,
                        historical_win_rate=validation["actual_win_rate"],
                        avg_return=validation["avg_return"],
                        description=pattern.description,
                        tags=pattern.tags,
                    )
                    self._patterns[symbol][i] = updated
                    logger.info(
                        f"Updated {pattern_name}: win_rate {pattern.historical_win_rate:.2%} -> "
                        f"{validation['actual_win_rate']:.2%}"
                    )
                    return True

        return False

    def calculate_pattern_confluence(
        self,
        symbol: str,
        check_date: date | None = None,
    ) -> dict[str, Any]:
        """
        Calculate pattern confluence score for a symbol.

        Multiple overlapping patterns in same direction = stronger signal.
        Conflicting patterns = weaker signal.

        Args:
            symbol: Symbol to analyze
            check_date: Date to check

        Returns:
            Dictionary with confluence analysis
        """
        active = self.get_active_patterns(symbol, check_date)

        if not active:
            return {
                "symbol": symbol,
                "confluence_score": 0,
                "direction": "neutral",
                "patterns_long": 0,
                "patterns_short": 0,
                "is_conflicting": False,
            }

        long_patterns = [p for p in active if p.direction == "long"]
        short_patterns = [p for p in active if p.direction == "short"]

        # Calculate weighted scores
        long_score = sum(p.historical_win_rate * (1 + p.avg_return) for p in long_patterns)
        short_score = sum(p.historical_win_rate * (1 + p.avg_return) for p in short_patterns)

        # Determine direction and confluence
        is_conflicting = len(long_patterns) > 0 and len(short_patterns) > 0

        if long_score > short_score:
            direction = "long"
            net_score = long_score - short_score * 0.5  # Conflicting patterns reduce score
        elif short_score > long_score:
            direction = "short"
            net_score = short_score - long_score * 0.5
        else:
            direction = "neutral"
            net_score = 0

        # Normalize confluence score (0-1 range)
        max_possible = len(active) * 1.5  # Theoretical max
        confluence_score = min(1.0, net_score / max_possible) if max_possible > 0 else 0

        return {
            "symbol": symbol,
            "date": (check_date or date.today()).isoformat(),
            "confluence_score": confluence_score,
            "direction": direction,
            "patterns_long": len(long_patterns),
            "patterns_short": len(short_patterns),
            "long_score": long_score,
            "short_score": short_score,
            "is_conflicting": is_conflicting,
            "active_pattern_names": [p.name for p in active],
        }

    def get_monthly_bias(self, symbol: str) -> dict[int, dict[str, float]]:
        """
        Get monthly directional bias for a symbol based on all patterns.

        Returns:
            Dictionary of month -> {"long": score, "short": score, "net": bias}
        """
        monthly_bias = {}

        for month in range(1, 13):
            long_score = 0.0
            short_score = 0.0

            for pattern in self._patterns.get(symbol, []):
                # Check if pattern is active in this month
                test_date = date(2024, month, 15)  # Use middle of month
                if pattern.is_active(test_date):
                    score = pattern.historical_win_rate * pattern.avg_return
                    if pattern.direction == "long":
                        long_score += score
                    else:
                        short_score += score

            monthly_bias[month] = {
                "long": long_score,
                "short": short_score,
                "net": long_score - short_score,
                "bias": "long" if long_score > short_score else ("short" if short_score > long_score else "neutral"),
            }

        return monthly_bias

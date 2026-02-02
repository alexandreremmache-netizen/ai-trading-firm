"""
Commodity Analytics Module
==========================

Addresses issues:
- #F18: Commodity sector rotation not implemented
- #F19: No commodity index tracking
- #F20: Missing commodity correlation matrix
- #F21: No commodity fundamental data feeds

Features:
- Sector rotation signals for commodities
- Index tracking (CRB, GSCI, BCOM)
- Correlation analysis across commodity groups
- Fundamental data integration framework
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


# Normalization factor for signal strength calculation in sector rotation.
# Represents the expected spread range (as a decimal) between top and bottom
# sectors for strong market regimes. A spread of 20% (0.20) is considered
# a fully differentiated market, yielding signal_strength of 1.0.
SIGNAL_STRENGTH_NORMALIZATION_FACTOR = 0.20


class CommoditySector(str, Enum):
    """Commodity sectors for rotation analysis."""
    ENERGY = "energy"
    PRECIOUS_METALS = "precious_metals"
    BASE_METALS = "base_metals"
    AGRICULTURE = "agriculture"
    LIVESTOCK = "livestock"
    SOFTS = "softs"


class CommodityIndex(str, Enum):
    """Major commodity indices."""
    CRB = "crb"  # CRB Commodity Index
    GSCI = "gsci"  # S&P GSCI
    BCOM = "bcom"  # Bloomberg Commodity Index
    DJP = "djp"  # iPath Bloomberg Commodity Index
    PDBC = "pdbc"  # Invesco Optimum Yield Diversified Commodity


# Commodity sector classifications
COMMODITY_SECTORS = {
    CommoditySector.ENERGY: [
        "CL",  # Crude Oil
        "NG",  # Natural Gas
        "RB",  # RBOB Gasoline
        "HO",  # Heating Oil
        "BZ",  # Brent Crude
    ],
    CommoditySector.PRECIOUS_METALS: [
        "GC",  # Gold
        "SI",  # Silver
        "PL",  # Platinum
        "PA",  # Palladium
    ],
    CommoditySector.BASE_METALS: [
        "HG",  # Copper
        "ALI",  # Aluminum
        "ZINC",  # Zinc (changed from ZN to avoid collision with Treasury Notes)
        "NI",  # Nickel
        "PB",  # Lead
    ],
    CommoditySector.AGRICULTURE: [
        "ZC",  # Corn
        "ZS",  # Soybeans
        "ZW",  # Wheat
        "ZM",  # Soybean Meal
        "ZL",  # Soybean Oil
        "ZO",  # Oats
        "ZR",  # Rough Rice
    ],
    CommoditySector.LIVESTOCK: [
        "LE",  # Live Cattle
        "GF",  # Feeder Cattle
        "HE",  # Lean Hogs
    ],
    CommoditySector.SOFTS: [
        "KC",  # Coffee
        "CT",  # Cotton
        "SB",  # Sugar
        "CC",  # Cocoa
        "OJ",  # Orange Juice
    ],
}


@dataclass
class SectorMetrics:
    """Metrics for a commodity sector."""
    sector: CommoditySector
    return_1d: float = 0.0
    return_1w: float = 0.0
    return_1m: float = 0.0
    return_3m: float = 0.0
    return_6m: float = 0.0
    return_ytd: float = 0.0
    volatility_20d: float = 0.0
    momentum_score: float = 0.0
    relative_strength: float = 0.0
    trend_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "sector": self.sector.value,
            "returns": {
                "1d_pct": self.return_1d * 100,
                "1w_pct": self.return_1w * 100,
                "1m_pct": self.return_1m * 100,
                "3m_pct": self.return_3m * 100,
                "6m_pct": self.return_6m * 100,
                "ytd_pct": self.return_ytd * 100,
            },
            "volatility_20d_pct": self.volatility_20d * 100,
            "momentum_score": self.momentum_score,
            "relative_strength": self.relative_strength,
            "trend_score": self.trend_score,
            "rank": self.rank,
        }


@dataclass
class SectorRotationSignal:
    """Sector rotation trading signal (#F18)."""
    timestamp: datetime
    overweight_sectors: list[CommoditySector]
    underweight_sectors: list[CommoditySector]
    neutral_sectors: list[CommoditySector]
    recommended_weights: dict[CommoditySector, float]
    signal_strength: float  # 0-1
    confidence: float  # 0-1
    reasoning: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overweight": [s.value for s in self.overweight_sectors],
            "underweight": [s.value for s in self.underweight_sectors],
            "neutral": [s.value for s in self.neutral_sectors],
            "weights": {s.value: w for s, w in self.recommended_weights.items()},
            "signal_strength": self.signal_strength,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


class CommoditySectorRotation:
    """
    Commodity sector rotation strategy (#F18).

    Analyzes momentum and relative strength across commodity sectors
    to generate rotation signals.
    """

    def __init__(
        self,
        lookback_momentum: int = 20,
        lookback_trend: int = 50,
        rebalance_threshold: float = 0.05,
    ):
        """
        Initialize sector rotation analyzer.

        Args:
            lookback_momentum: Days for momentum calculation
            lookback_trend: Days for trend calculation
            rebalance_threshold: Min change for rebalance signal
        """
        self.lookback_momentum = lookback_momentum
        self.lookback_trend = lookback_trend
        self.rebalance_threshold = rebalance_threshold

        # Price history: sector -> list of (date, price)
        self._sector_prices: dict[CommoditySector, list[tuple[datetime, float]]] = {
            sector: [] for sector in CommoditySector
        }
        self._last_signal: SectorRotationSignal | None = None

    def update_sector_price(
        self,
        sector: CommoditySector,
        date: datetime,
        price: float,
    ) -> None:
        """Update sector price (typically from sector ETF or index)."""
        self._sector_prices[sector].append((date, price))

        # Keep reasonable history
        if len(self._sector_prices[sector]) > 500:
            self._sector_prices[sector] = self._sector_prices[sector][-500:]

    def calculate_sector_metrics(self) -> list[SectorMetrics]:
        """Calculate metrics for all sectors."""
        metrics = []

        for sector in CommoditySector:
            prices = self._sector_prices[sector]
            if len(prices) < 20:
                continue

            # Get returns for different periods
            current_price = prices[-1][1]

            returns = {}
            for period, days in [("1d", 1), ("1w", 5), ("1m", 20), ("3m", 60), ("6m", 120)]:
                if len(prices) >= days + 1:
                    past_price = prices[-(days + 1)][1]
                    if past_price > 0:
                        returns[period] = (current_price - past_price) / past_price
                    else:
                        returns[period] = 0.0
                else:
                    returns[period] = 0.0

            # Calculate volatility
            if len(prices) >= 21:
                daily_returns = [
                    (prices[i][1] - prices[i-1][1]) / prices[i-1][1]
                    for i in range(len(prices) - 20, len(prices))
                    if prices[i-1][1] > 0  # Guard against division by zero
                ]
                volatility = statistics.stdev(daily_returns) * math.sqrt(252)
            else:
                volatility = 0.0

            # Momentum score (weighted returns)
            momentum = (
                returns.get("1w", 0) * 0.2 +
                returns.get("1m", 0) * 0.3 +
                returns.get("3m", 0) * 0.5
            )

            # Trend score (price vs moving average)
            if len(prices) >= self.lookback_trend:
                ma_prices = [p for _, p in prices[-self.lookback_trend:]]
                ma = statistics.mean(ma_prices)
                if ma > 0:
                    trend = (current_price - ma) / ma
                else:
                    trend = 0.0
            else:
                trend = 0.0

            metrics.append(SectorMetrics(
                sector=sector,
                return_1d=returns.get("1d", 0),
                return_1w=returns.get("1w", 0),
                return_1m=returns.get("1m", 0),
                return_3m=returns.get("3m", 0),
                return_6m=returns.get("6m", 0),
                volatility_20d=volatility,
                momentum_score=momentum,
                trend_score=trend,
            ))

        # Calculate relative strength and rankings
        if metrics:
            momentum_values = [m.momentum_score for m in metrics]
            avg_momentum = statistics.mean(momentum_values) if momentum_values else 0

            for m in metrics:
                m.relative_strength = m.momentum_score - avg_momentum

            # Rank by momentum
            sorted_metrics = sorted(metrics, key=lambda x: x.momentum_score, reverse=True)
            for rank, m in enumerate(sorted_metrics, 1):
                m.rank = rank

        return metrics

    def generate_rotation_signal(self) -> SectorRotationSignal | None:
        """
        Generate sector rotation signal.

        Returns signal with recommended over/underweight sectors.
        """
        metrics = self.calculate_sector_metrics()
        if not metrics:
            return None

        # Sort by composite score (momentum + trend)
        for m in metrics:
            m.relative_strength = m.momentum_score + m.trend_score * 0.5

        sorted_metrics = sorted(metrics, key=lambda x: x.relative_strength, reverse=True)

        # Top third = overweight, bottom third = underweight
        n = len(sorted_metrics)
        n_third = max(1, n // 3)

        overweight = [m.sector for m in sorted_metrics[:n_third]]
        underweight = [m.sector for m in sorted_metrics[-n_third:]]
        if n_third < n - n_third:
            neutral = [m.sector for m in sorted_metrics[n_third:-n_third]]
        else:
            neutral = []

        # Calculate recommended weights
        total_sectors = len(sorted_metrics)
        base_weight = 1.0 / total_sectors

        weights = {}
        for m in sorted_metrics:
            if m.sector in overweight:
                weights[m.sector] = base_weight * 1.5  # 50% overweight
            elif m.sector in underweight:
                weights[m.sector] = base_weight * 0.5  # 50% underweight
            else:
                weights[m.sector] = base_weight

        # Normalize
        total_weight = sum(weights.values())
        weights = {s: w / total_weight for s, w in weights.items()}

        # Signal strength based on spread
        if len(sorted_metrics) >= 2:
            spread = sorted_metrics[0].relative_strength - sorted_metrics[-1].relative_strength
            signal_strength = min(1.0, abs(spread) / SIGNAL_STRENGTH_NORMALIZATION_FACTOR)
        else:
            signal_strength = 0.5

        # Confidence based on trend alignment
        aligned = sum(1 for m in sorted_metrics if m.momentum_score * m.trend_score > 0)
        confidence = aligned / len(sorted_metrics) if sorted_metrics else 0.5

        signal = SectorRotationSignal(
            timestamp=datetime.now(timezone.utc),
            overweight_sectors=overweight,
            underweight_sectors=underweight,
            neutral_sectors=neutral,
            recommended_weights=weights,
            signal_strength=signal_strength,
            confidence=confidence,
            reasoning=self._generate_reasoning(sorted_metrics),
        )

        self._last_signal = signal
        return signal

    def _generate_reasoning(self, metrics: list[SectorMetrics]) -> str:
        """Generate reasoning for signal."""
        if not metrics:
            return "Insufficient data for analysis"

        top = metrics[0]
        bottom = metrics[-1]

        return (
            f"Overweight {top.sector.value} (momentum: {top.momentum_score:.2%}, "
            f"trend: {top.trend_score:.2%}). "
            f"Underweight {bottom.sector.value} (momentum: {bottom.momentum_score:.2%}, "
            f"trend: {bottom.trend_score:.2%})."
        )


# =========================================================================
# COMMODITY INDEX TRACKING (#F19)
# =========================================================================

@dataclass
class IndexComposition:
    """Composition of a commodity index."""
    index: CommodityIndex
    components: dict[str, float]  # symbol -> weight
    last_updated: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "index": self.index.value,
            "components": self.components,
            "last_updated": self.last_updated.isoformat(),
        }


class CommodityIndexTracker:
    """
    Tracks commodity indices (#F19).

    Replicates index performance and calculates tracking error.
    """

    # Approximate index compositions (simplified)
    INDEX_COMPOSITIONS = {
        CommodityIndex.BCOM: {
            "CL": 0.15,  # WTI Crude
            "NG": 0.08,  # Natural Gas
            "GC": 0.12,  # Gold
            "SI": 0.03,  # Silver
            "HG": 0.07,  # Copper
            "ZC": 0.07,  # Corn
            "ZS": 0.06,  # Soybeans
            "ZW": 0.04,  # Wheat
            "SB": 0.04,  # Sugar
            "KC": 0.03,  # Coffee
            "CT": 0.02,  # Cotton
            "LE": 0.03,  # Live Cattle
            "HE": 0.02,  # Lean Hogs
            "ALI": 0.05,  # Aluminum
        },
        CommodityIndex.GSCI: {
            "CL": 0.24,
            "BZ": 0.12,
            "NG": 0.05,
            "RB": 0.04,
            "HO": 0.04,
            "GC": 0.04,
            "SI": 0.01,
            "HG": 0.04,
            "ALI": 0.03,
            "ZC": 0.06,
            "ZS": 0.03,
            "ZW": 0.05,
            "LE": 0.03,
            "HE": 0.02,
            "KC": 0.01,
            "SB": 0.03,
            "CT": 0.02,
            "CC": 0.01,
        },
    }

    def __init__(self):
        self._price_history: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        self._index_values: dict[CommodityIndex, list[tuple[datetime, float]]] = defaultdict(list)

    def update_price(self, symbol: str, date: datetime, price: float) -> None:
        """Update component price."""
        self._price_history[symbol].append((date, price))

    def calculate_index_value(
        self,
        index: CommodityIndex,
        as_of: datetime | None = None,
    ) -> float | None:
        """
        Calculate theoretical index value.

        Args:
            index: Index to calculate
            as_of: Date (default: latest)

        Returns:
            Index value or None if insufficient data
        """
        composition = self.INDEX_COMPOSITIONS.get(index)
        if not composition:
            return None

        total_value = 0.0
        total_weight = 0.0

        for symbol, weight in composition.items():
            prices = self._price_history.get(symbol, [])
            if not prices:
                continue

            if as_of:
                # Find price at or before as_of
                relevant_prices = [p for d, p in prices if d <= as_of]
                if not relevant_prices:
                    continue
                price = relevant_prices[-1]
            else:
                price = prices[-1][1]

            # Validate price before use
            if price <= 0:
                continue

            total_value += price * weight
            total_weight += weight

        if total_weight == 0:
            return None

        # Normalize
        return total_value / total_weight * 100

    def get_index_returns(
        self,
        index: CommodityIndex,
        period_days: int = 20,
    ) -> dict[str, float]:
        """Get index returns for different periods."""
        if index not in self._index_values or len(self._index_values[index]) < 2:
            return {}

        values = self._index_values[index]
        current = values[-1][1]

        returns = {}
        for period_name, days in [("1d", 1), ("1w", 5), ("1m", 20), ("3m", 60)]:
            if len(values) >= days + 1:
                past_value = values[-(days + 1)][1]
                returns[period_name] = (current - past_value) / past_value
            else:
                returns[period_name] = 0.0

        return returns

    def calculate_tracking_error(
        self,
        portfolio_returns: list[float],
        index_returns: list[float],
    ) -> float:
        """Calculate tracking error vs benchmark."""
        if len(portfolio_returns) != len(index_returns) or len(portfolio_returns) < 2:
            return 0.0

        tracking_diffs = [p - i for p, i in zip(portfolio_returns, index_returns)]
        return statistics.stdev(tracking_diffs) * math.sqrt(252)


# =========================================================================
# COMMODITY CORRELATION MATRIX (#F20)
# =========================================================================

@dataclass
class CorrelationMatrix:
    """Correlation matrix for commodities."""
    symbols: list[str]
    correlations: list[list[float]]
    as_of_date: datetime
    lookback_days: int

    def get_correlation(self, symbol1: str, symbol2: str) -> float | None:
        """Get correlation between two symbols."""
        try:
            i = self.symbols.index(symbol1)
            j = self.symbols.index(symbol2)
            return self.correlations[i][j]
        except (ValueError, IndexError):
            return None

    def get_highly_correlated(self, threshold: float = 0.7) -> list[tuple[str, str, float]]:
        """Get pairs with correlation above threshold."""
        pairs = []
        for i, sym1 in enumerate(self.symbols):
            for j, sym2 in enumerate(self.symbols):
                if i < j:
                    corr = self.correlations[i][j]
                    if abs(corr) >= threshold:
                        pairs.append((sym1, sym2, corr))
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbols": self.symbols,
            "correlations": self.correlations,
            "as_of_date": self.as_of_date.isoformat(),
            "lookback_days": self.lookback_days,
        }


class CommodityCorrelationAnalyzer:
    """
    Analyzes correlations across commodities (#F20).

    Tracks rolling correlations and detects regime changes.
    """

    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self._returns: dict[str, list[float]] = defaultdict(list)
        self._last_matrix: CorrelationMatrix | None = None

    def update_return(self, symbol: str, daily_return: float) -> None:
        """Update daily return for a symbol."""
        self._returns[symbol].append(daily_return)

        # Keep only lookback period
        if len(self._returns[symbol]) > self.lookback_days * 2:
            self._returns[symbol] = self._returns[symbol][-self.lookback_days:]

    def calculate_correlation_matrix(
        self,
        symbols: list[str] | None = None,
    ) -> CorrelationMatrix | None:
        """
        Calculate correlation matrix.

        Args:
            symbols: Symbols to include (default: all)

        Returns:
            CorrelationMatrix or None if insufficient data
        """
        if symbols is None:
            symbols = list(self._returns.keys())

        # Filter symbols with enough data
        valid_symbols = [
            s for s in symbols
            if len(self._returns.get(s, [])) >= self.lookback_days
        ]

        if len(valid_symbols) < 2:
            return None

        # Get aligned returns
        min_len = min(len(self._returns[s]) for s in valid_symbols)
        returns_matrix = [
            self._returns[s][-min_len:]
            for s in valid_symbols
        ]

        # Calculate correlations
        n = len(valid_symbols)
        correlations = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    correlations[i][j] = 1.0
                else:
                    correlations[i][j] = self._pearson_correlation(
                        returns_matrix[i],
                        returns_matrix[j],
                    )

        matrix = CorrelationMatrix(
            symbols=valid_symbols,
            correlations=correlations,
            as_of_date=datetime.now(timezone.utc),
            lookback_days=min_len,
        )

        self._last_matrix = matrix
        return matrix

    def _pearson_correlation(self, x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        denominator = math.sqrt(var_x * var_y)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_sector_correlations(self) -> dict[tuple[CommoditySector, CommoditySector], float]:
        """Get average correlations between sectors."""
        if self._last_matrix is None:
            self.calculate_correlation_matrix()

        if self._last_matrix is None:
            return {}

        sector_corrs = defaultdict(list)

        for i, sym1 in enumerate(self._last_matrix.symbols):
            for j, sym2 in enumerate(self._last_matrix.symbols):
                if i < j:
                    sec1 = self._get_sector(sym1)
                    sec2 = self._get_sector(sym2)

                    if sec1 and sec2:
                        key = (min(sec1, sec2), max(sec1, sec2))
                        sector_corrs[key].append(self._last_matrix.correlations[i][j])

        return {
            k: statistics.mean(v) if v else 0.0
            for k, v in sector_corrs.items()
        }

    def _get_sector(self, symbol: str) -> CommoditySector | None:
        """Get sector for a symbol."""
        for sector, symbols in COMMODITY_SECTORS.items():
            if symbol in symbols:
                return sector
        return None


# =========================================================================
# FUNDAMENTAL DATA FEEDS (#F21)
# =========================================================================

@dataclass
class FundamentalData:
    """Fundamental data point for a commodity."""
    symbol: str
    date: datetime
    data_type: str
    value: float
    unit: str
    source: str
    metadata: dict = field(default_factory=dict)


class CommodityFundamentalDataManager:
    """
    Manages commodity fundamental data feeds (#F21).

    Framework for integrating various fundamental data sources.
    """

    # Data types
    DATA_TYPES = [
        "inventory",
        "production",
        "consumption",
        "imports",
        "exports",
        "refinery_utilization",
        "rig_count",
        "planted_acres",
        "crop_progress",
        "weather_forecast",
        "cot_commercial_net",
        "cot_noncommercial_net",
    ]

    def __init__(self):
        self._data: dict[str, list[FundamentalData]] = defaultdict(list)
        self._data_sources: dict[str, Callable] = {}

    def register_data_source(
        self,
        data_type: str,
        fetch_function: Callable[[str, datetime], FundamentalData | None],
    ) -> None:
        """
        Register a fundamental data source.

        Args:
            data_type: Type of data
            fetch_function: Function to fetch data (symbol, date) -> data
        """
        self._data_sources[data_type] = fetch_function
        logger.info(f"Registered fundamental data source: {data_type}")

    def store_data(self, data: FundamentalData) -> None:
        """Store fundamental data point."""
        key = f"{data.symbol}:{data.data_type}"
        self._data[key].append(data)

        # Keep reasonable history
        if len(self._data[key]) > 1000:
            self._data[key] = self._data[key][-1000:]

    def get_latest(
        self,
        symbol: str,
        data_type: str,
    ) -> FundamentalData | None:
        """Get latest fundamental data for a symbol."""
        key = f"{symbol}:{data_type}"
        data_list = self._data.get(key, [])
        return data_list[-1] if data_list else None

    def get_time_series(
        self,
        symbol: str,
        data_type: str,
        lookback_days: int = 365,
    ) -> list[FundamentalData]:
        """Get time series of fundamental data."""
        key = f"{symbol}:{data_type}"
        data_list = self._data.get(key, [])

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        return [d for d in data_list if d.date >= cutoff]

    def get_supply_demand_balance(self, symbol: str) -> dict | None:
        """Calculate supply/demand balance for a commodity."""
        production = self.get_latest(symbol, "production")
        consumption = self.get_latest(symbol, "consumption")
        inventory = self.get_latest(symbol, "inventory")

        if not all([production, consumption]):
            return None

        balance = production.value - consumption.value

        return {
            "symbol": symbol,
            "production": production.value,
            "consumption": consumption.value,
            "balance": balance,
            "inventory": inventory.value if inventory else None,
            "balance_pct": (balance / consumption.value * 100) if consumption.value else 0,
            "as_of": max(production.date, consumption.date).isoformat(),
        }

    def get_available_data_types(self, symbol: str) -> list[str]:
        """Get available data types for a symbol."""
        types = []
        for key in self._data.keys():
            if key.startswith(f"{symbol}:"):
                data_type = key.split(":")[1]
                types.append(data_type)
        return types

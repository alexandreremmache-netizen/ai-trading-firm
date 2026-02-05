"""
Statistical Arbitrage Strategy
==============================

Implements pairs trading, mean reversion, and commodity spreads.

MATURITY: BETA
--------------
Status: Comprehensive implementation with commodity spreads
- [x] Cointegration testing (simplified ADF)
- [x] Hedge ratio estimation (OLS)
- [x] Half-life calculation (OU process)
- [x] Commodity spreads (crack, crush, inter-commodity)
- [x] Optimal lag selection for ADF (#Q3)
- [x] Dollar-neutral spread sizing
- [x] Johansen cointegration test (P3 implementation)
- [x] Kalman filter for dynamic hedge ratio (Phase 5.1)
- [x] Transaction cost modeling (Phase 5.2 in position_sizing.py)

Production Readiness:
- Unit tests: Partial coverage
- Backtesting: Spread definitions validated historically
- Live testing: Not yet performed

Use in production: WITH CAUTION
- Commodity spread ratios are industry-standard
- ADF test is simplified; production should use statsmodels
- Monitor half-life for regime changes

Features:
- Pairs trading (cointegration-based)
- Commodity spreads (crack, crush, calendar)
- Contract specs integration for proper sizing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

# Phase 5.1: Kalman Filter for dynamic hedge ratio
try:
    from core.kalman_filter import KalmanHedgeRatio, create_kalman_filter
    HAS_KALMAN = True
except ImportError:
    HAS_KALMAN = False
    KalmanHedgeRatio = None  # type: ignore


logger = logging.getLogger(__name__)


class HedgeRatioMethod(Enum):
    """Hedge ratio estimation method."""
    OLS = "ols"              # Static OLS regression
    ROLLING_OLS = "rolling"  # Rolling window OLS
    KALMAN = "kalman"        # Kalman filter (dynamic)


class SpreadType(Enum):
    """Types of spread trades."""
    PAIRS = "pairs"  # Simple pairs trade
    CRACK = "crack"  # Crude to products
    CRUSH = "crush"  # Soybeans to meal/oil
    CALENDAR = "calendar"  # Same commodity, different months
    INTER_COMMODITY = "inter_commodity"  # Related commodities


@dataclass
class CommoditySpread:
    """Definition of a commodity spread trade."""
    name: str
    spread_type: SpreadType
    legs: dict[str, float]  # symbol -> ratio (positive = long, negative = short)
    description: str
    typical_range: tuple[float, float] = (0.0, 0.0)  # Historical range
    seasonality: list[int] = field(default_factory=list)  # Best months
    margin_offset_pct: float = 0.0  # Margin reduction for recognized spread
    storage_cost_annual_pct: float = 0.0  # Annual storage/carry cost as percentage (COM-005)


# =============================================================================
# PREDEFINED COMMODITY SPREADS
# =============================================================================

COMMODITY_SPREADS: dict[str, CommoditySpread] = {
    # =========================================================================
    # CRACK SPREADS (Crude Oil to Products)
    # =========================================================================
    "3-2-1_crack": CommoditySpread(
        name="3:2:1 Crack Spread",
        spread_type=SpreadType.CRACK,
        legs={
            "CL": -3.0,  # Short 3 crude
            "RB": 2.0,   # Long 2 gasoline
            "HO": 1.0,   # Long 1 heating oil
        },
        description="Standard refinery margin: 3 barrels crude -> 2 gasoline + 1 heating oil",
        typical_range=(5.0, 35.0),  # USD per barrel
        seasonality=[2, 3, 4, 5],  # Feb-May (pre-driving season)
        margin_offset_pct=75.0,
        storage_cost_annual_pct=3.5,  # COM-005: Crude/products storage ~3-4% annually
    ),

    "gasoline_crack": CommoditySpread(
        name="Gasoline Crack Spread",
        spread_type=SpreadType.CRACK,
        legs={
            "CL": -1.0,  # Short crude
            "RB": 1.0,   # Long gasoline
        },
        description="Simple gasoline refining margin",
        typical_range=(5.0, 40.0),
        seasonality=[2, 3, 4, 5, 6],  # Feb-Jun
        margin_offset_pct=70.0,
        storage_cost_annual_pct=4.0,  # COM-005: Gasoline higher storage cost due to volatility
    ),

    "heating_oil_crack": CommoditySpread(
        name="Heating Oil Crack Spread",
        spread_type=SpreadType.CRACK,
        legs={
            "CL": -1.0,
            "HO": 1.0,
        },
        description="Heating oil refining margin",
        typical_range=(5.0, 30.0),
        seasonality=[9, 10, 11, 12],  # Sep-Dec (pre-winter)
        margin_offset_pct=70.0,
        storage_cost_annual_pct=3.0,  # COM-005: Heating oil stable storage
    ),

    # =========================================================================
    # CRUSH SPREADS (Soybeans to Products)
    # =========================================================================
    # CBOT Board Crush: 10 ZS : 12 ZM : 9 ZL (contract ratio)
    # Simplified: 1 ZS : 1.1 ZM : 0.9 ZL
    # Yield: 1 bushel soybeans (60 lbs) -> ~48 lbs meal + ~11 lbs oil
    # ZS = 5000 bu, ZM = 100 short tons (200,000 lbs), ZL = 60,000 lbs
    "soybean_crush": CommoditySpread(
        name="Soybean Crush Spread",
        spread_type=SpreadType.CRUSH,
        legs={
            "ZS": -10.0,   # Short 10 soybean contracts (50,000 bushels)
            "ZM": 12.0,    # Long 12 soybean meal contracts
            "ZL": 9.0,     # Long 9 soybean oil contracts
        },
        description="CBOT Board Crush: 10:12:9 ratio (processing margin)",
        typical_range=(0.50, 1.50),  # USD per bushel gross crush margin
        seasonality=[9, 10, 11],  # Post-harvest
        margin_offset_pct=60.0,
        storage_cost_annual_pct=5.0,  # COM-005: Grain storage ~4-6% annually
    ),

    "reverse_crush": CommoditySpread(
        name="Reverse Crush Spread",
        spread_type=SpreadType.CRUSH,
        legs={
            "ZS": 10.0,    # Long 10 soybean contracts
            "ZM": -12.0,   # Short 12 soybean meal contracts
            "ZL": -9.0,    # Short 9 soybean oil contracts
        },
        description="Reverse of board crush - long beans, short products",
        typical_range=(-1.50, -0.50),
        seasonality=[3, 4, 5],  # Planting season
        margin_offset_pct=60.0,
        storage_cost_annual_pct=5.0,  # COM-005: Grain storage ~4-6% annually
    ),

    # Single contract crush (for smaller accounts)
    "mini_crush": CommoditySpread(
        name="Mini Crush Spread",
        spread_type=SpreadType.CRUSH,
        legs={
            "ZS": -1.0,    # Short 1 soybean contract
            "ZM": 1.2,     # Long ~1.2 meal contracts (round to 1)
            "ZL": 0.9,     # Long ~0.9 oil contracts (round to 1)
        },
        description="Simplified 1:1:1 crush for smaller positions",
        typical_range=(0.50, 1.50),
        seasonality=[9, 10, 11],
        margin_offset_pct=50.0,  # Lower margin offset due to ratio mismatch
        storage_cost_annual_pct=5.0,  # COM-005: Grain storage ~4-6% annually
    ),

    # =========================================================================
    # INTER-COMMODITY SPREADS
    # =========================================================================
    # IMPORTANT: Contract-adjusted ratios
    # GC (Gold): 100 troy oz, ~$2000/oz = ~$200,000 per contract
    # SI (Silver): 5000 troy oz, ~$25/oz = ~$125,000 per contract
    # Dollar-neutral ratio: $200k / $125k = 1.6 SI per 1 GC (NOT price ratio 80!)
    "gold_silver_ratio": CommoditySpread(
        name="Gold/Silver Ratio (Contract-Adjusted)",
        spread_type=SpreadType.INTER_COMMODITY,
        legs={
            "GC": 1.0,    # Long 1 gold (100 oz × $2000 = $200,000)
            "SI": -1.6,   # Short 1.6 silver (8000 oz × $25 = $200,000)
        },
        description="Dollar-neutral gold/silver spread. Trade the ratio mean reversion. "
                    "Ratio = (GC_price × 100) / (SI_price × 5000) for actual ratio exposure.",
        typical_range=(60.0, 90.0),  # Gold/Silver PRICE ratio range (for reference)
        margin_offset_pct=50.0,
        storage_cost_annual_pct=0.5,  # COM-005: Precious metals low storage cost
    ),

    # Alternative: Micro gold/silver for smaller accounts
    # MGC = 10 oz gold, SIL = 1000 oz silver
    "micro_gold_silver": CommoditySpread(
        name="Micro Gold/Silver Ratio",
        spread_type=SpreadType.INTER_COMMODITY,
        legs={
            "MGC": 1.0,   # Long 1 micro gold (10 oz × $2000 = $20,000)
            "SIL": -0.8,  # Short ~0.8 micro silver (800 oz × $25 = $20,000)
        },
        description="Micro contract version of gold/silver spread",
        typical_range=(60.0, 90.0),
        margin_offset_pct=40.0,
        storage_cost_annual_pct=0.5,  # COM-005: Precious metals low storage cost
    ),

    "corn_wheat": CommoditySpread(
        name="Corn/Wheat Spread",
        spread_type=SpreadType.INTER_COMMODITY,
        legs={
            # Both contracts = 5000 bushels, so 1:1 is dollar-neutral at similar prices
            "ZC": 1.0,   # Long corn (5000 bu × ~$4.50 = $22,500)
            "ZW": -1.0,  # Short wheat (5000 bu × ~$6.00 = $30,000)
        },
        description="Feed grain substitution spread. Corn/Wheat ratio typically 0.7-0.9.",
        typical_range=(-1.50, 0.50),  # Wheat premium over corn in USD/bushel
        seasonality=[6, 7, 8],  # Summer
        margin_offset_pct=40.0,
        storage_cost_annual_pct=5.5,  # COM-005: Grain storage ~5-6% annually
    ),

    # Corn/Soybean ratio (feed competition)
    "corn_soybean_ratio": CommoditySpread(
        name="Corn/Soybean Ratio",
        spread_type=SpreadType.INTER_COMMODITY,
        legs={
            # ZC = 5000 bu, ZS = 5000 bu (same multiplier)
            "ZC": 2.5,    # Long 2.5 corn to approximate soybean value
            "ZS": -1.0,   # Short 1 soybean
        },
        description="Corn/Soybean ratio (new crop planting decision spread). "
                    "Ratio typically 2.0-3.0, mean around 2.4.",
        typical_range=(2.0, 3.0),  # Soybean/Corn price ratio
        seasonality=[1, 2, 3],  # Planting decision season
        margin_offset_pct=40.0,
        storage_cost_annual_pct=5.0,  # COM-005: Grain storage ~4-6% annually
    ),

    "brent_wti": CommoditySpread(
        name="Brent/WTI Spread",
        spread_type=SpreadType.INTER_COMMODITY,
        legs={
            "CL": 1.0,   # Long WTI (CL)
            # Note: Brent (BZ) would be added here
            # "BZ": -1.0,  # Both contracts = 1000 barrels
        },
        description="Atlantic Basin crude oil spread (requires Brent contract)",
        typical_range=(-5.0, 10.0),
        margin_offset_pct=60.0,
        storage_cost_annual_pct=3.5,  # COM-005: Crude oil storage ~3-4% annually
    ),

    # Platinum/Gold ratio
    "platinum_gold_ratio": CommoditySpread(
        name="Platinum/Gold Ratio",
        spread_type=SpreadType.INTER_COMMODITY,
        legs={
            # PL = 50 oz, GC = 100 oz
            "PL": 2.0,    # Long 2 platinum (100 oz × ~$1000 = $100,000)
            "GC": -0.5,   # Short 0.5 gold (50 oz × $2000 = $100,000)
        },
        description="Platinum/Gold ratio spread. Platinum historically trades at premium, "
                    "currently at deep discount.",
        typical_range=(0.5, 1.2),  # Pt/Au price ratio
        margin_offset_pct=50.0,
        storage_cost_annual_pct=0.5,  # COM-005: Precious metals low storage cost
    ),

    # Copper/Gold ratio (economic indicator)
    "copper_gold_ratio": CommoditySpread(
        name="Copper/Gold Ratio",
        spread_type=SpreadType.INTER_COMMODITY,
        legs={
            # HG = 25000 lbs, GC = 100 oz
            # HG: 25000 × $4 = $100,000, GC: 100 × $2000 = $200,000
            "HG": 2.0,    # Long 2 copper ($200,000 notional)
            "GC": -1.0,   # Short 1 gold ($200,000 notional)
        },
        description="Copper/Gold ratio - economic sentiment indicator. "
                    "Higher ratio = risk-on, lower = risk-off.",
        typical_range=(0.0015, 0.0025),  # Cu/Au price ratio
        margin_offset_pct=40.0,
        storage_cost_annual_pct=2.0,  # COM-005: Base metals moderate storage cost
    ),
}


@dataclass
class SpreadAnalysis:
    """Analysis results for a spread trade."""
    spread_name: str
    spread_type: SpreadType
    current_value: float
    zscore: float
    percentile: float  # Where current value is in historical range
    mean: float
    std: float
    half_life: float
    is_tradeable: bool
    signal_direction: str  # "long_spread", "short_spread", "flat", "exit"
    signal_strength: float
    # RISK-001: Stop-loss and take-profit levels (in z-score terms)
    stop_loss_zscore: float = 4.0           # Exit if z-score exceeds this
    take_profit_zscore: float = 0.5         # Exit when z-score returns to this
    stop_loss_pct: float | None = None      # Stop in percentage terms
    take_profit_pct: float | None = None    # Take profit in percentage terms
    # RISK-002: Maximum holding period
    max_holding_bars: int = 60              # Max bars to hold spread position
    # RISK-003: Strategy-level risk limit
    strategy_max_loss_pct: float = 5.0      # Max loss per spread trade
    # RISK-004: Regime suitability
    regime_suitable: bool = True            # False if cointegration breaking down
    regime_warning: str | None = None       # Warning message about regime
    # RISK-005: Exit signal info
    is_exit_signal: bool = False
    exit_reason: str | None = None


@dataclass
class PairAnalysis:
    """Analysis results for a trading pair."""
    symbol_a: str
    symbol_b: str
    correlation: float
    cointegration_pvalue: float
    hedge_ratio: float
    half_life: float
    current_zscore: float
    is_cointegrated: bool
    rationale: str = ""  # Explanation for cointegration status (ISSUE_002)
    # P2: Rolling correlation monitoring
    rolling_correlation: float = 0.0  # Short-term rolling correlation
    correlation_stability: float = 0.0  # Stability score (0-1)
    # P2: Mean-reversion speed estimation
    reversion_speed: float = 0.0  # Daily mean reversion speed
    expected_time_to_mean: float = 0.0  # Expected bars to reach mean
    # P2: Entry timing optimization
    entry_score: float = 0.0  # 0-1 score for entry quality
    optimal_entry: bool = False  # True if timing is optimal
    # RISK-001: Stop-loss and take-profit levels
    stop_loss_zscore: float = 4.0           # Exit if z-score exceeds this
    take_profit_zscore: float = 0.5         # Target z-score for exit
    stop_loss_price_a: float | None = None  # Stop price for asset A
    stop_loss_price_b: float | None = None  # Stop price for asset B
    # RISK-002: Maximum holding period
    max_holding_bars: int = 60              # Max bars to hold
    # RISK-003: Strategy-level risk
    strategy_max_loss_pct: float = 5.0      # Max loss per trade
    # RISK-004: Regime detection
    regime_suitable: bool = True            # False if regime unfavorable
    regime_warning: str | None = None       # Warning message


class StatArbStrategy:
    """
    Statistical Arbitrage Strategy Implementation.

    Implements:
    1. Cointegration testing (Engle-Granger, Johansen)
    2. Hedge ratio estimation (OLS, Rolling OLS, Kalman filter)
    3. Mean reversion signal generation
    4. Half-life estimation (Ornstein-Uhlenbeck)

    Features:
    - Static OLS hedge ratio (fast, suitable for stable pairs)
    - Rolling OLS hedge ratio (adapts to slow regime changes)
    - Kalman filter hedge ratio (Phase 5.1 - adapts dynamically to changing relationships)
    - Johansen cointegration test (Phase 3 - supports multiple cointegrating vectors)

    Configuration:
        hedge_ratio_method: 'ols' | 'rolling' | 'kalman'
        kalman_delta: Process noise (default 1e-4)
        kalman_ve: Measurement noise (default 1e-3)
        kalman_warmup_period: Warmup period (default 30)
    """

    def __init__(self, config: dict[str, Any]):
        self._lookback = config.get("lookback_days", 60)
        self._zscore_entry = config.get("zscore_entry_threshold", 2.0)
        self._zscore_exit = config.get("zscore_exit_threshold", 0.5)
        self._min_half_life = config.get("min_half_life_days", 1)
        self._max_half_life = config.get("max_half_life_days", 30)
        # Regime change detection thresholds (ISSUE_002)
        self._correlation_spike_threshold = config.get("correlation_spike_threshold", 0.95)
        self._bars_per_day = config.get("bars_per_day", 390)  # Default: minute bars for US equity

        # P2: Rolling correlation monitoring settings
        self._rolling_corr_window = config.get("rolling_corr_window", 20)  # Short-term window
        self._corr_stability_threshold = config.get("corr_stability_threshold", 0.1)  # Max acceptable std

        # P2: Mean-reversion speed estimation settings
        self._speed_estimation_window = config.get("speed_estimation_window", 30)

        # P2: Entry timing optimization settings
        self._zscore_acceleration_threshold = config.get("zscore_acceleration_threshold", 0.1)
        self._optimal_entry_zscore_range = config.get("optimal_entry_zscore_range", (1.8, 2.5))

        # RISK-001: Stop-loss and take-profit settings
        self._stop_loss_zscore = config.get("stop_loss_zscore", 4.0)  # Exit if spread widens to 4 sigma
        self._take_profit_zscore = config.get("take_profit_zscore", 0.5)  # Exit when mean reverts

        # RISK-002: Maximum holding period
        self._max_holding_bars = config.get("max_holding_bars", 60)  # ~3 months daily data

        # RISK-003: Strategy-level risk limit
        self._strategy_max_loss_pct = config.get("strategy_max_loss_pct", 5.0)

        # RISK-004: Regime detection thresholds
        self._regime_breakdown_pvalue = config.get("regime_breakdown_pvalue", 0.15)  # p-value threshold
        self._min_correlation_stability = config.get("min_correlation_stability", 0.3)

        # Phase 5.1: Kalman Filter settings for dynamic hedge ratio
        self._hedge_ratio_method = HedgeRatioMethod(
            config.get("hedge_ratio_method", "ols")
        )
        self._kalman_delta = config.get("kalman_delta", 1e-4)  # Process noise
        self._kalman_ve = config.get("kalman_ve", 1e-3)  # Measurement noise
        self._kalman_warmup = config.get("kalman_warmup_period", 30)
        self._kalman_include_intercept = config.get("kalman_include_intercept", True)

        # Kalman filter cache: (symbol_a, symbol_b) -> KalmanHedgeRatio instance
        self._kalman_filters: dict[tuple[str, str], Any] = {}

        logger.info(
            f"StatArbStrategy initialized: hedge_method={self._hedge_ratio_method.value}, "
            f"zscore_entry={self._zscore_entry}, zscore_exit={self._zscore_exit}"
        )

    def test_cointegration_rolling(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        hedge_ratio: float,
    ) -> tuple[bool, str]:
        """
        Test cointegration stability with rolling windows and regime change detection (ISSUE_002).

        Cointegration tested once is not enough - correlations can spike during crises.
        This method detects regime changes by:
        1. Comparing long-term (60d) vs short-term (10d) ADF test results
        2. Detecting correlation spikes (> 0.95) which indicate crisis/regime change

        Args:
            prices_a: Price series for asset A
            prices_b: Price series for asset B
            hedge_ratio: Estimated hedge ratio

        Returns:
            (is_stable, rationale): Tuple of stability flag and explanation
        """
        bars_per_day = self._bars_per_day
        min_bars_60d = 60 * bars_per_day
        min_bars_10d = 10 * bars_per_day
        min_bars_5d = 5 * bars_per_day

        # Calculate spread
        spread = prices_a - hedge_ratio * prices_b

        # Test 1: Rolling ADF comparison (60d vs 10d)
        if len(spread) >= min_bars_60d:
            _, pvalue_60d, _ = self._adf_test(spread[-min_bars_60d:])
            _, pvalue_10d, _ = self._adf_test(spread[-min_bars_10d:])

            # If short-term cointegration breaks down while long-term holds,
            # this indicates a potential regime change
            if pvalue_60d < 0.05 and pvalue_10d > 0.10:
                return (False, "REGIME_CHANGE: Short-term cointegration breakdown")

        # Test 2: Correlation spike detection (crisis indicator)
        if len(prices_a) >= min_bars_5d and len(prices_b) >= min_bars_5d:
            recent_corr = np.corrcoef(prices_a[-min_bars_5d:], prices_b[-min_bars_5d:])[0, 1]

            # Correlation spikes (near +1 or -1) indicate crisis/regime change
            # In normal markets, correlated assets have stable correlation
            # During crises, correlations spike as everything moves together
            if abs(recent_corr) > self._correlation_spike_threshold:
                return (False, f"REGIME_CHANGE: Correlation spike detected ({recent_corr:.3f})")

        # Test 3: Check for structural break in spread mean
        if len(spread) >= min_bars_60d:
            mean_60d = np.mean(spread[-min_bars_60d:])
            mean_10d = np.mean(spread[-min_bars_10d:])
            std_60d = np.std(spread[-min_bars_60d:])

            if std_60d > 1e-12:
                mean_shift_zscore = abs(mean_10d - mean_60d) / std_60d
                if mean_shift_zscore > 3.0:  # Mean shifted by more than 3 sigma
                    return (False, f"REGIME_CHANGE: Spread mean shifted {mean_shift_zscore:.1f} sigma")

        return (True, "STABLE: Rolling cointegration tests passed")

    def calculate_rolling_correlation(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        window: int | None = None
    ) -> tuple[float, float]:
        """
        P2: Calculate rolling correlation and stability.

        Monitors pair correlation over time to detect regime changes
        and assess relationship stability.

        Args:
            prices_a: Price series for asset A
            prices_b: Price series for asset B
            window: Rolling window size (defaults to _rolling_corr_window)

        Returns:
            (rolling_correlation, stability_score)
            - rolling_correlation: Current short-term correlation
            - stability_score: 0-1 where 1 is very stable
        """
        if window is None:
            window = self._rolling_corr_window

        if len(prices_a) < window * 2 or len(prices_b) < window * 2:
            return 0.0, 0.0

        # Calculate rolling correlations
        correlations = []
        for i in range(window, len(prices_a)):
            corr = np.corrcoef(
                prices_a[i - window:i],
                prices_b[i - window:i]
            )[0, 1]
            if np.isfinite(corr):
                correlations.append(corr)

        if len(correlations) < 5:
            return 0.0, 0.0

        # Current rolling correlation
        rolling_corr = correlations[-1]

        # Stability score based on correlation standard deviation
        corr_std = np.std(correlations)

        # Lower std = more stable = higher score
        # Normalize: std of 0 = score 1, std >= threshold = score 0
        stability = max(0.0, 1.0 - (corr_std / self._corr_stability_threshold))

        return rolling_corr, stability

    def estimate_reversion_speed(
        self,
        spread: np.ndarray,
        window: int | None = None
    ) -> tuple[float, float]:
        """
        P2: Estimate mean-reversion speed and expected time to mean.

        Uses Ornstein-Uhlenbeck process parameters to estimate
        how quickly the spread reverts to its mean.

        Args:
            spread: Spread time series
            window: Estimation window (defaults to _speed_estimation_window)

        Returns:
            (reversion_speed, expected_time_to_mean)
            - reversion_speed: Daily reversion rate (0-1)
            - expected_time_to_mean: Expected bars to reach mean
        """
        if window is None:
            window = self._speed_estimation_window

        if len(spread) < window + 1:
            return 0.0, float("inf")

        recent_spread = spread[-window - 1:]

        # Calculate spread changes
        spread_lag = recent_spread[:-1]
        spread_diff = np.diff(recent_spread)

        # Estimate theta (mean reversion rate) via OLS
        # dS = theta * (mu - S) * dt + noise
        # Regress dS on (S_lag - mean)
        mean_spread = np.mean(recent_spread)
        deviation = spread_lag - mean_spread

        var_deviation = np.var(deviation)
        if var_deviation < 1e-12:
            return 0.0, float("inf")

        cov_result = np.cov(spread_diff, deviation)
        if cov_result.ndim == 0:
            return 0.0, float("inf")

        # theta = -cov(dS, S-mu) / var(S-mu)
        theta = -cov_result[0, 1] / var_deviation

        if not np.isfinite(theta) or theta <= 0:
            return 0.0, float("inf")

        # Reversion speed: portion of deviation corrected per period
        reversion_speed = min(1.0, theta)

        # Expected time to mean (half-life based)
        # Expected time = |current deviation| / (theta * std(spread))
        current_deviation = abs(spread[-1] - mean_spread)
        std_spread = np.std(recent_spread)

        if std_spread > 1e-12 and theta > 0:
            # Expected time proportional to deviation in std units / theta
            expected_time = (current_deviation / std_spread) / theta
        else:
            expected_time = float("inf")

        return reversion_speed, expected_time

    def calculate_entry_timing(
        self,
        spread: np.ndarray,
        current_zscore: float,
        half_life: float
    ) -> tuple[float, bool]:
        """
        P2: Optimize entry timing based on spread dynamics.

        Determines if current moment is optimal for entry based on:
        1. Z-score level (not too extreme, not too mild)
        2. Z-score momentum (ideally decelerating)
        3. Half-life (faster reversion = better)

        Args:
            spread: Spread time series
            current_zscore: Current z-score
            half_life: Estimated half-life

        Returns:
            (entry_score, is_optimal)
            - entry_score: 0-1 score for entry quality
            - is_optimal: True if timing conditions are met
        """
        if len(spread) < 5:
            return 0.0, False

        abs_zscore = abs(current_zscore)

        # Factor 1: Z-score level (0-1 score)
        # Optimal range is typically 1.8-2.5 sigma
        min_entry, max_entry = self._optimal_entry_zscore_range

        if abs_zscore < min_entry:
            # Too close to mean - not enough edge
            zscore_score = 0.0
        elif abs_zscore > max_entry + 1.0:
            # Very extreme - might not revert (regime change risk)
            zscore_score = max(0.0, 1.0 - (abs_zscore - max_entry - 1.0) / 2.0)
        elif min_entry <= abs_zscore <= max_entry:
            # Optimal range
            zscore_score = 1.0
        else:
            # Between max_entry and max_entry + 1.0
            zscore_score = 0.8

        # Factor 2: Z-score momentum (looking for deceleration)
        zscore_series = []
        std_spread = np.std(spread)
        mean_spread = np.mean(spread)

        if std_spread > 1e-12:
            for i in range(-5, 0):
                if i + len(spread) >= 0:
                    z = (spread[i] - mean_spread) / std_spread
                    zscore_series.append(z)

        momentum_score = 0.5  # Default neutral
        if len(zscore_series) >= 3:
            # Calculate acceleration (second derivative)
            velocity = np.diff(zscore_series)
            if len(velocity) >= 2:
                acceleration = velocity[-1] - velocity[-2]

                # For entry, we want deceleration (acceleration opposite to velocity)
                if abs(velocity[-1]) > 0.01:
                    if np.sign(acceleration) != np.sign(velocity[-1]):
                        # Decelerating - good for entry
                        momentum_score = min(1.0, 0.7 + abs(acceleration) / self._zscore_acceleration_threshold * 0.3)
                    else:
                        # Accelerating away from mean - risky
                        momentum_score = max(0.0, 0.5 - abs(acceleration) / self._zscore_acceleration_threshold * 0.3)

        # Factor 3: Half-life quality (faster = better within bounds)
        if self._min_half_life <= half_life <= self._max_half_life:
            # Normalize: shorter half-life = higher score
            half_life_range = self._max_half_life - self._min_half_life
            half_life_score = 1.0 - (half_life - self._min_half_life) / half_life_range
        else:
            half_life_score = 0.0

        # Combined entry score (weighted average)
        entry_score = (
            zscore_score * 0.5 +
            momentum_score * 0.3 +
            half_life_score * 0.2
        )

        # Optimal entry: score > 0.6 and all factors reasonable
        is_optimal = (
            entry_score > 0.6 and
            zscore_score > 0.5 and
            momentum_score > 0.4 and
            half_life_score > 0.3
        )

        return entry_score, is_optimal

    def johansen_cointegration_test(
        self,
        price_series: list[np.ndarray],
        det_order: int = 0,
        k_ar_diff: int = 1
    ) -> dict[str, Any]:
        """
        Johansen cointegration test for multiple time series.

        The Johansen test is superior to Engle-Granger for pairs trading because:
        1. Can test multiple series simultaneously
        2. Identifies all cointegrating vectors
        3. Provides both trace and max-eigenvalue statistics
        4. More powerful for detecting cointegration

        Implementation based on Johansen (1991) likelihood ratio tests.

        Args:
            price_series: List of price arrays (all same length)
            det_order: Deterministic trend order (-1=no const, 0=const, 1=trend)
            k_ar_diff: Number of lagged differences in VECM

        Returns:
            Dictionary with test results:
            - n_cointegrating: Number of cointegrating relationships (0 to n-1)
            - trace_stats: Trace test statistics
            - max_eigen_stats: Max eigenvalue test statistics
            - critical_values_trace: 95% critical values for trace test
            - critical_values_max: 95% critical values for max eigenvalue test
            - eigenvectors: Cointegrating vectors (if any)
            - is_cointegrated: Boolean (at least one cointegrating relation)
        """
        n_series = len(price_series)
        if n_series < 2:
            return {
                "n_cointegrating": 0,
                "is_cointegrated": False,
                "error": "Need at least 2 series"
            }

        # Ensure all series have same length
        min_len = min(len(s) for s in price_series)
        if min_len < 50:
            return {
                "n_cointegrating": 0,
                "is_cointegrated": False,
                "error": "Insufficient data (need at least 50 observations)"
            }

        # Stack price series into matrix: T x n
        Y = np.column_stack([s[-min_len:] for s in price_series])
        T, n = Y.shape

        # Calculate first differences
        dY = np.diff(Y, axis=0)  # (T-1) x n

        # Build lagged differences matrix for VECM
        # dY_t = Pi * Y_{t-1} + sum(Gamma_i * dY_{t-i}) + constant
        k = k_ar_diff
        if T - k - 1 < n + 5:
            k = max(1, T - n - 6)

        # Dependent variable: dY[k:]
        T_eff = T - k - 1
        Z0 = dY[k:]  # (T_eff) x n

        # Lagged levels: Y[k:-1]
        Z1 = Y[k:-1]  # (T_eff) x n

        # Lagged differences for VECM
        Z2_list = []
        for i in range(1, k + 1):
            Z2_list.append(dY[k-i:-i if i < k else None or len(dY)])

        # Add constant if det_order >= 0
        if det_order >= 0:
            Z2_list.append(np.ones((T_eff, 1)))

        if Z2_list:
            Z2 = np.column_stack(Z2_list) if len(Z2_list) > 1 else Z2_list[0]
            if Z2.ndim == 1:
                Z2 = Z2.reshape(-1, 1)

            # Residuals from regressing Z0, Z1 on Z2
            try:
                Z2_inv = np.linalg.lstsq(Z2, np.eye(Z2.shape[0]), rcond=None)[0].T
                P = np.eye(T_eff) - Z2 @ np.linalg.lstsq(Z2, np.eye(T_eff), rcond=None)[0]
            except np.linalg.LinAlgError:
                P = np.eye(T_eff)

            R0 = P @ Z0
            R1 = P @ Z1
        else:
            R0 = Z0
            R1 = Z1

        # Calculate moment matrices
        S00 = (R0.T @ R0) / T_eff
        S01 = (R0.T @ R1) / T_eff
        S10 = (R1.T @ R0) / T_eff
        S11 = (R1.T @ R1) / T_eff

        # Solve eigenvalue problem: |lambda * S11 - S10 * S00^{-1} * S01| = 0
        try:
            S00_inv = np.linalg.inv(S00)
            S11_inv = np.linalg.inv(S11)
        except np.linalg.LinAlgError:
            return {
                "n_cointegrating": 0,
                "is_cointegrated": False,
                "error": "Singular matrix in eigenvalue problem"
            }

        # Matrix for eigenvalue problem
        M = S11_inv @ S10 @ S00_inv @ S01

        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(M)

        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.real(eigenvalues[idx])
        eigenvectors = np.real(eigenvectors[:, idx])

        # Ensure eigenvalues are in [0, 1) for log calculation
        eigenvalues = np.clip(eigenvalues, 1e-10, 1 - 1e-10)

        # Calculate trace and max-eigenvalue statistics
        trace_stats = []
        max_eigen_stats = []

        for r in range(n):
            # Trace statistic: -T * sum(log(1 - lambda_i)) for i = r+1 to n
            trace_stat = -T_eff * np.sum(np.log(1 - eigenvalues[r:]))
            trace_stats.append(trace_stat)

            # Max eigenvalue statistic: -T * log(1 - lambda_{r+1})
            if r < n:
                max_stat = -T_eff * np.log(1 - eigenvalues[r])
                max_eigen_stats.append(max_stat)

        # Critical values (95% level, from Osterwald-Lenum 1992)
        # These are approximate - for production, use tabulated values
        # Format: critical_values[n_series][r] where r is null hypothesis rank
        trace_cv_table = {
            2: [15.41, 3.76],
            3: [29.68, 15.41, 3.76],
            4: [47.21, 29.68, 15.41, 3.76],
            5: [68.52, 47.21, 29.68, 15.41, 3.76],
        }
        max_cv_table = {
            2: [14.07, 3.76],
            3: [21.12, 14.07, 3.76],
            4: [27.42, 21.12, 14.07, 3.76],
            5: [33.46, 27.42, 21.12, 14.07, 3.76],
        }

        # Get critical values for this number of series
        if n in trace_cv_table:
            cv_trace = trace_cv_table[n]
            cv_max = max_cv_table[n]
        else:
            # Approximate for larger systems
            cv_trace = [15.41 + 10 * (i + 1) for i in range(n)]
            cv_max = [14.07 + 7 * (i + 1) for i in range(n)]

        # Determine number of cointegrating relationships
        # Using trace test: reject H0(r) if trace_stat > critical_value
        n_coint_trace = 0
        for r in range(n):
            if r < len(trace_stats) and r < len(cv_trace):
                if trace_stats[r] > cv_trace[r]:
                    n_coint_trace = r + 1
                else:
                    break

        # Using max eigenvalue test
        n_coint_max = 0
        for r in range(n):
            if r < len(max_eigen_stats) and r < len(cv_max):
                if max_eigen_stats[r] > cv_max[r]:
                    n_coint_max = r + 1
                else:
                    break

        # Conservative: use minimum of trace and max tests
        n_cointegrating = min(n_coint_trace, n_coint_max)

        # Extract cointegrating vectors (first n_cointegrating columns)
        if n_cointegrating > 0:
            coint_vectors = eigenvectors[:, :n_cointegrating]
            # Normalize so first element is 1
            for i in range(n_cointegrating):
                if abs(coint_vectors[0, i]) > 1e-10:
                    coint_vectors[:, i] = coint_vectors[:, i] / coint_vectors[0, i]
        else:
            coint_vectors = None

        return {
            "n_cointegrating": n_cointegrating,
            "is_cointegrated": n_cointegrating > 0,
            "trace_stats": trace_stats,
            "max_eigen_stats": max_eigen_stats,
            "critical_values_trace": cv_trace[:n],
            "critical_values_max": cv_max[:n],
            "eigenvalues": eigenvalues.tolist(),
            "eigenvectors": coint_vectors.tolist() if coint_vectors is not None else None,
            "n_series": n,
            "n_observations": T_eff,
            "lags_used": k,
        }

    def test_cointegration_johansen(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> PairAnalysis:
        """
        Test for cointegration using Johansen test (more robust than Engle-Granger).

        This is an enhanced version of test_cointegration that uses the
        Johansen likelihood ratio test instead of the simplified ADF.

        Args:
            prices_a: Price series for asset A
            prices_b: Price series for asset B

        Returns:
            PairAnalysis with cointegration test results
        """
        if len(prices_a) != len(prices_b) or len(prices_a) < 50:
            return PairAnalysis(
                symbol_a="",
                symbol_b="",
                correlation=0,
                cointegration_pvalue=1.0,
                hedge_ratio=1.0,
                half_life=float("inf"),
                current_zscore=0,
                is_cointegrated=False,
                rationale="REJECTED: Insufficient data for Johansen test (need 50+ observations)",
            )

        # Run Johansen test
        johansen_result = self.johansen_cointegration_test([prices_a, prices_b])

        # Calculate correlation
        correlation = np.corrcoef(prices_a, prices_b)[0, 1]

        # If cointegrated, extract hedge ratio from eigenvector
        if johansen_result["is_cointegrated"] and johansen_result["eigenvectors"]:
            # Cointegrating vector [1, -beta] means spread = A - beta*B
            coint_vec = johansen_result["eigenvectors"]
            if len(coint_vec) > 0 and len(coint_vec[0]) >= 2:
                hedge_ratio = -coint_vec[1][0] if isinstance(coint_vec[0], list) else -coint_vec[1]
            else:
                hedge_ratio = self._estimate_hedge_ratio(prices_a, prices_b)
        else:
            hedge_ratio = self._estimate_hedge_ratio(prices_a, prices_b)

        # Calculate spread and statistics
        spread = prices_a - hedge_ratio * prices_b
        spread_std = np.std(spread)
        if spread_std < 1e-12:
            zscore = 0.0
        else:
            zscore = (spread[-1] - np.mean(spread)) / spread_std

        # Estimate half-life
        half_life = self._estimate_half_life(spread)

        # Build rationale
        is_cointegrated = johansen_result["is_cointegrated"]
        if is_cointegrated:
            trace_stat = johansen_result["trace_stats"][0] if johansen_result["trace_stats"] else 0
            cv = johansen_result["critical_values_trace"][0] if johansen_result["critical_values_trace"] else 0
            rationale = f"JOHANSEN: Cointegrated (trace={trace_stat:.2f} > cv={cv:.2f}), half_life={half_life:.1f}"
        else:
            rationale = "JOHANSEN: Not cointegrated at 95% level"

        # Additional stability checks
        is_stable, stability_rationale = self.test_cointegration_rolling(prices_a, prices_b, hedge_ratio)
        if not is_stable:
            is_cointegrated = False
            rationale = stability_rationale

        # Check half-life bounds
        if is_cointegrated:
            if not (self._min_half_life <= half_life <= self._max_half_life):
                is_cointegrated = False
                rationale = f"REJECTED: Half-life {half_life:.1f} outside bounds [{self._min_half_life}, {self._max_half_life}]"

        # P2: Calculate rolling correlation and stability
        rolling_corr, corr_stability = self.calculate_rolling_correlation(prices_a, prices_b)

        # P2: Estimate mean-reversion speed
        reversion_speed, expected_time = self.estimate_reversion_speed(spread)

        # P2: Calculate entry timing
        entry_score, optimal_entry = self.calculate_entry_timing(spread, zscore, half_life)

        return PairAnalysis(
            symbol_a="",
            symbol_b="",
            correlation=correlation,
            cointegration_pvalue=0.05 if is_cointegrated else 0.10,  # Approximate from Johansen
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            current_zscore=zscore,
            is_cointegrated=is_cointegrated,
            rationale=rationale,
            rolling_correlation=rolling_corr,
            correlation_stability=corr_stability,
            reversion_speed=reversion_speed,
            expected_time_to_mean=expected_time,
            entry_score=entry_score,
            optimal_entry=optimal_entry,
        )

    def test_cointegration(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> PairAnalysis:
        """
        Test for cointegration between two price series.

        Includes rolling cointegration test and regime change detection (ISSUE_002).

        TODO: Implement proper cointegration tests:
        - Augmented Dickey-Fuller on residuals
        - Johansen test for multiple series
        """
        if len(prices_a) != len(prices_b) or len(prices_a) < 30:
            return PairAnalysis(
                symbol_a="",
                symbol_b="",
                correlation=0,
                cointegration_pvalue=1.0,
                hedge_ratio=1.0,
                half_life=float("inf"),
                current_zscore=0,
                is_cointegrated=False,
                rationale="REJECTED: Insufficient data",
            )

        # Calculate correlation
        correlation = np.corrcoef(prices_a, prices_b)[0, 1]

        # Estimate hedge ratio via OLS
        hedge_ratio = self._estimate_hedge_ratio(prices_a, prices_b)

        # Calculate spread
        spread = prices_a - hedge_ratio * prices_b

        # Test stationarity of spread (simplified ADF)
        # TODO: Use proper statsmodels ADF test
        pvalue = self._simplified_adf_test(spread)

        # Estimate half-life
        half_life = self._estimate_half_life(spread)

        # Calculate current z-score
        # Guard against division by zero (std=0 means constant spread)
        spread_std = np.std(spread)
        if spread_std < 1e-12:
            zscore = 0.0
        else:
            zscore = (spread[-1] - np.mean(spread)) / spread_std

        # Basic cointegration check
        basic_cointegrated = (
            pvalue < 0.05
            and self._min_half_life <= half_life <= self._max_half_life
        )

        # Rolling cointegration and regime change detection (ISSUE_002)
        is_stable, rationale = self.test_cointegration_rolling(prices_a, prices_b, hedge_ratio)

        # Final cointegration decision: must pass both basic test AND stability test
        is_cointegrated = basic_cointegrated and is_stable

        if not basic_cointegrated:
            rationale = f"REJECTED: ADF p-value={pvalue:.3f}, half_life={half_life:.1f}"

        # P2: Calculate rolling correlation and stability
        rolling_corr, corr_stability = self.calculate_rolling_correlation(prices_a, prices_b)

        # P2: Estimate mean-reversion speed
        reversion_speed, expected_time = self.estimate_reversion_speed(spread)

        # P2: Calculate entry timing
        entry_score, optimal_entry = self.calculate_entry_timing(spread, zscore, half_life)

        return PairAnalysis(
            symbol_a="",
            symbol_b="",
            correlation=correlation,
            cointegration_pvalue=pvalue,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            current_zscore=zscore,
            is_cointegrated=is_cointegrated,
            rationale=rationale,
            rolling_correlation=rolling_corr,
            correlation_stability=corr_stability,
            reversion_speed=reversion_speed,
            expected_time_to_mean=expected_time,
            entry_score=entry_score,
            optimal_entry=optimal_entry,
        )

    def _estimate_hedge_ratio(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> float:
        """
        Estimate hedge ratio using OLS (static method).

        For dynamic hedge ratio that adapts over time, use
        estimate_hedge_ratio_kalman() instead.
        """
        # OLS: minimize sum((a - beta*b)^2)
        # Guard against zero variance in prices_b
        var_b = np.var(prices_b)
        if var_b < 1e-12:
            logger.warning("Zero variance in prices_b, returning hedge ratio of 1.0")
            return 1.0

        cov_matrix = np.cov(prices_a, prices_b)
        # Handle the case when cov_matrix is a scalar (single data point)
        if cov_matrix.ndim == 0:
            return 1.0

        beta = cov_matrix[0, 1] / var_b
        # Ensure beta is finite
        if not np.isfinite(beta):
            logger.warning(f"Non-finite hedge ratio computed: {beta}, returning 1.0")
            return 1.0
        return beta

    # =========================================================================
    # Phase 5.1: Kalman Filter for Dynamic Hedge Ratio
    # =========================================================================

    def get_or_create_kalman_filter(
        self,
        symbol_a: str,
        symbol_b: str,
    ) -> "KalmanHedgeRatio | None":
        """
        Get or create Kalman filter for a pair.

        The filter is cached per pair to maintain state across updates.

        Args:
            symbol_a: First symbol (dependent variable)
            symbol_b: Second symbol (independent variable)

        Returns:
            KalmanHedgeRatio instance or None if not available
        """
        if not HAS_KALMAN:
            logger.warning("Kalman filter not available, using OLS")
            return None

        pair_key = (symbol_a, symbol_b)
        if pair_key not in self._kalman_filters:
            self._kalman_filters[pair_key] = create_kalman_filter({
                "delta": self._kalman_delta,
                "ve": self._kalman_ve,
                "warmup_period": self._kalman_warmup,
                "include_intercept": self._kalman_include_intercept,
            })
            logger.debug(f"Created Kalman filter for pair {symbol_a}:{symbol_b}")

        return self._kalman_filters[pair_key]

    def estimate_hedge_ratio_kalman(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> dict[str, Any]:
        """
        Estimate hedge ratio using Kalman filter (Phase 5.1).

        Unlike OLS which gives a static estimate, the Kalman filter provides:
        - Time-varying hedge ratio that adapts to regime changes
        - Confidence intervals for the estimate
        - Stability metrics to assess relationship quality

        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            prices_a: Price series for asset A
            prices_b: Price series for asset B

        Returns:
            Dictionary with:
            - hedge_ratio: Current dynamic hedge ratio
            - intercept: Intercept term (if enabled)
            - confidence_interval: (lower, upper) bounds
            - stability_score: 0-1 stability metric
            - spread_zscore: Current z-score of spread
            - is_warmed_up: Whether filter has sufficient history
            - comparison_vs_ols: Difference from OLS estimate
        """
        kf = self.get_or_create_kalman_filter(symbol_a, symbol_b)

        if kf is None:
            # Fallback to OLS
            ols_beta = self._estimate_hedge_ratio(prices_a, prices_b)
            return {
                "hedge_ratio": ols_beta,
                "intercept": 0.0,
                "confidence_interval": (ols_beta * 0.9, ols_beta * 1.1),
                "stability_score": 0.5,
                "spread_zscore": 0.0,
                "is_warmed_up": True,
                "comparison_vs_ols": 0.0,
                "method": "ols_fallback",
            }

        # Process the full series through the Kalman filter
        results = kf.process_series(prices_a, prices_b)

        if not results:
            ols_beta = self._estimate_hedge_ratio(prices_a, prices_b)
            return {
                "hedge_ratio": ols_beta,
                "intercept": 0.0,
                "confidence_interval": (ols_beta * 0.9, ols_beta * 1.1),
                "stability_score": 0.5,
                "spread_zscore": 0.0,
                "is_warmed_up": False,
                "comparison_vs_ols": 0.0,
                "method": "ols_fallback",
            }

        # Get final result
        final_result = results[-1]

        # Get confidence interval
        beta, lower, upper = kf.get_hedge_ratio_with_confidence(confidence=0.95)

        # Get current z-score
        zscore = kf.get_zscore(prices_a[-1], prices_b[-1])

        # Compare with OLS
        ols_beta = self._estimate_hedge_ratio(prices_a, prices_b)

        return {
            "hedge_ratio": final_result.hedge_ratio,
            "intercept": final_result.intercept,
            "confidence_interval": (lower, upper),
            "stability_score": final_result.stability_score,
            "spread_zscore": zscore,
            "is_warmed_up": final_result.is_stable,
            "comparison_vs_ols": final_result.hedge_ratio - ols_beta,
            "n_observations": final_result.n_observations,
            "method": "kalman",
        }

    def estimate_hedge_ratio_adaptive(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        method: HedgeRatioMethod | None = None,
    ) -> dict[str, Any]:
        """
        Estimate hedge ratio using configured method (Phase 5.1).

        This is the main entry point that selects the appropriate method
        based on configuration or explicit override.

        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            prices_a: Price series for asset A
            prices_b: Price series for asset B
            method: Override default method

        Returns:
            Dictionary with hedge ratio and metadata
        """
        if method is None:
            method = self._hedge_ratio_method

        if method == HedgeRatioMethod.KALMAN and HAS_KALMAN:
            return self.estimate_hedge_ratio_kalman(
                symbol_a, symbol_b, prices_a, prices_b
            )

        elif method == HedgeRatioMethod.ROLLING_OLS:
            # Rolling OLS with configurable window
            window = min(self._lookback, len(prices_a))
            beta = self._estimate_hedge_ratio(
                prices_a[-window:], prices_b[-window:]
            )
            return {
                "hedge_ratio": beta,
                "intercept": 0.0,
                "confidence_interval": (beta * 0.95, beta * 1.05),
                "stability_score": 0.7,
                "spread_zscore": 0.0,
                "is_warmed_up": True,
                "comparison_vs_ols": 0.0,
                "method": "rolling_ols",
                "window": window,
            }

        else:
            # Standard OLS
            beta = self._estimate_hedge_ratio(prices_a, prices_b)
            return {
                "hedge_ratio": beta,
                "intercept": 0.0,
                "confidence_interval": (beta * 0.95, beta * 1.05),
                "stability_score": 0.5,
                "spread_zscore": 0.0,
                "is_warmed_up": True,
                "comparison_vs_ols": 0.0,
                "method": "ols",
            }

    def update_kalman_hedge_ratio(
        self,
        symbol_a: str,
        symbol_b: str,
        price_a: float,
        price_b: float,
    ) -> dict[str, Any] | None:
        """
        Update Kalman filter with new price observation (Phase 5.1).

        Call this method with each new price tick/bar to update
        the dynamic hedge ratio estimate.

        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            price_a: Current price of asset A
            price_b: Current price of asset B

        Returns:
            Updated hedge ratio info or None if not using Kalman
        """
        kf = self.get_or_create_kalman_filter(symbol_a, symbol_b)
        if kf is None:
            return None

        result = kf.update(price_a, price_b)

        return {
            "hedge_ratio": result.hedge_ratio,
            "spread": result.spread,
            "spread_zscore": result.zscore,
            "is_stable": result.is_stable,
            "n_observations": result.n_observations,
        }

    def reset_kalman_filter(
        self,
        symbol_a: str,
        symbol_b: str,
    ) -> None:
        """
        Reset Kalman filter for a pair.

        Call this after a regime change or when relationship breaks down.
        """
        pair_key = (symbol_a, symbol_b)
        if pair_key in self._kalman_filters:
            self._kalman_filters[pair_key].reset()
            logger.info(f"Reset Kalman filter for pair {symbol_a}:{symbol_b}")

    def get_kalman_comparison(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        ols_window: int = 60,
    ) -> dict[str, Any]:
        """
        Compare Kalman filter vs OLS hedge ratios (Phase 5.1).

        Useful for backtesting and validation.

        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            prices_a: Price series for asset A
            prices_b: Price series for asset B
            ols_window: Window for rolling OLS comparison

        Returns:
            Dictionary with comparison metrics
        """
        kf = self.get_or_create_kalman_filter(symbol_a, symbol_b)
        if kf is None:
            return {"error": "Kalman filter not available"}

        # Reset and process full series
        kf.reset()
        comparison = kf.compare_with_ols(prices_a, prices_b, ols_window)

        return {
            "pair": f"{symbol_a}:{symbol_b}",
            "kalman_beta_mean": comparison.get("kalman_beta_mean", 0),
            "ols_beta_mean": comparison.get("ols_beta_mean", 0),
            "kalman_beta_std": comparison.get("kalman_beta_std", 0),
            "ols_beta_std": comparison.get("ols_beta_std", 0),
            "stability_improvement_pct": comparison.get("stability_improvement_pct", 0),
            "recommendation": (
                "KALMAN" if comparison.get("stability_improvement_pct", 0) > 10
                else "OLS"
            ),
        }

    def _select_optimal_lag(
        self,
        series: np.ndarray,
        max_lags: int | None = None,
        method: str = "aic"
    ) -> int:
        """
        Select optimal lag for ADF test using information criteria (#Q3).

        Methods:
        - 'aic': Akaike Information Criterion (default)
        - 'bic': Bayesian Information Criterion
        - 'schwert': Schwert's rule: int(12 * (n/100)^(1/4))

        Args:
            series: Time series data
            max_lags: Maximum lags to consider (default: Schwert's rule)
            method: Selection method

        Returns:
            Optimal number of lags
        """
        n = len(series)

        # Default max_lags using Schwert's rule
        if max_lags is None:
            max_lags = int(12 * (n / 100) ** 0.25)

        max_lags = min(max_lags, n // 5)  # Don't use more than n/5 lags

        if method == "schwert":
            return max_lags

        # Calculate AIC/BIC for each lag count
        diff = np.diff(series)
        lagged = series[:-1]

        best_ic = float('inf')
        best_lag = 1

        for k in range(0, max_lags + 1):
            try:
                # Build design matrix
                if k == 0:
                    y = diff
                    X = np.column_stack([lagged, np.ones(len(diff))])
                else:
                    y = diff[k:]
                    X_cols = [lagged[k:]]

                    for lag in range(1, k + 1):
                        start = k - lag
                        end = -lag if lag < k else None
                        if end is None:
                            X_cols.append(diff[start:len(y) + start])
                        else:
                            X_cols.append(diff[start:end])

                    X_cols.append(np.ones(len(y)))
                    X = np.column_stack(X_cols)

                if len(y) < len(X[0]) + 2:
                    continue

                # OLS
                XtX = X.T @ X
                Xty = X.T @ y
                beta = np.linalg.solve(XtX, Xty)

                # Residuals
                residuals = y - X @ beta
                sse = np.sum(residuals ** 2)
                n_obs = len(y)
                n_params = len(beta)

                # Information criterion
                if sse > 0:
                    log_likelihood = -n_obs / 2 * (1 + np.log(2 * np.pi) + np.log(sse / n_obs))

                    if method == "aic":
                        ic = -2 * log_likelihood + 2 * n_params
                    else:  # bic
                        ic = -2 * log_likelihood + n_params * np.log(n_obs)

                    if ic < best_ic:
                        best_ic = ic
                        best_lag = k

            except (np.linalg.LinAlgError, ValueError):
                continue

        return max(1, best_lag)

    def _adf_test(
        self,
        series: np.ndarray,
        max_lags: int | None = None,
        lag_method: str = "aic"
    ) -> tuple[float, float, int]:
        """
        Augmented Dickey-Fuller test for stationarity with optimal lag selection (#Q3).

        Tests H0: series has a unit root (non-stationary)
        vs H1: series is stationary (mean-reverting)

        Args:
            series: Time series data
            max_lags: Maximum lags to consider (None = auto using Schwert's rule)
            lag_method: Lag selection method ('aic', 'bic', 'schwert')

        Returns:
            (test_statistic, p_value, selected_lags)
        """
        n = len(series)
        if n < 20:
            return (0.0, 1.0, 0)

        # Select optimal lag count (#Q3)
        lags = self._select_optimal_lag(series, max_lags, lag_method)

        # Calculate first difference
        diff = np.diff(series)
        lagged = series[:-1]

        # Ensure lags is reasonable
        lags = min(lags, n // 5, len(diff) - 5)
        lags = max(0, lags)

        # Build design matrix: [lagged_level, lagged_diffs, constant]
        if lags == 0:
            y = diff
            X_cols = [lagged]
        else:
            y = diff[lags:]
            X_cols = [lagged[lags:]]

            # Add lagged differences
            for lag in range(1, lags + 1):
                start = lags - lag
                end = -lag if lag < lags else None
                if end is None:
                    X_cols.append(diff[start:len(y) + start])
                else:
                    X_cols.append(diff[start:end])

        # Add constant
        X_cols.append(np.ones(len(y)))

        # Stack into design matrix
        X = np.column_stack(X_cols)

        # OLS regression
        try:
            # Solve normal equations: beta = (X'X)^-1 X'y
            XtX = X.T @ X
            Xty = X.T @ y
            beta = np.linalg.solve(XtX, Xty)

            # Test statistic for coefficient on lagged level (gamma)
            gamma = beta[0]

            # Calculate residuals and standard error
            residuals = y - X @ beta
            sigma2 = np.sum(residuals ** 2) / (len(y) - len(beta))
            var_beta = sigma2 * np.linalg.inv(XtX)
            se_gamma = np.sqrt(var_beta[0, 0])

            if se_gamma < 1e-10:
                return (0.0, 1.0, lags)

            # ADF test statistic
            adf_stat = gamma / se_gamma

            # Approximate p-value using MacKinnon critical values
            # Critical values depend on sample size and whether trend/constant included
            # Using values for constant only, n=100: -3.51 (1%), -2.89 (5%), -2.58 (10%)
            # Adjusted for sample size using interpolation
            n_eff = len(y)

            # MacKinnon asymptotic critical values (constant, no trend)
            # tau_c: -3.43 (1%), -2.86 (5%), -2.57 (10%)
            # Small-sample adjustments
            cv_1 = -3.43 - 6.5 / n_eff - 16.5 / (n_eff ** 2)
            cv_5 = -2.86 - 2.9 / n_eff - 4.3 / (n_eff ** 2)
            cv_10 = -2.57 - 1.5 / n_eff - 2.0 / (n_eff ** 2)

            if adf_stat < cv_1:
                p_value = 0.01
            elif adf_stat < cv_5:
                p_value = 0.01 + (adf_stat - cv_1) / (cv_5 - cv_1) * 0.04
            elif adf_stat < cv_10:
                p_value = 0.05 + (adf_stat - cv_5) / (cv_10 - cv_5) * 0.05
            elif adf_stat < -1.95:
                p_value = 0.10 + (adf_stat - cv_10) / (-1.95 - cv_10) * 0.15
            else:
                p_value = 0.25 + min(0.75, (adf_stat + 1.95) * 0.3)

            return (adf_stat, max(0.001, min(0.999, p_value)), lags)

        except (np.linalg.LinAlgError, ValueError):
            return (0.0, 1.0, lags)

    def _simplified_adf_test(self, series: np.ndarray) -> float:
        """
        Wrapper for backward compatibility.

        Returns p-value from ADF test with optimal lag selection (#Q3).
        """
        _, p_value, lags = self._adf_test(series)
        logger.debug(f"ADF test used {lags} lags (selected by AIC)")
        return p_value

    def _estimate_half_life(self, spread: np.ndarray) -> float:
        """
        Estimate mean reversion half-life.

        Uses Ornstein-Uhlenbeck process:
        dX = theta * (mu - X) * dt + sigma * dW

        Half-life = ln(2) / theta

        TODO: Implement proper OU estimation via MLE.
        """
        if len(spread) < 10:
            return float("inf")

        # Simplified: regress change on level
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)

        if np.std(spread_lag) < 1e-8:
            return float("inf")

        # theta estimate
        var_lag = np.var(spread_lag)
        if var_lag < 1e-12:
            return float("inf")

        cov_result = np.cov(spread_diff, spread_lag)
        # Handle scalar result (shouldn't happen but defensive)
        if cov_result.ndim == 0:
            return float("inf")

        theta = -cov_result[0, 1] / var_lag

        # Guard against non-positive theta or non-finite values
        if not np.isfinite(theta) or theta <= 0:
            return float("inf")

        half_life = np.log(2) / theta

        # Sanity check on half-life
        if not np.isfinite(half_life) or half_life < 0:
            return float("inf")

        return half_life

    def check_exit_conditions(
        self,
        current_zscore: float,
        entry_zscore: float,
        bars_held: int,
        position_direction: str,
        correlation_stability: float
    ) -> tuple[bool, str | None]:
        """
        RISK-005: Check if exit conditions are met for stat arb position.

        Args:
            current_zscore: Current spread z-score
            entry_zscore: Z-score at entry
            bars_held: Number of bars position has been held
            position_direction: "long_spread" or "short_spread"
            correlation_stability: Current correlation stability score

        Returns:
            (should_exit, reason)
        """
        # Check max holding period
        if bars_held >= self._max_holding_bars:
            return True, "max_holding_period_reached"

        # Check stop-loss (spread widening further)
        if position_direction == "long_spread" and current_zscore < -self._stop_loss_zscore:
            return True, "stop_loss_spread_widened"
        elif position_direction == "short_spread" and current_zscore > self._stop_loss_zscore:
            return True, "stop_loss_spread_widened"

        # Check take-profit (mean reversion achieved)
        if abs(current_zscore) < self._take_profit_zscore:
            return True, "take_profit_mean_reverted"

        # Check regime breakdown (correlation becoming unstable)
        if correlation_stability < self._min_correlation_stability:
            return True, "regime_breakdown_correlation_unstable"

        return False, None

    def generate_exit_signal(
        self,
        spread_name: str,
        spread_type: SpreadType,
        exit_reason: str,
        current_value: float,
        zscore: float,
        mean: float,
        std: float
    ) -> SpreadAnalysis:
        """
        RISK-005: Generate exit signal for spread position.

        Args:
            spread_name: Name of the spread
            spread_type: Type of spread
            exit_reason: Reason for exit
            current_value: Current spread value
            zscore: Current z-score
            mean: Spread mean
            std: Spread standard deviation

        Returns:
            SpreadAnalysis with exit signal
        """
        return SpreadAnalysis(
            spread_name=spread_name,
            spread_type=spread_type,
            current_value=current_value,
            zscore=zscore,
            percentile=50.0,  # N/A for exit
            mean=mean,
            std=std,
            half_life=0.0,
            is_tradeable=False,
            signal_direction="exit",
            signal_strength=1.0,  # High strength for exits
            stop_loss_zscore=self._stop_loss_zscore,
            take_profit_zscore=self._take_profit_zscore,
            stop_loss_pct=None,
            take_profit_pct=None,
            max_holding_bars=self._max_holding_bars,
            strategy_max_loss_pct=self._strategy_max_loss_pct,
            regime_suitable=True,
            regime_warning=None,
            is_exit_signal=True,
            exit_reason=exit_reason,
        )

    def calculate_spread_risk_levels(
        self,
        current_value: float,
        mean: float,
        std: float,
        direction: str
    ) -> tuple[float | None, float | None]:
        """
        Calculate stop-loss and take-profit in percentage terms.

        Args:
            current_value: Current spread value
            mean: Spread mean
            std: Spread standard deviation
            direction: "long_spread" or "short_spread"

        Returns:
            (stop_loss_pct, take_profit_pct) - percentage moves from current
        """
        if std <= 0:
            return None, None

        # For long spread: stop if spread goes more negative, profit if returns to mean
        # For short spread: stop if spread goes more positive, profit if returns to mean
        if direction == "long_spread":
            stop_value = mean - self._stop_loss_zscore * std
            tp_value = mean - self._take_profit_zscore * std
        elif direction == "short_spread":
            stop_value = mean + self._stop_loss_zscore * std
            tp_value = mean + self._take_profit_zscore * std
        else:
            return None, None

        if current_value != 0:
            stop_loss_pct = ((stop_value - current_value) / abs(current_value)) * 100
            take_profit_pct = ((tp_value - current_value) / abs(current_value)) * 100
        else:
            stop_loss_pct = None
            take_profit_pct = None

        return stop_loss_pct, take_profit_pct

    def generate_signal(
        self,
        analysis: PairAnalysis,
    ) -> dict[str, Any]:
        """
        Generate trading signal from pair analysis.

        Returns signal dict with direction and strength.
        Includes P2 enhancements for correlation monitoring and entry timing.
        """
        if not analysis.is_cointegrated:
            return {
                "direction": "flat",
                "strength": 0.0,
                "correlation_stability": analysis.correlation_stability,
                "entry_score": analysis.entry_score,
                "regime_suitable": False,
                "regime_warning": analysis.rationale,
            }

        zscore = analysis.current_zscore

        # P2: Adjust strength based on correlation stability and entry timing
        stability_factor = 0.7 + (analysis.correlation_stability * 0.3)  # 0.7-1.0
        timing_factor = 0.8 + (analysis.entry_score * 0.2)  # 0.8-1.0

        # RISK-004: Check regime suitability
        regime_suitable = analysis.correlation_stability >= self._min_correlation_stability
        regime_warning = None
        if not regime_suitable:
            regime_warning = f"Low correlation stability: {analysis.correlation_stability:.2f}"

        if zscore > self._zscore_entry:
            # Spread too high - short A, long B
            base_strength = min(1.0, zscore / 3.0)
            adjusted_strength = base_strength * stability_factor * timing_factor

            # Reduce strength if regime unsuitable
            if not regime_suitable:
                adjusted_strength *= 0.5

            return {
                "direction": "short_spread",
                "strength": adjusted_strength,
                "zscore": zscore,
                "rolling_correlation": analysis.rolling_correlation,
                "correlation_stability": analysis.correlation_stability,
                "reversion_speed": analysis.reversion_speed,
                "expected_time_to_mean": analysis.expected_time_to_mean,
                "entry_score": analysis.entry_score,
                "optimal_entry": analysis.optimal_entry,
                "stop_loss_zscore": self._stop_loss_zscore,
                "take_profit_zscore": self._take_profit_zscore,
                "max_holding_bars": self._max_holding_bars,
                "strategy_max_loss_pct": self._strategy_max_loss_pct,
                "regime_suitable": regime_suitable,
                "regime_warning": regime_warning,
            }
        elif zscore < -self._zscore_entry:
            # Spread too low - long A, short B
            base_strength = min(1.0, abs(zscore) / 3.0)
            adjusted_strength = base_strength * stability_factor * timing_factor

            # Reduce strength if regime unsuitable
            if not regime_suitable:
                adjusted_strength *= 0.5

            return {
                "direction": "long_spread",
                "strength": adjusted_strength,
                "zscore": zscore,
                "rolling_correlation": analysis.rolling_correlation,
                "correlation_stability": analysis.correlation_stability,
                "reversion_speed": analysis.reversion_speed,
                "expected_time_to_mean": analysis.expected_time_to_mean,
                "entry_score": analysis.entry_score,
                "optimal_entry": analysis.optimal_entry,
                "stop_loss_zscore": self._stop_loss_zscore,
                "take_profit_zscore": self._take_profit_zscore,
                "max_holding_bars": self._max_holding_bars,
                "strategy_max_loss_pct": self._strategy_max_loss_pct,
                "regime_suitable": regime_suitable,
                "regime_warning": regime_warning,
            }
        elif abs(zscore) < self._zscore_exit:
            # Near mean - exit (take profit achieved)
            return {
                "direction": "exit",
                "strength": 0.0,
                "zscore": zscore,
                "correlation_stability": analysis.correlation_stability,
                "entry_score": analysis.entry_score,
                "is_exit_signal": True,
                "exit_reason": "take_profit_mean_reverted",
                "regime_suitable": regime_suitable,
                "regime_warning": regime_warning,
            }
        else:
            # Hold current position
            return {
                "direction": "hold",
                "strength": abs(zscore) / self._zscore_entry,
                "zscore": zscore,
                "correlation_stability": analysis.correlation_stability,
                "expected_time_to_mean": analysis.expected_time_to_mean,
                "entry_score": analysis.entry_score,
                "regime_suitable": regime_suitable,
                "regime_warning": regime_warning,
            }

    # =========================================================================
    # COMMODITY SPREAD ANALYSIS
    # =========================================================================

    def analyze_spread(
        self,
        spread_name: str,
        prices: dict[str, np.ndarray],
        lookback: int | None = None
    ) -> SpreadAnalysis | None:
        """
        Analyze a commodity spread.

        Args:
            spread_name: Name of predefined spread
            prices: Dictionary of symbol to price arrays
            lookback: Lookback period (default: self._lookback)

        Returns:
            SpreadAnalysis or None if spread can't be computed
        """
        if spread_name not in COMMODITY_SPREADS:
            logger.warning(f"Unknown spread: {spread_name}")
            return None

        spread_def = COMMODITY_SPREADS[spread_name]
        lookback = lookback or self._lookback

        # Check if we have all required prices
        for symbol in spread_def.legs:
            if symbol not in prices or len(prices[symbol]) < lookback:
                logger.warning(f"Missing or insufficient price data for {symbol}")
                return None

        # Calculate spread value series
        spread_values = self._calculate_spread_value(spread_def, prices, lookback)

        if len(spread_values) < 20:
            return None

        # Calculate statistics
        mean_val = np.mean(spread_values)
        std_val = np.std(spread_values)
        current_val = spread_values[-1]

        if std_val < 1e-8:
            zscore = 0.0
        else:
            zscore = (current_val - mean_val) / std_val

        # Calculate percentile in historical range
        percentile = stats.percentileofscore(spread_values, current_val)

        # Estimate half-life
        half_life = self._estimate_half_life(spread_values)

        # Determine if tradeable
        is_tradeable = (
            abs(zscore) > self._zscore_entry
            and self._min_half_life <= half_life <= self._max_half_life
        )

        # Generate signal
        if zscore > self._zscore_entry:
            signal_direction = "short_spread"
            signal_strength = min(1.0, zscore / 3.0)
        elif zscore < -self._zscore_entry:
            signal_direction = "long_spread"
            signal_strength = min(1.0, abs(zscore) / 3.0)
        elif abs(zscore) < self._zscore_exit:
            signal_direction = "exit"
            signal_strength = 0.0
        else:
            signal_direction = "flat"
            signal_strength = 0.0

        # RISK-001: Calculate stop-loss and take-profit percentages
        stop_loss_pct, take_profit_pct = self.calculate_spread_risk_levels(
            current_val, mean_val, std_val, signal_direction
        )

        # RISK-004: Check regime suitability
        # For spreads, use half-life as indicator - too long means cointegration weakening
        regime_suitable = self._min_half_life <= half_life <= self._max_half_life
        regime_warning = None
        if not regime_suitable:
            if half_life > self._max_half_life:
                regime_warning = f"Half-life too long ({half_life:.1f} days) - slow mean reversion"
            else:
                regime_warning = f"Half-life too short ({half_life:.1f} days) - noisy spread"

        return SpreadAnalysis(
            spread_name=spread_name,
            spread_type=spread_def.spread_type,
            current_value=current_val,
            zscore=zscore,
            percentile=percentile,
            mean=mean_val,
            std=std_val,
            half_life=half_life,
            is_tradeable=is_tradeable,
            signal_direction=signal_direction,
            signal_strength=signal_strength,
            stop_loss_zscore=self._stop_loss_zscore,
            take_profit_zscore=self._take_profit_zscore,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_holding_bars=self._max_holding_bars,
            strategy_max_loss_pct=self._strategy_max_loss_pct,
            regime_suitable=regime_suitable,
            regime_warning=regime_warning,
            is_exit_signal=signal_direction == "exit",
            exit_reason="take_profit_mean_reverted" if signal_direction == "exit" else None,
        )

    def _calculate_spread_value(
        self,
        spread_def: CommoditySpread,
        prices: dict[str, np.ndarray],
        lookback: int,
        contract_specs: dict[str, Any] | None = None,
        use_multipliers: bool = False
    ) -> np.ndarray:
        """
        Calculate spread value time series using vectorized NumPy operations (PERF-P1-004).

        Args:
            spread_def: Spread definition
            prices: Price series by symbol
            lookback: Number of periods
            contract_specs: Contract specifications (optional)
            use_multipliers: If True, multiply by contract multipliers for
                           dollar-based spread value (vs price-based)

        Returns:
            Array of spread values
        """
        # Determine common length (PERF-P1-004: vectorized approach)
        min_len = min(len(prices[s]) for s in spread_def.legs)
        lookback = min(lookback, min_len)

        # Build coefficient array and price matrix for vectorized computation
        symbols = list(spread_def.legs.keys())
        n_legs = len(symbols)

        # Preallocate price matrix: shape (lookback, n_legs)
        price_matrix = np.empty((lookback, n_legs), dtype=np.float64)
        coefficients = np.empty(n_legs, dtype=np.float64)

        for i, symbol in enumerate(symbols):
            ratio = spread_def.legs[symbol]
            price_series = prices[symbol][-lookback:]

            # Apply multiplier if requested
            multiplier = 1.0
            if use_multipliers and contract_specs and symbol in contract_specs:
                spec = contract_specs[symbol]
                if hasattr(spec, 'multiplier'):
                    multiplier = spec.multiplier
                elif isinstance(spec, dict):
                    multiplier = spec.get("multiplier", 1.0)

            price_matrix[:, i] = price_series
            coefficients[i] = ratio * multiplier

        # Vectorized spread calculation: single matrix-vector multiplication (PERF-P1-004)
        spread_values = price_matrix @ coefficients

        return spread_values

    def get_spread_legs(self, spread_name: str) -> dict[str, float] | None:
        """Get the legs (symbols and ratios) for a spread."""
        if spread_name not in COMMODITY_SPREADS:
            return None
        return dict(COMMODITY_SPREADS[spread_name].legs)

    def calculate_spread_position_sizes(
        self,
        spread_name: str,
        notional_value: float,
        prices: dict[str, float],
        contract_specs: dict[str, Any] | None = None,
        round_to_integer: bool = True
    ) -> dict[str, int] | None:
        """
        Calculate position sizes for spread legs with proper multiplier handling.

        Args:
            spread_name: Spread name
            notional_value: Total notional value to allocate
            prices: Current prices
            contract_specs: Contract specifications (multipliers)
            round_to_integer: Whether to round to integer contracts

        Returns:
            Dictionary of symbol to number of contracts

        Note on ratio interpretation:
        - Ratios in spread definitions are CONTRACT ratios, not price ratios
        - For example, gold/silver ratio of 1:-1.6 means:
          1 GC contract ($200k) vs 1.6 SI contracts (~$200k) = dollar-neutral
        """
        if spread_name not in COMMODITY_SPREADS:
            logger.warning(f"Unknown spread: {spread_name}")
            return None

        spread_def = COMMODITY_SPREADS[spread_name]
        positions = {}

        # Calculate notional value per "unit" of spread
        unit_cost = 0.0
        leg_values = {}

        for symbol, ratio in spread_def.legs.items():
            if symbol not in prices:
                logger.warning(f"Missing price for {symbol} in spread {spread_name}")
                return None

            price = prices[symbol]

            # Get contract multiplier
            multiplier = 1.0
            if contract_specs and symbol in contract_specs:
                spec = contract_specs[symbol]
                if hasattr(spec, 'multiplier'):
                    multiplier = spec.multiplier
                elif isinstance(spec, dict):
                    multiplier = spec.get("multiplier", 1.0)

            # Contract notional = price × multiplier × |ratio|
            contract_notional = price * multiplier * abs(ratio)
            leg_values[symbol] = contract_notional
            unit_cost += contract_notional

        if unit_cost == 0:
            logger.warning(f"Zero unit cost for spread {spread_name}")
            return None

        # Calculate number of "units" (spread sets)
        num_units = notional_value / unit_cost

        if num_units < 0.1:  # Less than 0.1 units not viable
            logger.warning(f"Notional {notional_value} too small for spread {spread_name} (unit_cost={unit_cost:.0f})")
            return None

        # Calculate contracts for each leg
        for symbol, ratio in spread_def.legs.items():
            raw_contracts = num_units * ratio

            if round_to_integer:
                # Round towards zero for safety (avoid over-sizing)
                if raw_contracts > 0:
                    contracts = int(raw_contracts)
                else:
                    contracts = -int(abs(raw_contracts))
            else:
                contracts = raw_contracts

            if contracts != 0:
                positions[symbol] = contracts

        # Validate we have at least 2 legs (for a proper spread)
        if len(positions) < 2:
            logger.warning(f"Spread {spread_name} resulted in fewer than 2 legs after sizing")
            return None

        # Log the position sizing
        logger.debug(
            f"Spread {spread_name}: {num_units:.2f} units, positions={positions}, "
            f"total_notional=${sum(leg_values[s] * abs(positions.get(s, 0)) for s in positions):.0f}"
        )

        return positions

    def calculate_dollar_neutral_spread(
        self,
        spread_name: str,
        target_notional: float,
        prices: dict[str, float],
        contract_specs: dict[str, Any] | None = None
    ) -> dict[str, int] | None:
        """
        Calculate dollar-neutral position sizes for a spread.

        This ensures each leg has approximately equal dollar exposure,
        which may differ from the defined contract ratios for some spreads.

        Args:
            spread_name: Spread name
            target_notional: Target notional per leg
            prices: Current prices
            contract_specs: Contract specifications (multipliers)

        Returns:
            Dictionary of symbol to number of contracts
        """
        if spread_name not in COMMODITY_SPREADS:
            return None

        spread_def = COMMODITY_SPREADS[spread_name]
        positions = {}

        for symbol, ratio in spread_def.legs.items():
            if symbol not in prices:
                return None

            price = prices[symbol]

            # Get contract multiplier
            multiplier = 1.0
            if contract_specs and symbol in contract_specs:
                spec = contract_specs[symbol]
                if hasattr(spec, 'multiplier'):
                    multiplier = spec.multiplier
                elif isinstance(spec, dict):
                    multiplier = spec.get("multiplier", 1.0)

            # Contract notional value
            contract_notional = price * multiplier

            if contract_notional == 0:
                continue

            # Number of contracts for target notional
            num_contracts = target_notional / contract_notional

            # Apply direction from ratio
            if ratio < 0:
                num_contracts = -num_contracts

            # Round to integer
            contracts = int(round(num_contracts))

            if contracts != 0:
                positions[symbol] = contracts

        return positions if len(positions) >= 2 else None

    def get_all_spreads(self) -> list[str]:
        """Get list of all available spread names."""
        return list(COMMODITY_SPREADS.keys())

    def get_spreads_by_type(self, spread_type: SpreadType) -> list[str]:
        """Get spreads of a specific type."""
        return [
            name for name, spread in COMMODITY_SPREADS.items()
            if spread.spread_type == spread_type
        ]

    def get_spread_info(self, spread_name: str) -> CommoditySpread | None:
        """Get detailed spread information."""
        return COMMODITY_SPREADS.get(spread_name)

    def scan_all_spreads(
        self,
        prices: dict[str, np.ndarray]
    ) -> list[SpreadAnalysis]:
        """
        Scan all spreads and return those with signals.

        Args:
            prices: Price data for all symbols

        Returns:
            List of SpreadAnalysis for spreads with active signals
        """
        signals = []

        for spread_name in COMMODITY_SPREADS:
            analysis = self.analyze_spread(spread_name, prices)
            if analysis and analysis.is_tradeable:
                signals.append(analysis)

        # Sort by signal strength
        signals.sort(key=lambda x: x.signal_strength, reverse=True)

        return signals

    def get_seasonal_spreads(self, month: int | None = None) -> list[str]:
        """
        Get spreads that are seasonally favorable.

        Args:
            month: Month to check (1-12), default is current month

        Returns:
            List of spread names
        """
        if month is None:
            month = datetime.now(timezone.utc).month

        seasonal = []
        for name, spread in COMMODITY_SPREADS.items():
            if month in spread.seasonality:
                seasonal.append(name)

        return seasonal

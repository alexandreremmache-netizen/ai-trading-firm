"""
Test Data Generator
===================

Utilities for generating synthetic market data for testing trading strategies.

This module provides functions to generate realistic synthetic price data
with various characteristics:
- Trends (bullish, bearish, sideways)
- Volatility regimes
- Regime changes
- Correlated assets
- Crisis scenarios (2008, COVID, flash crashes)

The generated data is useful for:
- Unit testing strategy logic
- Walk-forward validation testing
- Stress testing risk management
- Integration testing

Note: Synthetic data cannot replace real market data for final validation,
but is essential for repeatable, deterministic testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


class TrendType(str, Enum):
    """Price trend types."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    RANDOM_WALK = "random_walk"


class VolatilityRegime(str, Enum):
    """Volatility regime types."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRISIS = "crisis"


@dataclass
class SyntheticDataConfig:
    """
    Configuration for synthetic data generation.

    Attributes:
        n_days: Number of trading days to generate
        initial_price: Starting price
        annual_return: Expected annualized return (e.g., 0.08 for 8%)
        annual_volatility: Expected annualized volatility (e.g., 0.20 for 20%)
        trend: Type of trend to apply
        seed: Random seed for reproducibility
    """
    n_days: int = 252
    initial_price: float = 100.0
    annual_return: float = 0.08
    annual_volatility: float = 0.20
    trend: TrendType = TrendType.RANDOM_WALK
    seed: int | None = None


@dataclass
class OHLCVData:
    """
    Container for OHLCV (Open, High, Low, Close, Volume) data.

    Attributes:
        dates: Array of datetime objects
        open: Array of open prices
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        volume: Array of volumes
        returns: Array of daily returns (close-to-close)
    """
    dates: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    returns: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        """Calculate returns if not provided."""
        if len(self.returns) == 0 and len(self.close) > 1:
            self.returns = np.diff(self.close) / self.close[:-1]

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary format for strategy input."""
        return {
            "dates": self.dates,
            "prices": self.close,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    def __len__(self) -> int:
        return len(self.close)


class CrisisScenario(str, Enum):
    """
    Pre-defined crisis scenarios for stress testing.

    Each scenario has characteristic return patterns:
    - FINANCIAL_CRISIS_2008: Prolonged decline with extreme volatility
    - COVID_CRASH_2020: Sharp V-shaped crash and recovery
    - FLASH_CRASH: Brief extreme move with quick recovery
    - BLACK_MONDAY_1987: Single-day massive decline
    - DOT_COM_BUST: Gradual decline over extended period
    - VOLMAGEDDON_2018: Volatility spike event
    """
    FINANCIAL_CRISIS_2008 = "2008_crisis"
    COVID_CRASH_2020 = "covid_2020"
    FLASH_CRASH = "flash_crash"
    BLACK_MONDAY_1987 = "black_monday"
    DOT_COM_BUST = "dot_com"
    VOLMAGEDDON_2018 = "volmageddon"


# =============================================================================
# PRICE SERIES GENERATORS
# =============================================================================


def generate_price_series(
    n_days: int = 252,
    initial_price: float = 100.0,
    trend: float = 0.0,
    volatility: float = 0.20,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate synthetic price series using geometric Brownian motion.

    The model: dS = mu*S*dt + sigma*S*dW
    where:
    - S is the stock price
    - mu is the drift (trend)
    - sigma is the volatility
    - dW is Wiener process increment

    Args:
        n_days: Number of trading days to generate
        initial_price: Starting price
        trend: Annualized drift/trend (e.g., 0.08 for 8% annual return)
        volatility: Annualized volatility (e.g., 0.20 for 20%)
        seed: Random seed for reproducibility

    Returns:
        Numpy array of prices

    Example:
        ```python
        prices = generate_price_series(
            n_days=252,
            initial_price=100,
            trend=0.10,
            volatility=0.25,
            seed=42
        )
        ```
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert annual parameters to daily
    dt = 1 / 252
    daily_drift = trend * dt
    daily_vol = volatility * np.sqrt(dt)

    # Generate random normal returns
    random_returns = np.random.normal(0, 1, n_days)

    # Apply GBM model
    log_returns = daily_drift - 0.5 * daily_vol**2 + daily_vol * random_returns

    # Build price series from log returns
    cumulative_returns = np.cumsum(log_returns)
    prices = initial_price * np.exp(cumulative_returns)

    # Prepend initial price
    prices = np.insert(prices, 0, initial_price)[:n_days]

    return prices


def generate_ohlcv_data(
    n_bars: int = 252,
    initial_price: float = 100.0,
    trend: float = 0.0,
    volatility: float = 0.20,
    avg_volume: int = 1_000_000,
    volume_volatility: float = 0.30,
    seed: int | None = None,
    start_date: datetime | None = None,
) -> OHLCVData:
    """
    Generate full OHLCV data with realistic intrabar relationships.

    The relationship between O, H, L, C follows typical patterns:
    - High is always >= max(Open, Close)
    - Low is always <= min(Open, Close)
    - Open is typically close to previous close (gap is rare)
    - Volume is somewhat random but with volatility clustering

    Args:
        n_bars: Number of bars to generate
        initial_price: Starting price
        trend: Annualized drift (e.g., 0.08 for 8%)
        volatility: Annualized volatility (e.g., 0.20 for 20%)
        avg_volume: Average daily volume
        volume_volatility: Volatility of volume (relative to mean)
        seed: Random seed for reproducibility
        start_date: Starting date for the series

    Returns:
        OHLCVData object with all OHLCV arrays

    Example:
        ```python
        data = generate_ohlcv_data(
            n_bars=500,
            trend=0.05,
            volatility=0.18,
            seed=42
        )
        print(f"Generated {len(data)} bars")
        print(f"Price range: {data.low.min():.2f} - {data.high.max():.2f}")
        ```
    """
    if seed is not None:
        np.random.seed(seed)

    if start_date is None:
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

    # Generate base close prices
    close_prices = generate_price_series(
        n_days=n_bars,
        initial_price=initial_price,
        trend=trend,
        volatility=volatility,
    )

    # Generate dates (skip weekends)
    dates = []
    current_date = start_date
    for _ in range(n_bars):
        while current_date.weekday() >= 5:  # Skip weekends
            current_date += timedelta(days=1)
        dates.append(current_date)
        current_date += timedelta(days=1)
    dates = np.array(dates)

    # Generate Open prices (close to previous close with small gap)
    gap_factor = 0.001  # 0.1% typical gap
    gaps = np.random.normal(0, gap_factor, n_bars)
    open_prices = np.zeros(n_bars)
    open_prices[0] = initial_price
    for i in range(1, n_bars):
        open_prices[i] = close_prices[i - 1] * (1 + gaps[i])

    # Generate intrabar range
    daily_vol = volatility / np.sqrt(252)

    # High and Low based on close and volatility
    range_factor = np.abs(np.random.normal(0.5, 0.2, n_bars)) * daily_vol

    high_prices = np.maximum(open_prices, close_prices) * (1 + range_factor)
    low_prices = np.minimum(open_prices, close_prices) * (1 - range_factor)

    # Ensure OHLC consistency
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    # Generate volume with volatility clustering
    # Volume tends to be higher on high-volatility days
    vol_base = np.random.lognormal(
        mean=np.log(avg_volume),
        sigma=volume_volatility,
        size=n_bars
    )

    # Add correlation with price movement (higher volume on big moves)
    returns = np.abs(np.diff(np.log(close_prices)))
    returns = np.insert(returns, 0, 0)
    volume_boost = 1 + returns * 10  # 10x leverage on returns for volume

    volume = (vol_base * volume_boost).astype(int)

    return OHLCVData(
        dates=dates,
        open=open_prices,
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volume,
    )


def generate_correlated_series(
    n_series: int,
    correlation_matrix: np.ndarray,
    n_days: int = 252,
    initial_prices: np.ndarray | None = None,
    trends: np.ndarray | None = None,
    volatilities: np.ndarray | None = None,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate multiple correlated price series.

    Uses Cholesky decomposition to generate correlated random variables
    that follow the specified correlation structure.

    Args:
        n_series: Number of correlated series to generate
        correlation_matrix: n_series x n_series correlation matrix
            Must be symmetric positive semi-definite
        n_days: Number of trading days
        initial_prices: Array of initial prices per series (default: all 100)
        trends: Array of annual trends per series (default: all 0.05)
        volatilities: Array of volatilities per series (default: all 0.20)
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping series name to price array

    Example:
        ```python
        # Generate two correlated series (e.g., stock and its ETF)
        corr_matrix = np.array([
            [1.00, 0.85],
            [0.85, 1.00]
        ])

        series = generate_correlated_series(
            n_series=2,
            correlation_matrix=corr_matrix,
            seed=42
        )
        print(f"Correlation achieved: {np.corrcoef(np.diff(np.log(series['series_0'])), np.diff(np.log(series['series_1'])))[0,1]:.3f}")
        ```
    """
    if seed is not None:
        np.random.seed(seed)

    # Validate correlation matrix
    if correlation_matrix.shape != (n_series, n_series):
        raise ValueError(f"Correlation matrix must be {n_series}x{n_series}")

    if not np.allclose(correlation_matrix, correlation_matrix.T):
        raise ValueError("Correlation matrix must be symmetric")

    # Default parameters
    if initial_prices is None:
        initial_prices = np.full(n_series, 100.0)
    if trends is None:
        trends = np.full(n_series, 0.05)
    if volatilities is None:
        volatilities = np.full(n_series, 0.20)

    # Cholesky decomposition for correlated sampling
    try:
        cholesky = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # If not positive definite, use nearest positive definite
        eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        correlation_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        cholesky = np.linalg.cholesky(correlation_matrix)

    # Generate uncorrelated random numbers
    uncorrelated = np.random.normal(0, 1, (n_days, n_series))

    # Apply Cholesky to get correlated random numbers
    correlated = uncorrelated @ cholesky.T

    # Generate price series for each asset
    dt = 1 / 252
    series_dict = {}

    for i in range(n_series):
        daily_drift = trends[i] * dt
        daily_vol = volatilities[i] * np.sqrt(dt)

        log_returns = (
            daily_drift - 0.5 * daily_vol**2
            + daily_vol * correlated[:, i]
        )

        cumulative_returns = np.cumsum(log_returns)
        prices = initial_prices[i] * np.exp(
            np.insert(cumulative_returns, 0, 0)
        )[:n_days]

        series_dict[f"series_{i}"] = prices

    return series_dict


def generate_regime_switching_data(
    n_days: int = 504,  # 2 years
    n_regimes: int = 2,
    avg_regime_length: int = 63,  # Quarterly regime switches
    regime_params: list[dict[str, float]] | None = None,
    seed: int | None = None,
    initial_price: float = 100.0,
    start_date: datetime | None = None,
) -> OHLCVData:
    """
    Generate price data with regime-switching characteristics.

    Markets exhibit different regimes (bull/bear, high/low volatility)
    that switch over time. This function generates data that mimics
    these regime changes for testing strategy adaptation.

    Args:
        n_days: Total number of trading days
        n_regimes: Number of distinct regimes
        avg_regime_length: Average days per regime (geometrically distributed)
        regime_params: List of dicts with 'trend' and 'volatility' per regime
            Default: [{'trend': 0.20, 'vol': 0.15}, {'trend': -0.10, 'vol': 0.35}]
        seed: Random seed
        initial_price: Starting price
        start_date: Starting date

    Returns:
        OHLCVData with regime-switching characteristics

    Example:
        ```python
        # Generate data with bull and bear regimes
        regime_params = [
            {'trend': 0.25, 'volatility': 0.12},  # Bull market
            {'trend': -0.15, 'volatility': 0.40}, # Bear market
        ]

        data = generate_regime_switching_data(
            n_days=500,
            regime_params=regime_params,
            seed=42
        )
        ```
    """
    if seed is not None:
        np.random.seed(seed)

    if start_date is None:
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

    # Default regime parameters (bull vs bear)
    if regime_params is None:
        regime_params = [
            {"trend": 0.20, "volatility": 0.15},  # Bull regime
            {"trend": -0.10, "volatility": 0.35},  # Bear regime
        ]

    if len(regime_params) != n_regimes:
        raise ValueError(f"regime_params must have {n_regimes} entries")

    # Generate regime sequence
    regimes = []
    current_regime = np.random.randint(0, n_regimes)

    while len(regimes) < n_days:
        # Geometric distribution for regime length
        regime_length = np.random.geometric(1 / avg_regime_length)
        regime_length = min(regime_length, n_days - len(regimes))

        regimes.extend([current_regime] * regime_length)

        # Switch to different regime
        available_regimes = [r for r in range(n_regimes) if r != current_regime]
        current_regime = np.random.choice(available_regimes)

    regimes = np.array(regimes[:n_days])

    # Generate returns based on regime
    dt = 1 / 252
    returns = np.zeros(n_days)

    for i in range(n_days):
        regime = regimes[i]
        params = regime_params[regime]
        daily_drift = params["trend"] * dt
        daily_vol = params["volatility"] * np.sqrt(dt)

        returns[i] = daily_drift - 0.5 * daily_vol**2 + daily_vol * np.random.normal()

    # Build price series
    log_prices = np.log(initial_price) + np.cumsum(returns)
    close_prices = np.exp(log_prices)

    # Generate full OHLCV
    dates = []
    current_date = start_date
    for _ in range(n_days):
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        dates.append(current_date)
        current_date += timedelta(days=1)

    # Generate OHLC based on regime volatility
    open_prices = np.zeros(n_days)
    high_prices = np.zeros(n_days)
    low_prices = np.zeros(n_days)

    open_prices[0] = initial_price
    for i in range(1, n_days):
        # Open is previous close with small gap
        open_prices[i] = close_prices[i - 1] * (1 + np.random.normal(0, 0.001))

    for i in range(n_days):
        regime = regimes[i]
        daily_vol = regime_params[regime]["volatility"] / np.sqrt(252)

        range_factor = abs(np.random.normal(0.5, 0.2)) * daily_vol

        high_prices[i] = max(open_prices[i], close_prices[i]) * (1 + range_factor)
        low_prices[i] = min(open_prices[i], close_prices[i]) * (1 - range_factor)

    # Generate volume (higher in volatile regimes)
    base_volume = 1_000_000
    volume = np.zeros(n_days, dtype=int)
    for i in range(n_days):
        regime = regimes[i]
        vol_mult = regime_params[regime]["volatility"] / 0.20  # Normalize to 20%
        volume[i] = int(base_volume * vol_mult * np.random.lognormal(0, 0.3))

    return OHLCVData(
        dates=np.array(dates),
        open=open_prices,
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volume,
    )


def generate_crisis_scenario(
    scenario_name: CrisisScenario | str,
    n_days: int | None = None,
    initial_price: float = 100.0,
    seed: int | None = None,
    start_date: datetime | None = None,
) -> OHLCVData:
    """
    Generate price data mimicking historical crisis scenarios.

    Each crisis has characteristic patterns:
    - 2008: Gradual decline, spike in volatility, 50%+ drawdown
    - COVID: Sharp V-shaped crash, 30% drop in 3 weeks, quick recovery
    - Flash Crash: Extreme intraday move, largely recovered same day
    - Black Monday: Single-day 20%+ decline
    - Dot-com: Gradual decline over 2+ years
    - Volmageddon: Volatility spike causing 90%+ drop in vol products

    Args:
        scenario_name: Crisis scenario to generate (enum or string)
        n_days: Override default scenario length
        initial_price: Starting price
        seed: Random seed
        start_date: Starting date

    Returns:
        OHLCVData representing the crisis scenario

    Example:
        ```python
        # Generate 2008-like crisis data
        crisis_data = generate_crisis_scenario(
            CrisisScenario.FINANCIAL_CRISIS_2008,
            seed=42
        )

        # Check the drawdown
        dd = (crisis_data.high.max() - crisis_data.low.min()) / crisis_data.high.max()
        print(f"Max drawdown: {dd*100:.1f}%")
        ```
    """
    if seed is not None:
        np.random.seed(seed)

    if start_date is None:
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

    if isinstance(scenario_name, str):
        scenario_name = CrisisScenario(scenario_name)

    # Define scenario parameters
    scenario_configs = {
        CrisisScenario.FINANCIAL_CRISIS_2008: {
            "default_days": 400,  # ~1.5 years
            "phases": [
                {"days": 50, "trend": 0.10, "vol": 0.15},   # Pre-crisis rally
                {"days": 100, "trend": -0.60, "vol": 0.50},  # Crisis phase
                {"days": 80, "trend": -0.30, "vol": 0.60},   # Deep crisis
                {"days": 120, "trend": 0.40, "vol": 0.35},   # Recovery begins
                {"days": 50, "trend": 0.25, "vol": 0.25},    # Stabilization
            ],
        },
        CrisisScenario.COVID_CRASH_2020: {
            "default_days": 100,
            "phases": [
                {"days": 10, "trend": 0.20, "vol": 0.12},   # Pre-crash
                {"days": 23, "trend": -2.00, "vol": 0.80},  # Sharp crash
                {"days": 45, "trend": 1.50, "vol": 0.50},   # V-recovery
                {"days": 22, "trend": 0.30, "vol": 0.25},   # Stabilization
            ],
        },
        CrisisScenario.FLASH_CRASH: {
            "default_days": 5,
            "phases": [
                {"days": 1, "trend": 0.0, "vol": 0.10},     # Normal day start
                {"days": 1, "trend": -3.00, "vol": 2.00},   # Flash crash
                {"days": 1, "trend": 2.50, "vol": 1.50},    # Recovery
                {"days": 2, "trend": 0.05, "vol": 0.30},    # Aftermath
            ],
        },
        CrisisScenario.BLACK_MONDAY_1987: {
            "default_days": 30,
            "phases": [
                {"days": 10, "trend": 0.30, "vol": 0.15},   # Pre-crash rally
                {"days": 1, "trend": -5.00, "vol": 3.00},   # Black Monday
                {"days": 5, "trend": -0.50, "vol": 0.80},   # Continued decline
                {"days": 14, "trend": 0.20, "vol": 0.40},   # Recovery
            ],
        },
        CrisisScenario.DOT_COM_BUST: {
            "default_days": 600,  # ~2.5 years
            "phases": [
                {"days": 50, "trend": 0.80, "vol": 0.30},   # Bubble peak
                {"days": 150, "trend": -0.40, "vol": 0.35}, # First decline
                {"days": 100, "trend": 0.10, "vol": 0.30},  # Dead cat bounce
                {"days": 200, "trend": -0.30, "vol": 0.25}, # Continued decline
                {"days": 100, "trend": 0.15, "vol": 0.20},  # Bottom and recovery
            ],
        },
        CrisisScenario.VOLMAGEDDON_2018: {
            "default_days": 20,
            "phases": [
                {"days": 8, "trend": 0.20, "vol": 0.08},    # Low vol environment
                {"days": 2, "trend": -0.40, "vol": 1.50},   # Vol spike
                {"days": 3, "trend": -0.20, "vol": 0.60},   # Continued stress
                {"days": 7, "trend": 0.15, "vol": 0.25},    # Normalization
            ],
        },
    }

    config = scenario_configs[scenario_name]

    if n_days is None:
        n_days = config["default_days"]

    # Scale phases to fit requested n_days
    total_phase_days = sum(p["days"] for p in config["phases"])
    scale_factor = n_days / total_phase_days

    # Generate returns phase by phase
    returns = []
    dt = 1 / 252

    for phase in config["phases"]:
        phase_days = int(phase["days"] * scale_factor)
        if phase_days < 1:
            phase_days = 1

        daily_drift = phase["trend"] * dt
        daily_vol = phase["vol"] * np.sqrt(dt)

        phase_returns = (
            daily_drift - 0.5 * daily_vol**2
            + daily_vol * np.random.normal(0, 1, phase_days)
        )
        returns.extend(phase_returns)

    returns = np.array(returns[:n_days])

    # Ensure we have exactly n_days
    if len(returns) < n_days:
        # Pad with normal returns
        pad_returns = np.random.normal(0, 0.01, n_days - len(returns))
        returns = np.concatenate([returns, pad_returns])

    # Build price series
    log_prices = np.log(initial_price) + np.cumsum(returns)
    close_prices = np.exp(log_prices)

    # Generate dates
    dates = []
    current_date = start_date
    for _ in range(n_days):
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        dates.append(current_date)
        current_date += timedelta(days=1)

    # Generate OHLC with crisis-appropriate volatility
    open_prices = np.zeros(n_days)
    high_prices = np.zeros(n_days)
    low_prices = np.zeros(n_days)

    open_prices[0] = initial_price
    for i in range(1, n_days):
        gap = np.random.normal(0, 0.005)  # Larger gaps during crisis
        open_prices[i] = close_prices[i - 1] * (1 + gap)

    # Use realized volatility for high/low range
    for i in range(n_days):
        lookback = min(i + 1, 10)
        if lookback > 1:
            recent_vol = np.std(returns[max(0, i - lookback):i + 1]) * np.sqrt(252)
        else:
            recent_vol = 0.20

        range_factor = abs(np.random.normal(0.5, 0.3)) * recent_vol / np.sqrt(252)

        high_prices[i] = max(open_prices[i], close_prices[i]) * (1 + range_factor)
        low_prices[i] = min(open_prices[i], close_prices[i]) * (1 - range_factor)

    # Volume spikes during crisis
    volume = np.zeros(n_days, dtype=int)
    base_volume = 1_000_000

    for i in range(n_days):
        abs_return = abs(returns[i])
        vol_multiplier = 1 + abs_return * 50  # Higher volume on big moves
        volume[i] = int(base_volume * vol_multiplier * np.random.lognormal(0, 0.3))

    return OHLCVData(
        dates=np.array(dates),
        open=open_prices,
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volume,
    )


# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# =============================================================================


def generate_mean_reverting_series(
    n_days: int = 252,
    mean_price: float = 100.0,
    volatility: float = 0.15,
    mean_reversion_speed: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate mean-reverting price series (Ornstein-Uhlenbeck process).

    Useful for testing mean-reversion strategies or stat-arb pairs.

    The model: dX = theta*(mu - X)*dt + sigma*dW
    where:
    - theta is mean reversion speed
    - mu is long-term mean
    - sigma is volatility

    Args:
        n_days: Number of days
        mean_price: Long-term mean price
        volatility: Annualized volatility
        mean_reversion_speed: Rate of reversion (higher = faster reversion)
        seed: Random seed

    Returns:
        Array of prices
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1 / 252
    daily_vol = volatility * np.sqrt(dt)

    prices = np.zeros(n_days)
    prices[0] = mean_price

    for i in range(1, n_days):
        mean_rev = mean_reversion_speed * (mean_price - prices[i - 1])
        shock = daily_vol * np.random.normal()
        prices[i] = prices[i - 1] + mean_rev + prices[i - 1] * shock

    return prices


def generate_jump_diffusion_series(
    n_days: int = 252,
    initial_price: float = 100.0,
    trend: float = 0.05,
    volatility: float = 0.20,
    jump_intensity: float = 0.01,  # ~2-3 jumps per year
    jump_mean: float = -0.05,  # Average jump size (-5%)
    jump_std: float = 0.10,  # Jump size volatility
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate price series with occasional jumps (Merton jump-diffusion).

    Useful for testing strategies that need to handle sudden price gaps.

    Args:
        n_days: Number of days
        initial_price: Starting price
        trend: Annualized drift
        volatility: Annualized diffusion volatility
        jump_intensity: Daily probability of jump
        jump_mean: Average jump size (as fraction, e.g., -0.05 for -5%)
        jump_std: Standard deviation of jump size
        seed: Random seed

    Returns:
        Array of prices with jump events
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1 / 252
    daily_drift = trend * dt
    daily_vol = volatility * np.sqrt(dt)

    returns = np.zeros(n_days)

    for i in range(n_days):
        # Normal diffusion component
        diffusion = daily_drift - 0.5 * daily_vol**2 + daily_vol * np.random.normal()

        # Jump component (Poisson arrival, log-normal size)
        if np.random.random() < jump_intensity:
            jump = np.random.normal(jump_mean, jump_std)
        else:
            jump = 0

        returns[i] = diffusion + jump

    # Build price series
    log_prices = np.log(initial_price) + np.cumsum(returns)
    prices = np.exp(log_prices)

    return prices


def generate_seasonal_pattern(
    n_days: int = 252,
    initial_price: float = 100.0,
    trend: float = 0.05,
    volatility: float = 0.15,
    seasonality_amplitude: float = 0.05,  # +/- 5% seasonal swing
    seasonal_period: int = 252,  # Annual seasonality
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate price series with seasonal patterns.

    Useful for testing strategies that exploit seasonality (e.g., "Sell in May").

    Args:
        n_days: Number of days
        initial_price: Starting price
        trend: Annualized trend
        volatility: Volatility
        seasonality_amplitude: Peak-to-trough seasonal swing (as fraction)
        seasonal_period: Number of days for one seasonal cycle
        seed: Random seed

    Returns:
        Array of prices with seasonal pattern
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate base GBM prices
    base_prices = generate_price_series(
        n_days=n_days,
        initial_price=initial_price,
        trend=trend,
        volatility=volatility,
    )

    # Add seasonal component
    seasonal_factor = 1 + seasonality_amplitude * np.sin(
        2 * np.pi * np.arange(n_days) / seasonal_period
    )

    prices = base_prices * seasonal_factor

    return prices

"""
Index Spread Strategy (Phase 6.2)
=================================

MES/MNQ (E-mini S&P 500 / E-mini Nasdaq 100) spread trading strategy.

Key features:
- Ratio trading between correlated index futures
- Z-score based entry/exit signals
- Dollar-neutral positioning
- Dynamic hedge ratio adjustment

Research basis:
- MES:MNQ correlation typically ~0.85-0.95
- Mean reversion half-life: 3-10 days
- Best during range-bound macro environment

MATURITY: ALPHA
---------------
Status: New implementation
- [x] Spread calculation
- [x] Z-score signals
- [x] Dynamic hedge ratio
- [ ] Integration with main system
- [ ] Backtesting validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class SpreadRelationship(Enum):
    """Spread relationship state."""
    NORMAL = "normal"           # Within 1 std
    EXTENDED = "extended"       # 1-2 std
    EXTREME = "extreme"         # >2 std
    BROKEN = "broken"           # Relationship breakdown


@dataclass
class SpreadState:
    """Current state of the spread."""
    spread_value: float
    zscore: float
    hedge_ratio: float
    relationship: SpreadRelationship
    half_life_days: float
    correlation: float
    is_cointegrated: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SpreadSignal:
    """Signal from spread analysis."""
    symbol_long: str
    symbol_short: str
    direction: str  # "LONG_SPREAD" (long A, short B), "SHORT_SPREAD", "FLAT"
    strength: float  # 0.0 to 1.0
    hedge_ratio: float
    entry_zscore: float
    target_zscore: float
    stop_zscore: float
    position_size_ratio: dict[str, float]  # {symbol: ratio}
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# PREDEFINED INDEX SPREADS
# =============================================================================

INDEX_SPREAD_DEFINITIONS = {
    "MES_MNQ": {
        "name": "E-mini S&P/Nasdaq Spread",
        "leg_a": "MES",  # S&P 500
        "leg_b": "MNQ",  # Nasdaq 100
        "typical_correlation": 0.90,
        "typical_half_life": 5,
        "multiplier_a": 5.0,   # MES multiplier
        "multiplier_b": 2.0,   # MNQ multiplier
        "margin_offset_pct": 80.0,
        "description": "Tech-weighted vs broad market spread",
    },
    "ES_NQ": {
        "name": "Full-size S&P/Nasdaq Spread",
        "leg_a": "ES",
        "leg_b": "NQ",
        "typical_correlation": 0.90,
        "typical_half_life": 5,
        "multiplier_a": 50.0,
        "multiplier_b": 20.0,
        "margin_offset_pct": 80.0,
        "description": "Full-size index futures spread",
    },
    "MES_MYM": {
        "name": "S&P/Dow Spread",
        "leg_a": "MES",  # S&P 500
        "leg_b": "MYM",  # Dow Jones
        "typical_correlation": 0.95,
        "typical_half_life": 3,
        "multiplier_a": 5.0,
        "multiplier_b": 0.5,
        "margin_offset_pct": 85.0,
        "description": "Large-cap vs blue-chip spread",
    },
    "MNQ_M2K": {
        "name": "Nasdaq/Russell Spread",
        "leg_a": "MNQ",  # Nasdaq 100
        "leg_b": "M2K",  # Russell 2000
        "typical_correlation": 0.75,
        "typical_half_life": 7,
        "multiplier_a": 2.0,
        "multiplier_b": 5.0,
        "margin_offset_pct": 70.0,
        "description": "Large-cap tech vs small-cap spread",
    },
}


class IndexSpreadStrategy:
    """
    Index spread trading strategy (Phase 6.2).

    Trades the relative value between correlated index futures.
    Uses cointegration and z-score based entry/exit.

    Configuration:
        zscore_entry: Z-score threshold for entry (default: 2.0)
        zscore_exit: Z-score threshold for exit (default: 0.5)
        zscore_stop: Z-score for stop-loss (default: 3.5)
        lookback_days: Lookback for statistics (default: 60)
        min_correlation: Minimum correlation required (default: 0.7)
        min_half_life: Minimum half-life in days (default: 1)
        max_half_life: Maximum half-life in days (default: 20)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize index spread strategy."""
        config = config or {}

        # Entry/exit thresholds
        self._zscore_entry = config.get("zscore_entry", 2.0)
        self._zscore_exit = config.get("zscore_exit", 0.5)
        self._zscore_stop = config.get("zscore_stop", 3.5)

        # Statistics settings
        self._lookback_days = config.get("lookback_days", 60)
        self._min_correlation = config.get("min_correlation", 0.7)
        self._min_half_life = config.get("min_half_life", 1)
        self._max_half_life = config.get("max_half_life", 20)

        # State tracking
        self._spread_states: dict[str, SpreadState] = {}
        self._hedge_ratios: dict[str, float] = {}

        logger.info(
            f"IndexSpreadStrategy initialized: "
            f"entry={self._zscore_entry}, exit={self._zscore_exit}, "
            f"lookback={self._lookback_days}d"
        )

    def calculate_spread(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        hedge_ratio: float | None = None,
    ) -> np.ndarray:
        """
        Calculate spread between two price series.

        Args:
            prices_a: Prices of asset A (long leg)
            prices_b: Prices of asset B (short leg)
            hedge_ratio: Fixed hedge ratio (if None, calculated from data)

        Returns:
            Spread series
        """
        if hedge_ratio is None:
            hedge_ratio = self._estimate_hedge_ratio(prices_a, prices_b)

        return prices_a - hedge_ratio * prices_b

    def _estimate_hedge_ratio(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> float:
        """Estimate hedge ratio using OLS."""
        var_b = np.var(prices_b)
        if var_b < 1e-12:
            return 1.0

        cov_matrix = np.cov(prices_a, prices_b)
        if cov_matrix.ndim == 0:
            return 1.0

        beta = cov_matrix[0, 1] / var_b
        return beta if np.isfinite(beta) else 1.0

    def calculate_zscore(
        self,
        spread: np.ndarray,
        lookback: int | None = None,
    ) -> float:
        """
        Calculate z-score of current spread value.

        Args:
            spread: Spread series
            lookback: Lookback period (uses full series if None)

        Returns:
            Current z-score
        """
        if lookback is not None:
            spread = spread[-lookback:]

        if len(spread) < 2:
            return 0.0

        mean = np.mean(spread)
        std = np.std(spread)

        if std < 1e-12:
            return 0.0

        return (spread[-1] - mean) / std

    def calculate_half_life(
        self,
        spread: np.ndarray,
    ) -> float:
        """
        Calculate half-life of mean reversion (Ornstein-Uhlenbeck).

        Args:
            spread: Spread series

        Returns:
            Half-life in periods (bars)
        """
        if len(spread) < 10:
            return float('inf')

        # Lag-1 regression: spread_t = a + b * spread_{t-1} + e
        lagged = spread[:-1]
        current = spread[1:]

        # OLS regression
        n = len(lagged)
        x_mean = np.mean(lagged)
        y_mean = np.mean(current)

        numerator = np.sum((lagged - x_mean) * (current - y_mean))
        denominator = np.sum((lagged - x_mean) ** 2)

        if abs(denominator) < 1e-12:
            return float('inf')

        b = numerator / denominator

        # Half-life = -log(2) / log(b)
        if b <= 0 or b >= 1:
            return float('inf')

        half_life = -np.log(2) / np.log(b)
        return max(0.1, half_life)

    def analyze_spread(
        self,
        spread_name: str,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> SpreadState:
        """
        Analyze spread state.

        Args:
            spread_name: Name of the spread (e.g., "MES_MNQ")
            prices_a: Prices of first leg
            prices_b: Prices of second leg

        Returns:
            SpreadState with current analysis
        """
        # Calculate hedge ratio
        hedge_ratio = self._estimate_hedge_ratio(prices_a, prices_b)
        self._hedge_ratios[spread_name] = hedge_ratio

        # Calculate spread
        spread = self.calculate_spread(prices_a, prices_b, hedge_ratio)

        # Calculate statistics
        zscore = self.calculate_zscore(spread, self._lookback_days)
        half_life = self.calculate_half_life(spread)
        correlation = np.corrcoef(prices_a, prices_b)[0, 1]

        # Determine relationship state
        abs_z = abs(zscore)
        if abs_z > 3.0 and half_life > self._max_half_life:
            relationship = SpreadRelationship.BROKEN
        elif abs_z > 2.0:
            relationship = SpreadRelationship.EXTREME
        elif abs_z > 1.0:
            relationship = SpreadRelationship.EXTENDED
        else:
            relationship = SpreadRelationship.NORMAL

        # Check cointegration (simplified)
        is_cointegrated = (
            correlation >= self._min_correlation and
            self._min_half_life <= half_life <= self._max_half_life
        )

        state = SpreadState(
            spread_value=spread[-1],
            zscore=zscore,
            hedge_ratio=hedge_ratio,
            relationship=relationship,
            half_life_days=half_life,
            correlation=correlation,
            is_cointegrated=is_cointegrated,
        )

        self._spread_states[spread_name] = state
        return state

    def generate_signal(
        self,
        spread_name: str,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        current_position: str = "FLAT",
    ) -> SpreadSignal | None:
        """
        Generate trading signal for spread.

        Args:
            spread_name: Spread identifier
            prices_a: Prices of first leg
            prices_b: Prices of second leg
            current_position: Current position ("FLAT", "LONG_SPREAD", "SHORT_SPREAD")

        Returns:
            SpreadSignal if conditions met, None otherwise
        """
        # Get spread definition
        spread_def = INDEX_SPREAD_DEFINITIONS.get(spread_name, {})
        symbol_a = spread_def.get("leg_a", "A")
        symbol_b = spread_def.get("leg_b", "B")

        # Analyze spread
        state = self.analyze_spread(spread_name, prices_a, prices_b)

        # Check if relationship is valid for trading
        if not state.is_cointegrated:
            logger.debug(f"Spread {spread_name} not cointegrated, skipping")
            return None

        if state.relationship == SpreadRelationship.BROKEN:
            logger.warning(f"Spread {spread_name} relationship broken")
            return None

        # Determine signal
        direction = "FLAT"
        strength = 0.0

        if current_position == "FLAT":
            # Entry signals
            if state.zscore > self._zscore_entry:
                direction = "SHORT_SPREAD"  # Spread too high, short A, long B
                strength = min(1.0, (state.zscore - self._zscore_entry) / 2)
            elif state.zscore < -self._zscore_entry:
                direction = "LONG_SPREAD"  # Spread too low, long A, short B
                strength = min(1.0, (-state.zscore - self._zscore_entry) / 2)

        elif current_position == "LONG_SPREAD":
            # Exit or stop for long spread
            if state.zscore >= -self._zscore_exit:
                direction = "FLAT"  # Take profit
                strength = 0.8
            elif state.zscore < -self._zscore_stop:
                direction = "FLAT"  # Stop loss
                strength = 1.0

        elif current_position == "SHORT_SPREAD":
            # Exit or stop for short spread
            if state.zscore <= self._zscore_exit:
                direction = "FLAT"  # Take profit
                strength = 0.8
            elif state.zscore > self._zscore_stop:
                direction = "FLAT"  # Stop loss
                strength = 1.0

        if direction == "FLAT" and current_position == "FLAT":
            return None

        # Calculate position sizing
        mult_a = spread_def.get("multiplier_a", 1.0)
        mult_b = spread_def.get("multiplier_b", 1.0)

        # Dollar-neutral sizing
        notional_a = prices_a[-1] * mult_a
        notional_b = prices_b[-1] * mult_b * state.hedge_ratio

        total_notional = notional_a + notional_b
        ratio_a = notional_a / total_notional if total_notional > 0 else 0.5
        ratio_b = notional_b / total_notional if total_notional > 0 else 0.5

        return SpreadSignal(
            symbol_long=symbol_a if direction == "LONG_SPREAD" else symbol_b,
            symbol_short=symbol_b if direction == "LONG_SPREAD" else symbol_a,
            direction=direction,
            strength=strength,
            hedge_ratio=state.hedge_ratio,
            entry_zscore=state.zscore,
            target_zscore=0.0,  # Mean reversion target
            stop_zscore=self._zscore_stop * (1 if state.zscore < 0 else -1),
            position_size_ratio={symbol_a: ratio_a, symbol_b: ratio_b},
            rationale=(
                f"Spread {spread_name}: zscore={state.zscore:.2f}, "
                f"half_life={state.half_life_days:.1f}d, corr={state.correlation:.2f}"
            ),
        )

    def get_status(self) -> dict[str, Any]:
        """Get strategy status."""
        return {
            "zscore_entry": self._zscore_entry,
            "zscore_exit": self._zscore_exit,
            "zscore_stop": self._zscore_stop,
            "lookback_days": self._lookback_days,
            "tracked_spreads": len(self._spread_states),
            "spread_states": {
                name: {
                    "zscore": state.zscore,
                    "relationship": state.relationship.value,
                    "is_cointegrated": state.is_cointegrated,
                }
                for name, state in self._spread_states.items()
            },
        }


def create_index_spread_strategy(config: dict[str, Any] | None = None) -> IndexSpreadStrategy:
    """Create IndexSpreadStrategy instance."""
    return IndexSpreadStrategy(config)

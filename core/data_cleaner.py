"""
Data Cleaner Module
===================

Handles data cleaning and adjustments for market data.

Features:
- Price adjustment for stock splits
- Dividend adjustment
- Corporate action handling
- Data normalization
- Gap filling

P3 Enhancement: Data cleaning for accurate historical analysis.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CorporateActionType(str, Enum):
    """Types of corporate actions."""
    STOCK_SPLIT = "stock_split"
    REVERSE_SPLIT = "reverse_split"
    DIVIDEND = "dividend"
    SPECIAL_DIVIDEND = "special_dividend"
    SPIN_OFF = "spin_off"
    MERGER = "merger"
    ACQUISITION = "acquisition"
    NAME_CHANGE = "name_change"
    SYMBOL_CHANGE = "symbol_change"
    RIGHTS_ISSUE = "rights_issue"
    BONUS_ISSUE = "bonus_issue"


@dataclass
class CorporateAction:
    """Represents a corporate action event."""
    action_type: CorporateActionType
    symbol: str
    effective_date: datetime
    # For splits: ratio is new_shares/old_shares (e.g., 2.0 for 2-for-1 split)
    # For dividends: amount is dividend per share
    ratio: float = 1.0
    amount: float = 0.0
    new_symbol: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate corporate action."""
        if self.action_type in (CorporateActionType.STOCK_SPLIT, CorporateActionType.REVERSE_SPLIT):
            if self.ratio <= 0:
                raise ValueError(f"Split ratio must be positive: {self.ratio}")
        if self.action_type in (CorporateActionType.DIVIDEND, CorporateActionType.SPECIAL_DIVIDEND):
            if self.amount < 0:
                raise ValueError(f"Dividend amount cannot be negative: {self.amount}")


@dataclass
class AdjustmentResult:
    """Result of price/data adjustment."""
    original_value: float
    adjusted_value: float
    adjustment_factor: float
    adjustment_reason: str
    corporate_action: CorporateAction | None = None


class CorporateActionManager:
    """
    Manages corporate actions and their effects on prices.

    Tracks historical corporate actions and applies adjustments
    to historical price data for accurate backtesting.
    """

    def __init__(self):
        """Initialize corporate action manager."""
        # Corporate actions by symbol, sorted by date
        self._actions: dict[str, list[CorporateAction]] = {}
        # Cumulative adjustment factors by symbol (for quick lookups)
        self._adjustment_factors: dict[str, dict[datetime, float]] = {}

    def add_action(self, action: CorporateAction) -> None:
        """
        Add a corporate action.

        Args:
            action: Corporate action to add
        """
        if action.symbol not in self._actions:
            self._actions[action.symbol] = []

        self._actions[action.symbol].append(action)
        # Keep sorted by date
        self._actions[action.symbol].sort(key=lambda a: a.effective_date)
        # Invalidate cached factors
        self._adjustment_factors.pop(action.symbol, None)

        logger.info(
            f"Added corporate action for {action.symbol}: "
            f"{action.action_type.value} on {action.effective_date.date()}"
        )

    def get_actions(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[CorporateAction]:
        """
        Get corporate actions for a symbol within date range.

        Args:
            symbol: Instrument symbol
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            List of corporate actions
        """
        actions = self._actions.get(symbol, [])

        if start_date:
            actions = [a for a in actions if a.effective_date >= start_date]
        if end_date:
            actions = [a for a in actions if a.effective_date <= end_date]

        return actions

    def calculate_split_adjustment(
        self,
        symbol: str,
        as_of_date: datetime,
    ) -> float:
        """
        Calculate cumulative split adjustment factor as of a date.

        The factor represents how many current shares equal one historical share.
        E.g., after a 2-for-1 split, factor is 2.0.

        Args:
            symbol: Instrument symbol
            as_of_date: Date to calculate adjustment as of

        Returns:
            Cumulative adjustment factor
        """
        actions = self._actions.get(symbol, [])
        factor = 1.0

        for action in actions:
            if action.effective_date > as_of_date:
                continue

            if action.action_type == CorporateActionType.STOCK_SPLIT:
                factor *= action.ratio
            elif action.action_type == CorporateActionType.REVERSE_SPLIT:
                factor /= action.ratio
            elif action.action_type == CorporateActionType.BONUS_ISSUE:
                factor *= (1 + action.ratio)

        return factor

    def adjust_price_for_splits(
        self,
        symbol: str,
        price: float,
        price_date: datetime,
        target_date: datetime | None = None,
    ) -> AdjustmentResult:
        """
        Adjust a historical price for splits.

        Args:
            symbol: Instrument symbol
            price: Original price
            price_date: Date of the original price
            target_date: Date to adjust to (default: now)

        Returns:
            AdjustmentResult with adjusted price
        """
        target_date = target_date or datetime.now(timezone.utc)

        # Get factors at both dates
        factor_at_price = self.calculate_split_adjustment(symbol, price_date)
        factor_at_target = self.calculate_split_adjustment(symbol, target_date)

        if factor_at_price == 0:
            adjustment_factor = 1.0
        else:
            adjustment_factor = factor_at_target / factor_at_price

        adjusted_price = price / adjustment_factor

        # Find the actions that caused this adjustment
        actions = [
            a for a in self._actions.get(symbol, [])
            if price_date < a.effective_date <= target_date
            and a.action_type in (
                CorporateActionType.STOCK_SPLIT,
                CorporateActionType.REVERSE_SPLIT,
                CorporateActionType.BONUS_ISSUE,
            )
        ]

        return AdjustmentResult(
            original_value=price,
            adjusted_value=adjusted_price,
            adjustment_factor=adjustment_factor,
            adjustment_reason="split_adjustment" if adjustment_factor != 1.0 else "no_adjustment",
            corporate_action=actions[0] if actions else None,
        )

    def adjust_price_for_dividends(
        self,
        symbol: str,
        price: float,
        price_date: datetime,
        target_date: datetime | None = None,
    ) -> AdjustmentResult:
        """
        Adjust a historical price for dividends (dividend-adjusted price).

        This is useful for calculating total return.

        Args:
            symbol: Instrument symbol
            price: Original price
            price_date: Date of the original price
            target_date: Date to adjust to (default: now)

        Returns:
            AdjustmentResult with adjusted price
        """
        target_date = target_date or datetime.now(timezone.utc)

        # Get dividend actions between dates
        dividend_actions = [
            a for a in self._actions.get(symbol, [])
            if price_date < a.effective_date <= target_date
            and a.action_type in (
                CorporateActionType.DIVIDEND,
                CorporateActionType.SPECIAL_DIVIDEND,
            )
        ]

        # Calculate cumulative dividend adjustment
        total_dividend = sum(a.amount for a in dividend_actions)

        # Adjust price down by dividends paid
        adjusted_price = price - total_dividend
        adjustment_factor = adjusted_price / price if price > 0 else 1.0

        return AdjustmentResult(
            original_value=price,
            adjusted_value=max(adjusted_price, 0),
            adjustment_factor=adjustment_factor,
            adjustment_reason="dividend_adjustment" if total_dividend > 0 else "no_adjustment",
            corporate_action=dividend_actions[0] if dividend_actions else None,
        )

    def adjust_volume_for_splits(
        self,
        symbol: str,
        volume: int,
        volume_date: datetime,
        target_date: datetime | None = None,
    ) -> AdjustmentResult:
        """
        Adjust historical volume for splits.

        Args:
            symbol: Instrument symbol
            volume: Original volume
            volume_date: Date of the original volume
            target_date: Date to adjust to (default: now)

        Returns:
            AdjustmentResult with adjusted volume
        """
        target_date = target_date or datetime.now(timezone.utc)

        factor_at_volume = self.calculate_split_adjustment(symbol, volume_date)
        factor_at_target = self.calculate_split_adjustment(symbol, target_date)

        if factor_at_volume == 0:
            adjustment_factor = 1.0
        else:
            adjustment_factor = factor_at_target / factor_at_volume

        adjusted_volume = int(volume * adjustment_factor)

        return AdjustmentResult(
            original_value=float(volume),
            adjusted_value=float(adjusted_volume),
            adjustment_factor=adjustment_factor,
            adjustment_reason="split_adjustment" if adjustment_factor != 1.0 else "no_adjustment",
        )


class DataCleaner:
    """
    Cleans and normalizes market data.

    Features:
    - Remove invalid values (NaN, Inf)
    - Fill data gaps
    - Apply corporate action adjustments
    - Normalize data
    """

    def __init__(
        self,
        corporate_action_manager: CorporateActionManager | None = None,
    ):
        """
        Initialize data cleaner.

        Args:
            corporate_action_manager: Optional manager for corporate action adjustments
        """
        self._ca_manager = corporate_action_manager or CorporateActionManager()

    def clean_ohlcv_bar(
        self,
        bar: dict[str, Any],
        previous_bar: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Clean a single OHLCV bar.

        Args:
            bar: OHLCV bar data
            previous_bar: Previous bar for forward-fill

        Returns:
            Cleaned bar data
        """
        cleaned = bar.copy()

        # Clean price fields
        for field_name in ["open", "high", "low", "close"]:
            value = bar.get(field_name)
            if value is None or math.isnan(value) or math.isinf(value):
                # Try forward-fill from previous bar
                if previous_bar and field_name in previous_bar:
                    cleaned[field_name] = previous_bar[field_name]
                else:
                    cleaned[field_name] = 0.0
                logger.debug(f"Cleaned {field_name}: {value} -> {cleaned[field_name]}")

        # Ensure OHLC relationships
        open_p = cleaned.get("open", 0)
        high = cleaned.get("high", 0)
        low = cleaned.get("low", 0)
        close = cleaned.get("close", 0)

        # Fix high/low relationship
        if low > high:
            cleaned["high"] = low
            cleaned["low"] = high
            high, low = low, high

        # Ensure open is within range
        if open_p > 0 and high > 0:
            if open_p > high:
                cleaned["open"] = high
            elif open_p < low:
                cleaned["open"] = low

        # Ensure close is within range
        if close > 0 and high > 0:
            if close > high:
                cleaned["close"] = high
            elif close < low:
                cleaned["close"] = low

        # Clean volume
        volume = bar.get("volume", 0)
        if volume is None or math.isnan(volume) or math.isinf(volume) or volume < 0:
            cleaned["volume"] = 0

        return cleaned

    def clean_ohlcv_series(
        self,
        data: list[dict[str, Any]],
        fill_gaps: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Clean a series of OHLCV bars.

        Args:
            data: List of OHLCV bars
            fill_gaps: Whether to forward-fill gaps

        Returns:
            List of cleaned bars
        """
        if not data:
            return []

        cleaned = []
        previous_bar = None

        for bar in data:
            cleaned_bar = self.clean_ohlcv_bar(
                bar,
                previous_bar if fill_gaps else None,
            )
            cleaned.append(cleaned_bar)
            previous_bar = cleaned_bar

        return cleaned

    def adjust_for_splits(
        self,
        symbol: str,
        data: list[dict[str, Any]],
        target_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Adjust OHLCV series for stock splits.

        Args:
            symbol: Instrument symbol
            data: List of OHLCV bars with 'timestamp' field
            target_date: Date to adjust to (default: now)

        Returns:
            Split-adjusted data
        """
        target_date = target_date or datetime.now(timezone.utc)
        adjusted = []

        for bar in data:
            bar_date = self._parse_timestamp(bar.get("timestamp"))
            if bar_date is None:
                adjusted.append(bar)
                continue

            adjusted_bar = bar.copy()

            # Adjust prices
            for field_name in ["open", "high", "low", "close"]:
                if field_name in bar and bar[field_name] is not None:
                    result = self._ca_manager.adjust_price_for_splits(
                        symbol,
                        bar[field_name],
                        bar_date,
                        target_date,
                    )
                    adjusted_bar[field_name] = result.adjusted_value

            # Adjust volume
            if "volume" in bar and bar["volume"] is not None:
                result = self._ca_manager.adjust_volume_for_splits(
                    symbol,
                    int(bar["volume"]),
                    bar_date,
                    target_date,
                )
                adjusted_bar["volume"] = int(result.adjusted_value)

            adjusted.append(adjusted_bar)

        return adjusted

    def adjust_for_dividends(
        self,
        symbol: str,
        data: list[dict[str, Any]],
        target_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Adjust OHLCV series for dividends.

        Args:
            symbol: Instrument symbol
            data: List of OHLCV bars with 'timestamp' field
            target_date: Date to adjust to (default: now)

        Returns:
            Dividend-adjusted data
        """
        target_date = target_date or datetime.now(timezone.utc)
        adjusted = []

        for bar in data:
            bar_date = self._parse_timestamp(bar.get("timestamp"))
            if bar_date is None:
                adjusted.append(bar)
                continue

            adjusted_bar = bar.copy()

            # Adjust prices
            for field_name in ["open", "high", "low", "close"]:
                if field_name in bar and bar[field_name] is not None:
                    result = self._ca_manager.adjust_price_for_dividends(
                        symbol,
                        bar[field_name],
                        bar_date,
                        target_date,
                    )
                    adjusted_bar[field_name] = result.adjusted_value

            adjusted.append(adjusted_bar)

        return adjusted

    def fully_adjusted_series(
        self,
        symbol: str,
        data: list[dict[str, Any]],
        target_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Apply all adjustments (splits, dividends) to a series.

        Args:
            symbol: Instrument symbol
            data: List of OHLCV bars
            target_date: Date to adjust to

        Returns:
            Fully adjusted data
        """
        # First clean the data
        cleaned = self.clean_ohlcv_series(data)

        # Then adjust for splits
        split_adjusted = self.adjust_for_splits(symbol, cleaned, target_date)

        # Finally adjust for dividends
        fully_adjusted = self.adjust_for_dividends(symbol, split_adjusted, target_date)

        return fully_adjusted

    def fill_gaps(
        self,
        data: list[dict[str, Any]],
        expected_interval_seconds: int = 86400,
        method: str = "forward_fill",
    ) -> list[dict[str, Any]]:
        """
        Fill gaps in time series data.

        Args:
            data: List of OHLCV bars with 'timestamp' field
            expected_interval_seconds: Expected interval between bars
            method: Fill method: 'forward_fill', 'interpolate', 'zero'

        Returns:
            Data with gaps filled
        """
        if not data or len(data) < 2:
            return data

        filled = [data[0].copy()]

        for i in range(1, len(data)):
            prev_ts = self._parse_timestamp(data[i - 1].get("timestamp"))
            curr_ts = self._parse_timestamp(data[i].get("timestamp"))

            if prev_ts is None or curr_ts is None:
                filled.append(data[i].copy())
                continue

            gap_seconds = (curr_ts - prev_ts).total_seconds()
            missing_periods = int(gap_seconds / expected_interval_seconds) - 1

            # Fill missing periods
            for j in range(missing_periods):
                fill_ts = prev_ts.timestamp() + (j + 1) * expected_interval_seconds
                fill_bar = self._create_fill_bar(
                    data[i - 1],
                    data[i],
                    fill_ts,
                    j / (missing_periods + 1) if method == "interpolate" else 0,
                    method,
                )
                filled.append(fill_bar)

            filled.append(data[i].copy())

        return filled

    def _create_fill_bar(
        self,
        prev_bar: dict[str, Any],
        next_bar: dict[str, Any],
        timestamp: float,
        interpolation_factor: float,
        method: str,
    ) -> dict[str, Any]:
        """Create a fill bar for a gap."""
        fill_bar = {
            "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
            "volume": 0,  # Gaps have no volume
        }

        for field_name in ["open", "high", "low", "close"]:
            prev_val = prev_bar.get(field_name, 0)
            next_val = next_bar.get(field_name, 0)

            if method == "forward_fill":
                fill_bar[field_name] = prev_val
            elif method == "interpolate":
                fill_bar[field_name] = prev_val + (next_val - prev_val) * interpolation_factor
            elif method == "zero":
                fill_bar[field_name] = 0
            else:
                fill_bar[field_name] = prev_val

        return fill_bar

    def _parse_timestamp(self, ts: Any) -> datetime | None:
        """Parse timestamp to datetime."""
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return None
        if isinstance(ts, (int, float)):
            try:
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, OSError):
                return None
        return None


class PriceNormalizer:
    """
    Normalizes prices for comparison across different instruments.
    """

    @staticmethod
    def to_returns(
        prices: list[float],
        log_returns: bool = False,
    ) -> list[float]:
        """
        Convert prices to returns.

        Args:
            prices: List of prices
            log_returns: Calculate log returns instead of simple returns

        Returns:
            List of returns (one less than input)
        """
        if len(prices) < 2:
            return []

        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] <= 0:
                returns.append(0.0)
                continue

            if log_returns:
                if prices[i] <= 0:
                    returns.append(0.0)
                else:
                    returns.append(math.log(prices[i] / prices[i - 1]))
            else:
                returns.append((prices[i] - prices[i - 1]) / prices[i - 1])

        return returns

    @staticmethod
    def normalize_to_base(
        prices: list[float],
        base_value: float = 100.0,
    ) -> list[float]:
        """
        Normalize prices to a base value (rebased series).

        Args:
            prices: List of prices
            base_value: Value to rebase to

        Returns:
            Rebased prices starting at base_value
        """
        if not prices or prices[0] <= 0:
            return prices

        first_price = prices[0]
        return [p / first_price * base_value for p in prices]

    @staticmethod
    def zscore_normalize(
        values: list[float],
    ) -> list[float]:
        """
        Normalize values using z-score normalization.

        Args:
            values: List of values

        Returns:
            Z-score normalized values
        """
        if len(values) < 2:
            return values

        # Filter out NaN/Inf
        valid_values = [v for v in values if not (math.isnan(v) or math.isinf(v))]
        if len(valid_values) < 2:
            return values

        mean = sum(valid_values) / len(valid_values)
        variance = sum((v - mean) ** 2 for v in valid_values) / len(valid_values)
        std = math.sqrt(variance) if variance > 0 else 1.0

        if std == 0:
            return [0.0] * len(values)

        return [(v - mean) / std if not (math.isnan(v) or math.isinf(v)) else 0.0 for v in values]

    @staticmethod
    def minmax_normalize(
        values: list[float],
        target_min: float = 0.0,
        target_max: float = 1.0,
    ) -> list[float]:
        """
        Normalize values using min-max normalization.

        Args:
            values: List of values
            target_min: Target minimum value
            target_max: Target maximum value

        Returns:
            Min-max normalized values
        """
        if len(values) < 2:
            return values

        # Filter out NaN/Inf
        valid_values = [v for v in values if not (math.isnan(v) or math.isinf(v))]
        if len(valid_values) < 2:
            return values

        min_val = min(valid_values)
        max_val = max(valid_values)
        range_val = max_val - min_val

        if range_val == 0:
            return [(target_min + target_max) / 2] * len(values)

        target_range = target_max - target_min
        return [
            target_min + (v - min_val) / range_val * target_range
            if not (math.isnan(v) or math.isinf(v))
            else target_min
            for v in values
        ]

"""
Data Validation Module
======================

Provides comprehensive data quality validation for market data.

Features:
- NaN/Inf detection
- Outlier detection
- Data freshness checks
- Price reasonability validation
- Volume anomaly detection

P3 Enhancement: Data quality validation for robust trading operations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class DataQualityIssue(str, Enum):
    """Types of data quality issues."""
    NAN_VALUE = "nan_value"
    INF_VALUE = "inf_value"
    NEGATIVE_PRICE = "negative_price"
    ZERO_PRICE = "zero_price"
    STALE_DATA = "stale_data"
    OUTLIER_PRICE = "outlier_price"
    OUTLIER_VOLUME = "outlier_volume"
    CROSSED_QUOTES = "crossed_quotes"
    WIDE_SPREAD = "wide_spread"
    NEGATIVE_SPREAD = "negative_spread"
    DATA_GAP = "data_gap"
    INVALID_TIMESTAMP = "invalid_timestamp"
    MISSING_FIELD = "missing_field"


@dataclass
class DataValidationAlert:
    """Single data validation alert."""
    issue_type: DataQualityIssue
    symbol: str
    field_name: str
    value: Any
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: str = "warning"  # "info", "warning", "error"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.symbol}.{self.field_name}: {self.message}"


@dataclass
class DataValidationResult:
    """Result of data validation."""
    valid: bool
    alerts: list[DataValidationAlert] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_alert(
        self,
        issue_type: DataQualityIssue,
        symbol: str,
        field_name: str,
        value: Any,
        message: str,
        severity: str = "warning",
    ) -> None:
        """Add a validation alert."""
        self.alerts.append(DataValidationAlert(
            issue_type=issue_type,
            symbol=symbol,
            field_name=field_name,
            value=value,
            message=message,
            severity=severity,
        ))
        if severity == "error":
            self.valid = False

    def has_issues(self) -> bool:
        """Check if any issues were found."""
        return len(self.alerts) > 0

    def get_errors(self) -> list[DataValidationAlert]:
        """Get only error-level alerts."""
        return [a for a in self.alerts if a.severity == "error"]

    def get_warnings(self) -> list[DataValidationAlert]:
        """Get only warning-level alerts."""
        return [a for a in self.alerts if a.severity == "warning"]


class MarketDataValidator:
    """
    Validates market data for quality issues.

    Detects:
    - NaN/Inf values
    - Outlier prices and volumes
    - Stale data
    - Crossed or unreasonable quotes
    """

    def __init__(
        self,
        max_price_change_pct: float = 20.0,
        max_spread_pct: float = 5.0,
        stale_threshold_seconds: float = 60.0,
        min_volume: int = 0,
        max_volume_zscore: float = 5.0,
    ):
        """
        Initialize validator with thresholds.

        Args:
            max_price_change_pct: Max allowed price change from previous (%)
            max_spread_pct: Max allowed bid-ask spread (%)
            stale_threshold_seconds: Data older than this is stale
            min_volume: Minimum valid volume
            max_volume_zscore: Max volume z-score before flagging outlier
        """
        self._max_price_change_pct = max_price_change_pct
        self._max_spread_pct = max_spread_pct
        self._stale_threshold = timedelta(seconds=stale_threshold_seconds)
        self._min_volume = min_volume
        self._max_volume_zscore = max_volume_zscore

        # Track historical data for outlier detection
        self._price_history: dict[str, list[float]] = {}
        self._volume_history: dict[str, list[int]] = {}
        self._last_update: dict[str, datetime] = {}
        self._history_size = 100

    def validate_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        last: float,
        volume: int,
        timestamp: datetime | None = None,
    ) -> DataValidationResult:
        """
        Validate a market quote.

        Args:
            symbol: Instrument symbol
            bid: Bid price
            ask: Ask price
            last: Last traded price
            volume: Trading volume
            timestamp: Quote timestamp

        Returns:
            DataValidationResult with any issues found
        """
        result = DataValidationResult(valid=True)
        now = datetime.now(timezone.utc)
        timestamp = timestamp or now

        # Check for NaN/Inf values
        self._check_nan_inf(result, symbol, "bid", bid)
        self._check_nan_inf(result, symbol, "ask", ask)
        self._check_nan_inf(result, symbol, "last", last)
        self._check_nan_inf(result, symbol, "volume", float(volume))

        # Check for invalid prices
        self._check_price_validity(result, symbol, "bid", bid)
        self._check_price_validity(result, symbol, "ask", ask)
        self._check_price_validity(result, symbol, "last", last)

        # Check for crossed or wide spreads
        if bid > 0 and ask > 0:
            self._check_spread(result, symbol, bid, ask)

        # Check for stale data
        self._check_staleness(result, symbol, timestamp, now)

        # Check for price outliers
        if last > 0 and not math.isnan(last) and not math.isinf(last):
            self._check_price_outlier(result, symbol, last)

        # Check volume
        self._check_volume(result, symbol, volume)

        # Update history
        self._update_history(symbol, last, volume, timestamp)

        return result

    def _check_nan_inf(
        self,
        result: DataValidationResult,
        symbol: str,
        field_name: str,
        value: float,
    ) -> None:
        """Check for NaN or Inf values."""
        if math.isnan(value):
            result.add_alert(
                DataQualityIssue.NAN_VALUE,
                symbol,
                field_name,
                value,
                f"NaN value detected in {field_name}",
                severity="error",
            )
        elif math.isinf(value):
            result.add_alert(
                DataQualityIssue.INF_VALUE,
                symbol,
                field_name,
                value,
                f"Infinite value detected in {field_name}",
                severity="error",
            )

    def _check_price_validity(
        self,
        result: DataValidationResult,
        symbol: str,
        field_name: str,
        price: float,
    ) -> None:
        """Check basic price validity."""
        if price < 0:
            result.add_alert(
                DataQualityIssue.NEGATIVE_PRICE,
                symbol,
                field_name,
                price,
                f"Negative price detected: {price}",
                severity="error",
            )
        # Note: Zero prices are sometimes valid (e.g., certain derivatives)
        # Log as warning, not error
        elif price == 0:
            result.add_alert(
                DataQualityIssue.ZERO_PRICE,
                symbol,
                field_name,
                price,
                "Zero price detected",
                severity="warning",
            )

    def _check_spread(
        self,
        result: DataValidationResult,
        symbol: str,
        bid: float,
        ask: float,
    ) -> None:
        """Check bid-ask spread validity."""
        if bid > ask:
            result.add_alert(
                DataQualityIssue.CROSSED_QUOTES,
                symbol,
                "spread",
                f"{bid}/{ask}",
                f"Crossed quotes: bid {bid} > ask {ask}",
                severity="error",
            )
        elif ask > 0:
            spread_pct = ((ask - bid) / ask) * 100
            if spread_pct > self._max_spread_pct:
                result.add_alert(
                    DataQualityIssue.WIDE_SPREAD,
                    symbol,
                    "spread",
                    spread_pct,
                    f"Wide spread: {spread_pct:.2f}% (threshold: {self._max_spread_pct}%)",
                    severity="warning",
                )

    def _check_staleness(
        self,
        result: DataValidationResult,
        symbol: str,
        timestamp: datetime,
        now: datetime,
    ) -> None:
        """Check for stale data."""
        age = now - timestamp
        if age > self._stale_threshold:
            result.add_alert(
                DataQualityIssue.STALE_DATA,
                symbol,
                "timestamp",
                timestamp.isoformat(),
                f"Data is {age.total_seconds():.1f}s old (threshold: {self._stale_threshold.total_seconds()}s)",
                severity="warning",
            )

        # Check for data gaps
        if symbol in self._last_update:
            gap = timestamp - self._last_update[symbol]
            # Data gap if more than 5x the stale threshold
            if gap > self._stale_threshold * 5:
                result.add_alert(
                    DataQualityIssue.DATA_GAP,
                    symbol,
                    "timestamp",
                    gap.total_seconds(),
                    f"Data gap detected: {gap.total_seconds():.1f}s since last update",
                    severity="warning",
                )

    def _check_price_outlier(
        self,
        result: DataValidationResult,
        symbol: str,
        price: float,
    ) -> None:
        """Check for price outliers based on historical data."""
        history = self._price_history.get(symbol, [])
        if len(history) < 5:
            return  # Need sufficient history

        # Calculate rolling average
        avg_price = sum(history[-20:]) / min(len(history), 20)
        if avg_price <= 0:
            return

        change_pct = abs((price - avg_price) / avg_price) * 100
        if change_pct > self._max_price_change_pct:
            result.add_alert(
                DataQualityIssue.OUTLIER_PRICE,
                symbol,
                "last",
                price,
                f"Price outlier: {change_pct:.2f}% change from average {avg_price:.4f} (threshold: {self._max_price_change_pct}%)",
                severity="warning",
            )

    def _check_volume(
        self,
        result: DataValidationResult,
        symbol: str,
        volume: int,
    ) -> None:
        """Check volume validity and outliers."""
        if volume < self._min_volume:
            result.add_alert(
                DataQualityIssue.OUTLIER_VOLUME,
                symbol,
                "volume",
                volume,
                f"Volume below minimum: {volume} < {self._min_volume}",
                severity="info",
            )
            return

        history = self._volume_history.get(symbol, [])
        if len(history) < 10:
            return  # Need sufficient history

        # Calculate z-score
        avg_volume = sum(history) / len(history)
        if avg_volume <= 0:
            return

        variance = sum((v - avg_volume) ** 2 for v in history) / len(history)
        std_volume = math.sqrt(variance) if variance > 0 else 1.0

        if std_volume > 0:
            z_score = (volume - avg_volume) / std_volume
            if abs(z_score) > self._max_volume_zscore:
                result.add_alert(
                    DataQualityIssue.OUTLIER_VOLUME,
                    symbol,
                    "volume",
                    volume,
                    f"Volume outlier: z-score={z_score:.2f} (threshold: +/-{self._max_volume_zscore})",
                    severity="info",
                )

    def _update_history(
        self,
        symbol: str,
        price: float,
        volume: int,
        timestamp: datetime,
    ) -> None:
        """Update historical data for outlier detection."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        if symbol not in self._volume_history:
            self._volume_history[symbol] = []

        # Only add valid prices
        if price > 0 and not math.isnan(price) and not math.isinf(price):
            self._price_history[symbol].append(price)
            if len(self._price_history[symbol]) > self._history_size:
                self._price_history[symbol] = self._price_history[symbol][-self._history_size:]

        # Only add valid volumes
        if volume >= 0:
            self._volume_history[symbol].append(volume)
            if len(self._volume_history[symbol]) > self._history_size:
                self._volume_history[symbol] = self._volume_history[symbol][-self._history_size:]

        self._last_update[symbol] = timestamp

    def get_data_freshness(self, symbol: str) -> float | None:
        """
        Get data freshness for a symbol (seconds since last update).

        Returns None if no data has been received.
        """
        if symbol not in self._last_update:
            return None

        age = datetime.now(timezone.utc) - self._last_update[symbol]
        return age.total_seconds()

    def is_data_fresh(self, symbol: str) -> bool:
        """Check if data for a symbol is fresh."""
        freshness = self.get_data_freshness(symbol)
        if freshness is None:
            return False
        return freshness <= self._stale_threshold.total_seconds()

    def get_stale_symbols(self) -> list[str]:
        """Get list of symbols with stale data."""
        return [
            symbol for symbol in self._last_update
            if not self.is_data_fresh(symbol)
        ]

    def clear_history(self, symbol: str | None = None) -> None:
        """Clear historical data for one or all symbols."""
        if symbol:
            self._price_history.pop(symbol, None)
            self._volume_history.pop(symbol, None)
            self._last_update.pop(symbol, None)
        else:
            self._price_history.clear()
            self._volume_history.clear()
            self._last_update.clear()


class TimeSeriesValidator:
    """
    Validates time series data for quality issues.

    Useful for historical data validation before backtesting.
    """

    def __init__(
        self,
        max_gap_periods: int = 5,
        max_price_change_pct: float = 50.0,
    ):
        """
        Initialize time series validator.

        Args:
            max_gap_periods: Max allowed gap in time series
            max_price_change_pct: Max allowed period-over-period change
        """
        self._max_gap_periods = max_gap_periods
        self._max_price_change_pct = max_price_change_pct

    def validate_ohlcv(
        self,
        symbol: str,
        data: list[dict[str, Any]],
        expected_interval_seconds: int = 86400,
    ) -> DataValidationResult:
        """
        Validate OHLCV time series data.

        Args:
            symbol: Instrument symbol
            data: List of OHLCV bars with keys: timestamp, open, high, low, close, volume
            expected_interval_seconds: Expected interval between bars

        Returns:
            DataValidationResult with any issues found
        """
        result = DataValidationResult(valid=True)

        if not data:
            result.add_alert(
                DataQualityIssue.MISSING_FIELD,
                symbol,
                "data",
                None,
                "Empty data set",
                severity="error",
            )
            return result

        prev_bar = None
        prev_timestamp = None

        for i, bar in enumerate(data):
            # Check required fields
            for field_name in ["timestamp", "open", "high", "low", "close"]:
                if field_name not in bar or bar[field_name] is None:
                    result.add_alert(
                        DataQualityIssue.MISSING_FIELD,
                        symbol,
                        field_name,
                        None,
                        f"Missing {field_name} at index {i}",
                        severity="error",
                    )
                    continue

            # Get values
            try:
                open_p = float(bar.get("open", 0))
                high = float(bar.get("high", 0))
                low = float(bar.get("low", 0))
                close = float(bar.get("close", 0))
                volume = int(bar.get("volume", 0))

                # Parse timestamp
                ts = bar.get("timestamp")
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, TypeError) as e:
                result.add_alert(
                    DataQualityIssue.INVALID_TIMESTAMP,
                    symbol,
                    "bar",
                    bar,
                    f"Invalid bar data at index {i}: {e}",
                    severity="error",
                )
                continue

            # Check for NaN/Inf
            for field_name, value in [("open", open_p), ("high", high), ("low", low), ("close", close)]:
                if math.isnan(value):
                    result.add_alert(
                        DataQualityIssue.NAN_VALUE,
                        symbol,
                        field_name,
                        value,
                        f"NaN in {field_name} at index {i}",
                        severity="error",
                    )
                elif math.isinf(value):
                    result.add_alert(
                        DataQualityIssue.INF_VALUE,
                        symbol,
                        field_name,
                        value,
                        f"Inf in {field_name} at index {i}",
                        severity="error",
                    )

            # Check OHLC relationship
            if not (math.isnan(low) or math.isnan(high)):
                if low > high:
                    result.add_alert(
                        DataQualityIssue.OUTLIER_PRICE,
                        symbol,
                        "low/high",
                        f"{low}/{high}",
                        f"Low > High at index {i}: {low} > {high}",
                        severity="error",
                    )
                if open_p > 0 and (open_p < low or open_p > high):
                    result.add_alert(
                        DataQualityIssue.OUTLIER_PRICE,
                        symbol,
                        "open",
                        open_p,
                        f"Open outside high/low range at index {i}",
                        severity="warning",
                    )
                if close > 0 and (close < low or close > high):
                    result.add_alert(
                        DataQualityIssue.OUTLIER_PRICE,
                        symbol,
                        "close",
                        close,
                        f"Close outside high/low range at index {i}",
                        severity="warning",
                    )

            # Check for price outliers vs previous bar
            if prev_bar is not None:
                prev_close = float(prev_bar.get("close", 0))
                if prev_close > 0 and close > 0:
                    change_pct = abs((close - prev_close) / prev_close) * 100
                    if change_pct > self._max_price_change_pct:
                        result.add_alert(
                            DataQualityIssue.OUTLIER_PRICE,
                            symbol,
                            "close",
                            close,
                            f"Large price change at index {i}: {change_pct:.2f}%",
                            severity="warning",
                        )

            # Check for time gaps
            if prev_timestamp is not None:
                gap = (ts - prev_timestamp).total_seconds()
                expected_gap = expected_interval_seconds
                if gap > expected_gap * self._max_gap_periods:
                    result.add_alert(
                        DataQualityIssue.DATA_GAP,
                        symbol,
                        "timestamp",
                        gap,
                        f"Time gap at index {i}: {gap}s (expected ~{expected_gap}s)",
                        severity="warning",
                    )

            prev_bar = bar
            prev_timestamp = ts

        return result


def validate_numeric_value(
    value: Any,
    field_name: str,
    allow_nan: bool = False,
    allow_inf: bool = False,
    allow_negative: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[bool, str | None]:
    """
    Validate a single numeric value.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        allow_nan: Allow NaN values
        allow_inf: Allow infinite values
        allow_negative: Allow negative values
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        Tuple of (is_valid, error_message)
    """
    if value is None:
        return False, f"{field_name} is None"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False, f"{field_name} is not numeric: {value}"

    if not allow_nan and math.isnan(numeric_value):
        return False, f"{field_name} is NaN"

    if not allow_inf and math.isinf(numeric_value):
        return False, f"{field_name} is infinite"

    if not allow_negative and numeric_value < 0:
        return False, f"{field_name} is negative: {numeric_value}"

    if min_value is not None and numeric_value < min_value:
        return False, f"{field_name} below minimum: {numeric_value} < {min_value}"

    if max_value is not None and numeric_value > max_value:
        return False, f"{field_name} above maximum: {numeric_value} > {max_value}"

    return True, None


def sanitize_price(
    value: float,
    fallback: float = 0.0,
    max_value: float = 1e12,
) -> float:
    """
    Sanitize a price value, replacing invalid values with fallback.

    Args:
        value: Price value to sanitize
        fallback: Value to use if invalid
        max_value: Maximum reasonable price

    Returns:
        Sanitized price value
    """
    if math.isnan(value) or math.isinf(value):
        return fallback
    if value < 0:
        return fallback
    if value > max_value:
        return fallback
    return value


def detect_outliers_zscore(
    values: list[float],
    threshold: float = 3.0,
) -> list[int]:
    """
    Detect outliers in a list of values using z-score.

    Args:
        values: List of numeric values
        threshold: Z-score threshold for outlier detection

    Returns:
        List of indices that are outliers
    """
    if len(values) < 3:
        return []

    # Filter out NaN/Inf
    valid_values = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    if len(valid_values) < 3:
        return []

    mean = sum(valid_values) / len(valid_values)
    variance = sum((v - mean) ** 2 for v in valid_values) / len(valid_values)
    std = math.sqrt(variance) if variance > 0 else 1.0

    if std == 0:
        return []

    outliers = []
    for i, v in enumerate(values):
        if math.isnan(v) or math.isinf(v):
            outliers.append(i)
        elif abs((v - mean) / std) > threshold:
            outliers.append(i)

    return outliers


def detect_outliers_iqr(
    values: list[float],
    multiplier: float = 1.5,
) -> list[int]:
    """
    Detect outliers in a list of values using IQR method.

    Args:
        values: List of numeric values
        multiplier: IQR multiplier for outlier bounds

    Returns:
        List of indices that are outliers
    """
    if len(values) < 4:
        return []

    # Filter and sort valid values
    valid_values = sorted([v for v in values if not (math.isnan(v) or math.isinf(v))])
    if len(valid_values) < 4:
        return []

    n = len(valid_values)
    q1_idx = n // 4
    q3_idx = (3 * n) // 4

    q1 = valid_values[q1_idx]
    q3 = valid_values[q3_idx]
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outliers = []
    for i, v in enumerate(values):
        if math.isnan(v) or math.isinf(v):
            outliers.append(i)
        elif v < lower_bound or v > upper_bound:
            outliers.append(i)

    return outliers

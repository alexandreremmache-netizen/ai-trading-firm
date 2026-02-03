"""
LLM Analytics
=============

Track LLM usage, costs, and performance for the AI Trading Firm.

Features:
- Request tracking with full metadata (provider, model, tokens, cost, latency)
- Usage statistics aggregation by purpose, model, and time period
- Cost calculation per model (Claude and GPT pricing)
- Rate limit tracking and warnings
- Quality metrics: sentiment accuracy, forecast accuracy
- Export for cost dashboards and charts
- WebSocket-ready export to dict
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class LLMPurpose(Enum):
    """Purpose categories for LLM requests."""
    SENTIMENT = "sentiment"           # Sentiment analysis of news/text
    FORECAST = "forecast"             # Price/market forecasting
    ANALYSIS = "analysis"             # General market analysis
    SUMMARIZATION = "summarization"   # Text summarization
    CLASSIFICATION = "classification" # Market regime classification
    OTHER = "other"


# Model pricing per 1M tokens (USD) - as of 2024
# Input tokens / Output tokens
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude models
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    # OpenAI GPT models
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-16k": {"input": 3.00, "output": 4.00},
}

# Default pricing for unknown models
DEFAULT_PRICING = {"input": 1.00, "output": 3.00}

# Rate limits per provider (requests per minute)
RATE_LIMITS: dict[str, dict[str, int]] = {
    "anthropic": {
        "requests_per_minute": 60,
        "tokens_per_minute": 100000,
    },
    "openai": {
        "requests_per_minute": 500,
        "tokens_per_minute": 200000,
    },
}


@dataclass
class LLMRequest:
    """
    Record of a single LLM API request.

    Captures comprehensive metadata for tracking and analytics.
    """
    request_id: str
    timestamp: datetime
    provider: LLMProvider
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: Decimal
    latency_ms: float
    purpose: LLMPurpose
    success: bool
    # Optional fields
    symbol: str | None = None
    error_message: str | None = None
    cached: bool = False
    # Quality metrics (filled in later when ground truth available)
    predicted_value: float | None = None  # e.g., sentiment score
    actual_value: float | None = None     # e.g., actual price movement
    accuracy_score: float | None = None   # Computed accuracy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider.value,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": float(self.cost_usd),
            "latency_ms": round(self.latency_ms, 2),
            "purpose": self.purpose.value,
            "success": self.success,
            "symbol": self.symbol,
            "error_message": self.error_message,
            "cached": self.cached,
            "predicted_value": self.predicted_value,
            "actual_value": self.actual_value,
            "accuracy_score": self.accuracy_score,
        }


@dataclass
class LLMUsageStats:
    """
    Aggregated LLM usage statistics.

    Provides overview metrics for monitoring and cost tracking.
    """
    total_requests: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: Decimal = field(default_factory=lambda: Decimal("0.00"))
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    success_rate: float = 1.0
    cache_hit_rate: float = 0.0
    requests_by_purpose: dict[str, int] = field(default_factory=dict)
    requests_by_model: dict[str, int] = field(default_factory=dict)
    requests_by_provider: dict[str, int] = field(default_factory=dict)
    tokens_by_model: dict[str, int] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    cost_by_purpose: dict[str, float] = field(default_factory=dict)
    # Time-based metrics
    requests_last_minute: int = 0
    requests_last_hour: int = 0
    requests_last_day: int = 0
    cost_last_hour: float = 0.0
    cost_last_day: float = 0.0
    # Quality metrics
    avg_accuracy: float | None = None
    sentiment_accuracy: float | None = None
    forecast_accuracy: float | None = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost": float(self.total_cost),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "success_rate": round(self.success_rate, 4),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "requests_by_purpose": self.requests_by_purpose,
            "requests_by_model": self.requests_by_model,
            "requests_by_provider": self.requests_by_provider,
            "tokens_by_model": self.tokens_by_model,
            "cost_by_model": {k: round(v, 4) for k, v in self.cost_by_model.items()},
            "cost_by_purpose": {k: round(v, 4) for k, v in self.cost_by_purpose.items()},
            "requests_last_minute": self.requests_last_minute,
            "requests_last_hour": self.requests_last_hour,
            "requests_last_day": self.requests_last_day,
            "cost_last_hour": round(self.cost_last_hour, 4),
            "cost_last_day": round(self.cost_last_day, 4),
            "avg_accuracy": round(self.avg_accuracy, 4) if self.avg_accuracy is not None else None,
            "sentiment_accuracy": round(self.sentiment_accuracy, 4) if self.sentiment_accuracy is not None else None,
            "forecast_accuracy": round(self.forecast_accuracy, 4) if self.forecast_accuracy is not None else None,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class RateLimitStatus:
    """
    Current rate limit status for a provider.

    Tracks usage against limits and provides warnings.
    """
    provider: str
    requests_per_minute_limit: int
    tokens_per_minute_limit: int
    requests_current: int = 0
    tokens_current: int = 0
    requests_remaining: int = 0
    tokens_remaining: int = 0
    is_rate_limited: bool = False
    reset_time: datetime | None = None
    warning_threshold_pct: float = 80.0
    is_warning: bool = False
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "provider": self.provider,
            "requests_per_minute_limit": self.requests_per_minute_limit,
            "tokens_per_minute_limit": self.tokens_per_minute_limit,
            "requests_current": self.requests_current,
            "tokens_current": self.tokens_current,
            "requests_remaining": self.requests_remaining,
            "tokens_remaining": self.tokens_remaining,
            "requests_usage_pct": round(
                (self.requests_current / self.requests_per_minute_limit) * 100, 1
            ) if self.requests_per_minute_limit > 0 else 0,
            "tokens_usage_pct": round(
                (self.tokens_current / self.tokens_per_minute_limit) * 100, 1
            ) if self.tokens_per_minute_limit > 0 else 0,
            "is_rate_limited": self.is_rate_limited,
            "reset_time": self.reset_time.isoformat() if self.reset_time else None,
            "is_warning": self.is_warning,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class DailyUsage:
    """
    Daily LLM usage summary.

    Aggregated metrics for a single day.
    """
    date: str  # YYYY-MM-DD format
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0
    requests_by_purpose: dict[str, int] = field(default_factory=dict)
    requests_by_model: dict[str, int] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "date": self.date,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "requests_by_purpose": self.requests_by_purpose,
            "requests_by_model": self.requests_by_model,
            "cost_by_model": {k: round(v, 4) for k, v in self.cost_by_model.items()},
        }


class LLMAnalytics:
    """
    Tracks and analyzes LLM usage, costs, and performance.

    Provides comprehensive analytics for monitoring LLM API usage,
    calculating costs, tracking rate limits, and measuring quality.

    Usage:
        analytics = LLMAnalytics()

        # Record a request
        analytics.record_request(
            request_id="req_123",
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-haiku-20240307",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=150.5,
            purpose=LLMPurpose.SENTIMENT,
            success=True,
            symbol="AAPL",
        )

        # Get usage statistics
        stats = analytics.get_usage_stats()

        # Get cost breakdown
        costs = analytics.get_cost_breakdown()

        # Get performance metrics
        perf = analytics.get_performance_metrics()

        # Get daily usage for charts
        daily = analytics.get_daily_usage(days=30)

        # Export for WebSocket streaming
        data = analytics.to_dict()

    Thread Safety:
        Uses asyncio locks for safe concurrent access.
    """

    # Maximum number of requests to keep in circular buffer
    MAX_REQUESTS = 10000

    # Time window for "recent" metrics (seconds)
    RECENT_WINDOW_SECONDS = 60

    def __init__(
        self,
        max_requests: int = MAX_REQUESTS,
        enable_quality_tracking: bool = True,
    ):
        """
        Initialize the LLM analytics tracker.

        Args:
            max_requests: Maximum requests to keep in circular buffer
            enable_quality_tracking: Enable accuracy/quality metric tracking
        """
        self._max_requests = max_requests
        self._enable_quality_tracking = enable_quality_tracking

        # Circular buffer for requests
        self._requests: deque[LLMRequest] = deque(maxlen=max_requests)

        # Rate limit tracking per provider
        self._rate_limit_windows: dict[str, deque[tuple[datetime, int]]] = {
            "anthropic": deque(maxlen=1000),
            "openai": deque(maxlen=1000),
        }

        # Daily usage cache
        self._daily_usage_cache: dict[str, DailyUsage] = {}

        # Quality tracking: request_id -> (predicted, actual)
        self._quality_records: dict[str, dict[str, float | None]] = {}

        # Thread safety
        self._lock = asyncio.Lock()

        # Counters
        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost = Decimal("0.00")
        self._successful_requests = 0
        self._cached_requests = 0
        self._latencies: deque[float] = deque(maxlen=1000)

        logger.info(
            f"LLMAnalytics initialized with max_requests={max_requests}, "
            f"quality_tracking={enable_quality_tracking}"
        )

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Decimal:
        """
        Calculate the cost of an LLM request.

        Args:
            model: Model name
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Cost in USD as Decimal
        """
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)

        input_cost = Decimal(str(prompt_tokens)) * Decimal(str(pricing["input"])) / Decimal("1000000")
        output_cost = Decimal(str(completion_tokens)) * Decimal(str(pricing["output"])) / Decimal("1000000")

        return input_cost + output_cost

    def record_request(
        self,
        request_id: str,
        provider: LLMProvider | str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        purpose: LLMPurpose | str,
        success: bool,
        symbol: str | None = None,
        error_message: str | None = None,
        cached: bool = False,
        predicted_value: float | None = None,
    ) -> LLMRequest:
        """
        Record an LLM API request.

        Args:
            request_id: Unique identifier for the request
            provider: LLM provider (anthropic/openai)
            model: Model name
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            latency_ms: Request latency in milliseconds
            purpose: Purpose category of the request
            success: Whether the request was successful
            symbol: Trading symbol (if applicable)
            error_message: Error message (if failed)
            cached: Whether result was from cache
            predicted_value: Predicted value for quality tracking

        Returns:
            LLMRequest record
        """
        # Normalize enums
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        if isinstance(purpose, str):
            try:
                purpose = LLMPurpose(purpose.lower())
            except ValueError:
                purpose = LLMPurpose.OTHER

        # Calculate cost
        total_tokens = prompt_tokens + completion_tokens
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)

        # Create request record
        request = LLMRequest(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            purpose=purpose,
            success=success,
            symbol=symbol,
            error_message=error_message,
            cached=cached,
            predicted_value=predicted_value,
        )

        # Add to buffer
        self._requests.append(request)

        # Update counters
        self._total_requests += 1
        self._total_tokens += total_tokens
        self._total_cost += cost

        if success:
            self._successful_requests += 1
        if cached:
            self._cached_requests += 1

        self._latencies.append(latency_ms)

        # Update rate limit tracking
        self._rate_limit_windows[provider.value].append(
            (request.timestamp, total_tokens)
        )

        # Update daily usage cache
        self._update_daily_cache(request)

        # Store for quality tracking
        if self._enable_quality_tracking and predicted_value is not None:
            self._quality_records[request_id] = {
                "predicted": predicted_value,
                "actual": None,
                "purpose": purpose.value,
            }

        logger.debug(
            f"Recorded LLM request {request_id[:8]}: "
            f"{model} {total_tokens} tokens ${float(cost):.4f} {latency_ms:.0f}ms"
        )

        return request

    def record_request_sync(
        self,
        request_id: str,
        provider: LLMProvider | str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        purpose: LLMPurpose | str,
        success: bool,
        **kwargs: Any,
    ) -> LLMRequest:
        """
        Synchronous version of record_request for non-async contexts.

        Same parameters as record_request.
        """
        return self.record_request(
            request_id=request_id,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            purpose=purpose,
            success=success,
            **kwargs,
        )

    def _update_daily_cache(self, request: LLMRequest) -> None:
        """Update the daily usage cache with a new request."""
        date_str = request.timestamp.strftime("%Y-%m-%d")

        if date_str not in self._daily_usage_cache:
            self._daily_usage_cache[date_str] = DailyUsage(date=date_str)

        daily = self._daily_usage_cache[date_str]
        daily.total_requests += 1
        daily.total_tokens += request.total_tokens
        daily.total_cost += float(request.cost_usd)

        # Update breakdowns
        purpose_key = request.purpose.value
        daily.requests_by_purpose[purpose_key] = daily.requests_by_purpose.get(purpose_key, 0) + 1

        model_key = request.model
        daily.requests_by_model[model_key] = daily.requests_by_model.get(model_key, 0) + 1
        daily.cost_by_model[model_key] = daily.cost_by_model.get(model_key, 0.0) + float(request.cost_usd)

    def update_actual_value(
        self,
        request_id: str,
        actual_value: float,
    ) -> bool:
        """
        Update the actual value for quality tracking.

        Called when ground truth becomes available (e.g., actual price movement).

        Args:
            request_id: ID of the original request
            actual_value: Actual observed value

        Returns:
            True if updated successfully, False if request not found
        """
        if not self._enable_quality_tracking:
            return False

        if request_id not in self._quality_records:
            return False

        self._quality_records[request_id]["actual"] = actual_value

        # Calculate accuracy
        predicted = self._quality_records[request_id]["predicted"]
        if predicted is not None:
            # Simple accuracy: 1 - normalized error
            # For sentiment: both values in [-1, 1]
            error = abs(predicted - actual_value) / 2.0  # Normalize to [0, 1]
            accuracy = 1.0 - error

            # Update the request record
            for req in self._requests:
                if req.request_id == request_id:
                    req.actual_value = actual_value
                    req.accuracy_score = accuracy
                    break

        logger.debug(f"Updated actual value for request {request_id[:8]}: {actual_value}")
        return True

    def get_usage_stats(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> LLMUsageStats:
        """
        Get aggregated usage statistics.

        Args:
            start_time: Start of time range (default: all time)
            end_time: End of time range (default: now)

        Returns:
            LLMUsageStats with aggregated metrics
        """
        now = datetime.now(timezone.utc)

        if end_time is None:
            end_time = now

        # Filter requests by time range
        requests = [
            r for r in self._requests
            if (start_time is None or r.timestamp >= start_time)
            and r.timestamp <= end_time
        ]

        stats = LLMUsageStats(last_updated=now)

        if not requests:
            return stats

        # Aggregate metrics
        stats.total_requests = len(requests)
        stats.total_tokens = sum(r.total_tokens for r in requests)
        stats.total_prompt_tokens = sum(r.prompt_tokens for r in requests)
        stats.total_completion_tokens = sum(r.completion_tokens for r in requests)
        stats.total_cost = sum((r.cost_usd for r in requests), Decimal("0.00"))

        # Latency stats
        latencies = [r.latency_ms for r in requests]
        stats.avg_latency_ms = sum(latencies) / len(latencies)
        stats.min_latency_ms = min(latencies)
        stats.max_latency_ms = max(latencies)

        # Success and cache rates
        successful = sum(1 for r in requests if r.success)
        cached = sum(1 for r in requests if r.cached)
        stats.success_rate = successful / len(requests) if requests else 1.0
        stats.cache_hit_rate = cached / len(requests) if requests else 0.0

        # Breakdowns
        for r in requests:
            purpose_key = r.purpose.value
            stats.requests_by_purpose[purpose_key] = stats.requests_by_purpose.get(purpose_key, 0) + 1
            stats.cost_by_purpose[purpose_key] = stats.cost_by_purpose.get(purpose_key, 0.0) + float(r.cost_usd)

            model_key = r.model
            stats.requests_by_model[model_key] = stats.requests_by_model.get(model_key, 0) + 1
            stats.tokens_by_model[model_key] = stats.tokens_by_model.get(model_key, 0) + r.total_tokens
            stats.cost_by_model[model_key] = stats.cost_by_model.get(model_key, 0.0) + float(r.cost_usd)

            provider_key = r.provider.value
            stats.requests_by_provider[provider_key] = stats.requests_by_provider.get(provider_key, 0) + 1

        # Time-based counts
        one_minute_ago = now - timedelta(minutes=1)
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)

        stats.requests_last_minute = sum(1 for r in requests if r.timestamp >= one_minute_ago)
        stats.requests_last_hour = sum(1 for r in requests if r.timestamp >= one_hour_ago)
        stats.requests_last_day = sum(1 for r in requests if r.timestamp >= one_day_ago)

        stats.cost_last_hour = sum(
            float(r.cost_usd) for r in requests if r.timestamp >= one_hour_ago
        )
        stats.cost_last_day = sum(
            float(r.cost_usd) for r in requests if r.timestamp >= one_day_ago
        )

        # Quality metrics
        if self._enable_quality_tracking:
            accuracy_scores = [r.accuracy_score for r in requests if r.accuracy_score is not None]
            if accuracy_scores:
                stats.avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)

            sentiment_scores = [
                r.accuracy_score for r in requests
                if r.accuracy_score is not None and r.purpose == LLMPurpose.SENTIMENT
            ]
            if sentiment_scores:
                stats.sentiment_accuracy = sum(sentiment_scores) / len(sentiment_scores)

            forecast_scores = [
                r.accuracy_score for r in requests
                if r.accuracy_score is not None and r.purpose == LLMPurpose.FORECAST
            ]
            if forecast_scores:
                stats.forecast_accuracy = sum(forecast_scores) / len(forecast_scores)

        return stats

    def get_cost_breakdown(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get detailed cost breakdown.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dict with cost breakdown by model, purpose, provider, and time
        """
        now = datetime.now(timezone.utc)

        if end_time is None:
            end_time = now

        # Filter requests
        requests = [
            r for r in self._requests
            if (start_time is None or r.timestamp >= start_time)
            and r.timestamp <= end_time
        ]

        breakdown = {
            "total_cost": 0.0,
            "by_model": {},
            "by_purpose": {},
            "by_provider": {},
            "by_day": {},
            "input_tokens_cost": 0.0,
            "output_tokens_cost": 0.0,
            "avg_cost_per_request": 0.0,
            "avg_cost_per_1k_tokens": 0.0,
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat(),
            },
            "timestamp": now.isoformat(),
        }

        if not requests:
            return breakdown

        total_cost = Decimal("0.00")
        total_input_cost = Decimal("0.00")
        total_output_cost = Decimal("0.00")
        total_tokens = 0

        for r in requests:
            cost = r.cost_usd
            total_cost += cost
            total_tokens += r.total_tokens

            # Estimate input/output cost split
            pricing = MODEL_PRICING.get(r.model, DEFAULT_PRICING)
            input_cost = Decimal(str(r.prompt_tokens)) * Decimal(str(pricing["input"])) / Decimal("1000000")
            output_cost = Decimal(str(r.completion_tokens)) * Decimal(str(pricing["output"])) / Decimal("1000000")
            total_input_cost += input_cost
            total_output_cost += output_cost

            # By model
            model = r.model
            if model not in breakdown["by_model"]:
                breakdown["by_model"][model] = {"cost": 0.0, "requests": 0, "tokens": 0}
            breakdown["by_model"][model]["cost"] += float(cost)
            breakdown["by_model"][model]["requests"] += 1
            breakdown["by_model"][model]["tokens"] += r.total_tokens

            # By purpose
            purpose = r.purpose.value
            if purpose not in breakdown["by_purpose"]:
                breakdown["by_purpose"][purpose] = {"cost": 0.0, "requests": 0}
            breakdown["by_purpose"][purpose]["cost"] += float(cost)
            breakdown["by_purpose"][purpose]["requests"] += 1

            # By provider
            provider = r.provider.value
            if provider not in breakdown["by_provider"]:
                breakdown["by_provider"][provider] = {"cost": 0.0, "requests": 0}
            breakdown["by_provider"][provider]["cost"] += float(cost)
            breakdown["by_provider"][provider]["requests"] += 1

            # By day
            day = r.timestamp.strftime("%Y-%m-%d")
            if day not in breakdown["by_day"]:
                breakdown["by_day"][day] = {"cost": 0.0, "requests": 0}
            breakdown["by_day"][day]["cost"] += float(cost)
            breakdown["by_day"][day]["requests"] += 1

        breakdown["total_cost"] = float(total_cost)
        breakdown["input_tokens_cost"] = float(total_input_cost)
        breakdown["output_tokens_cost"] = float(total_output_cost)
        breakdown["avg_cost_per_request"] = float(total_cost) / len(requests) if requests else 0.0
        breakdown["avg_cost_per_1k_tokens"] = (
            float(total_cost) / (total_tokens / 1000) if total_tokens > 0 else 0.0
        )

        return breakdown

    def get_performance_metrics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get performance metrics.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dict with performance metrics (latency, success rate, quality)
        """
        now = datetime.now(timezone.utc)

        if end_time is None:
            end_time = now

        # Filter requests
        requests = [
            r for r in self._requests
            if (start_time is None or r.timestamp >= start_time)
            and r.timestamp <= end_time
        ]

        metrics = {
            "latency": {
                "avg_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            },
            "success_rate": 1.0,
            "error_rate": 0.0,
            "cache_hit_rate": 0.0,
            "by_model": {},
            "by_purpose": {},
            "quality": {
                "avg_accuracy": None,
                "sentiment_accuracy": None,
                "forecast_accuracy": None,
                "evaluated_requests": 0,
            },
            "errors": {
                "total": 0,
                "by_type": {},
            },
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat(),
            },
            "timestamp": now.isoformat(),
        }

        if not requests:
            return metrics

        # Latency metrics
        latencies = sorted([r.latency_ms for r in requests])
        metrics["latency"]["avg_ms"] = sum(latencies) / len(latencies)
        metrics["latency"]["min_ms"] = latencies[0]
        metrics["latency"]["max_ms"] = latencies[-1]
        metrics["latency"]["p50_ms"] = latencies[len(latencies) // 2]
        metrics["latency"]["p95_ms"] = latencies[int(len(latencies) * 0.95)]
        metrics["latency"]["p99_ms"] = latencies[int(len(latencies) * 0.99)]

        # Success/error rates
        successful = sum(1 for r in requests if r.success)
        cached = sum(1 for r in requests if r.cached)
        metrics["success_rate"] = successful / len(requests)
        metrics["error_rate"] = 1 - metrics["success_rate"]
        metrics["cache_hit_rate"] = cached / len(requests)

        # Errors by type
        metrics["errors"]["total"] = len(requests) - successful
        for r in requests:
            if not r.success and r.error_message:
                error_type = r.error_message.split(":")[0] if ":" in r.error_message else r.error_message
                metrics["errors"]["by_type"][error_type] = metrics["errors"]["by_type"].get(error_type, 0) + 1

        # By model performance
        model_requests: dict[str, list[LLMRequest]] = {}
        for r in requests:
            if r.model not in model_requests:
                model_requests[r.model] = []
            model_requests[r.model].append(r)

        for model, reqs in model_requests.items():
            lats = [r.latency_ms for r in reqs]
            succ = sum(1 for r in reqs if r.success)
            metrics["by_model"][model] = {
                "avg_latency_ms": sum(lats) / len(lats),
                "success_rate": succ / len(reqs),
                "requests": len(reqs),
            }

        # By purpose performance
        purpose_requests: dict[str, list[LLMRequest]] = {}
        for r in requests:
            key = r.purpose.value
            if key not in purpose_requests:
                purpose_requests[key] = []
            purpose_requests[key].append(r)

        for purpose, reqs in purpose_requests.items():
            lats = [r.latency_ms for r in reqs]
            succ = sum(1 for r in reqs if r.success)
            metrics["by_purpose"][purpose] = {
                "avg_latency_ms": sum(lats) / len(lats),
                "success_rate": succ / len(reqs),
                "requests": len(reqs),
            }

        # Quality metrics
        if self._enable_quality_tracking:
            accuracy_scores = [r.accuracy_score for r in requests if r.accuracy_score is not None]
            metrics["quality"]["evaluated_requests"] = len(accuracy_scores)

            if accuracy_scores:
                metrics["quality"]["avg_accuracy"] = sum(accuracy_scores) / len(accuracy_scores)

            sentiment_scores = [
                r.accuracy_score for r in requests
                if r.accuracy_score is not None and r.purpose == LLMPurpose.SENTIMENT
            ]
            if sentiment_scores:
                metrics["quality"]["sentiment_accuracy"] = sum(sentiment_scores) / len(sentiment_scores)

            forecast_scores = [
                r.accuracy_score for r in requests
                if r.accuracy_score is not None and r.purpose == LLMPurpose.FORECAST
            ]
            if forecast_scores:
                metrics["quality"]["forecast_accuracy"] = sum(forecast_scores) / len(forecast_scores)

        return metrics

    def get_daily_usage(self, days: int = 30) -> list[DailyUsage]:
        """
        Get daily usage summaries for charting.

        Args:
            days: Number of days to include (default 30)

        Returns:
            List of DailyUsage objects sorted by date
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days)

        daily_list = []
        for date_str, daily in self._daily_usage_cache.items():
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if date >= cutoff:
                    # Calculate success rate for this day
                    day_requests = [
                        r for r in self._requests
                        if r.timestamp.strftime("%Y-%m-%d") == date_str
                    ]
                    if day_requests:
                        successful = sum(1 for r in day_requests if r.success)
                        daily.success_rate = successful / len(day_requests)
                        latencies = [r.latency_ms for r in day_requests]
                        daily.avg_latency_ms = sum(latencies) / len(latencies)

                    daily_list.append(daily)
            except ValueError:
                continue

        # Sort by date
        daily_list.sort(key=lambda d: d.date)

        return daily_list

    def get_rate_limit_status(self, provider: str | LLMProvider) -> RateLimitStatus:
        """
        Get current rate limit status for a provider.

        Args:
            provider: Provider name or enum

        Returns:
            RateLimitStatus with current usage and limits
        """
        if isinstance(provider, LLMProvider):
            provider = provider.value

        provider = provider.lower()
        limits = RATE_LIMITS.get(provider, {"requests_per_minute": 60, "tokens_per_minute": 100000})

        now = datetime.now(timezone.utc)
        one_minute_ago = now - timedelta(minutes=1)

        # Get requests in the last minute
        window = self._rate_limit_windows.get(provider, deque())
        recent = [(ts, tokens) for ts, tokens in window if ts >= one_minute_ago]

        requests_current = len(recent)
        tokens_current = sum(tokens for _, tokens in recent)

        status = RateLimitStatus(
            provider=provider,
            requests_per_minute_limit=limits["requests_per_minute"],
            tokens_per_minute_limit=limits["tokens_per_minute"],
            requests_current=requests_current,
            tokens_current=tokens_current,
            requests_remaining=max(0, limits["requests_per_minute"] - requests_current),
            tokens_remaining=max(0, limits["tokens_per_minute"] - tokens_current),
            last_updated=now,
        )

        # Check if rate limited
        status.is_rate_limited = (
            requests_current >= limits["requests_per_minute"] or
            tokens_current >= limits["tokens_per_minute"]
        )

        # Check if approaching limit (warning)
        requests_pct = (requests_current / limits["requests_per_minute"]) * 100
        tokens_pct = (tokens_current / limits["tokens_per_minute"]) * 100
        status.is_warning = requests_pct >= status.warning_threshold_pct or tokens_pct >= status.warning_threshold_pct

        # Estimate reset time
        if status.is_rate_limited and recent:
            oldest_request = min(ts for ts, _ in recent)
            status.reset_time = oldest_request + timedelta(minutes=1)

        if status.is_warning:
            logger.warning(
                f"LLM rate limit warning for {provider}: "
                f"{requests_current}/{limits['requests_per_minute']} requests, "
                f"{tokens_current}/{limits['tokens_per_minute']} tokens"
            )

        return status

    def get_recent_requests(self, limit: int = 100) -> list[LLMRequest]:
        """
        Get the most recent requests.

        Args:
            limit: Maximum number of requests to return

        Returns:
            List of LLMRequest, most recent first
        """
        requests = list(self._requests)
        requests.reverse()
        return requests[:limit]

    def get_requests_by_symbol(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[LLMRequest]:
        """
        Get requests for a specific symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of requests to return

        Returns:
            List of matching LLMRequest
        """
        matching = [r for r in self._requests if r.symbol == symbol]
        matching.reverse()
        return matching[:limit]

    def get_requests_by_purpose(
        self,
        purpose: LLMPurpose | str,
        limit: int = 100,
    ) -> list[LLMRequest]:
        """
        Get requests for a specific purpose.

        Args:
            purpose: Purpose category
            limit: Maximum number of requests to return

        Returns:
            List of matching LLMRequest
        """
        if isinstance(purpose, str):
            purpose = LLMPurpose(purpose.lower())

        matching = [r for r in self._requests if r.purpose == purpose]
        matching.reverse()
        return matching[:limit]

    def get_model_comparison(self) -> dict[str, Any]:
        """
        Get comparison metrics across models.

        Returns:
            Dict with model comparison data for charts
        """
        model_data: dict[str, dict[str, Any]] = {}

        for r in self._requests:
            if r.model not in model_data:
                model_data[r.model] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0,
                    "latencies": [],
                    "successes": 0,
                    "provider": r.provider.value,
                }

            data = model_data[r.model]
            data["requests"] += 1
            data["tokens"] += r.total_tokens
            data["cost"] += float(r.cost_usd)
            data["latencies"].append(r.latency_ms)
            if r.success:
                data["successes"] += 1

        # Calculate summary metrics
        comparison = {}
        for model, data in model_data.items():
            comparison[model] = {
                "provider": data["provider"],
                "total_requests": data["requests"],
                "total_tokens": data["tokens"],
                "total_cost": round(data["cost"], 4),
                "avg_latency_ms": round(sum(data["latencies"]) / len(data["latencies"]), 2) if data["latencies"] else 0,
                "success_rate": round(data["successes"] / data["requests"], 4) if data["requests"] > 0 else 1.0,
                "cost_per_1k_tokens": round(data["cost"] / (data["tokens"] / 1000), 4) if data["tokens"] > 0 else 0,
            }

        return comparison

    def export_for_dashboard(self) -> dict[str, Any]:
        """
        Export data formatted for dashboard charts.

        Returns:
            Dict with data ready for charting libraries
        """
        stats = self.get_usage_stats()
        daily = self.get_daily_usage(days=30)
        cost_breakdown = self.get_cost_breakdown()
        model_comparison = self.get_model_comparison()

        return {
            "summary": stats.to_dict(),
            "daily_usage": [d.to_dict() for d in daily],
            "cost_breakdown": cost_breakdown,
            "model_comparison": model_comparison,
            "rate_limits": {
                "anthropic": self.get_rate_limit_status("anthropic").to_dict(),
                "openai": self.get_rate_limit_status("openai").to_dict(),
            },
            "charts": {
                "cost_by_day": [
                    {"date": d.date, "cost": d.total_cost}
                    for d in daily
                ],
                "requests_by_day": [
                    {"date": d.date, "requests": d.total_requests}
                    for d in daily
                ],
                "tokens_by_day": [
                    {"date": d.date, "tokens": d.total_tokens}
                    for d in daily
                ],
                "cost_by_model": [
                    {"model": k, "cost": v}
                    for k, v in stats.cost_by_model.items()
                ],
                "cost_by_purpose": [
                    {"purpose": k, "cost": v}
                    for k, v in stats.cost_by_purpose.items()
                ],
                "requests_by_purpose": [
                    {"purpose": k, "count": v}
                    for k, v in stats.requests_by_purpose.items()
                ],
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def clear(self) -> None:
        """Clear all tracked data and reset counters."""
        self._requests.clear()
        self._daily_usage_cache.clear()
        self._quality_records.clear()
        self._latencies.clear()

        for window in self._rate_limit_windows.values():
            window.clear()

        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost = Decimal("0.00")
        self._successful_requests = 0
        self._cached_requests = 0

        logger.info("LLMAnalytics cleared")

    def to_dict(self) -> dict[str, Any]:
        """
        Export complete tracker state to dictionary for WebSocket streaming.

        Returns:
            Complete tracker state as dict
        """
        stats = self.get_usage_stats()
        recent = self.get_recent_requests(limit=50)

        return {
            "statistics": stats.to_dict(),
            "recent_requests": [r.to_dict() for r in recent],
            "rate_limits": {
                "anthropic": self.get_rate_limit_status("anthropic").to_dict(),
                "openai": self.get_rate_limit_status("openai").to_dict(),
            },
            "cost_breakdown": self.get_cost_breakdown(),
            "model_comparison": self.get_model_comparison(),
            "totals": {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "total_cost": float(self._total_cost),
                "successful_requests": self._successful_requests,
                "cached_requests": self._cached_requests,
            },
            "buffer_size": len(self._requests),
            "buffer_max_size": self._max_requests,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def to_dict_async(self) -> dict[str, Any]:
        """
        Async version of to_dict with proper locking.

        Returns:
            Complete tracker state as dict
        """
        async with self._lock:
            return self.to_dict()

    @property
    def total_requests(self) -> int:
        """Total number of requests recorded."""
        return self._total_requests

    @property
    def total_cost(self) -> Decimal:
        """Total cost in USD."""
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self._total_tokens

    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        if self._total_requests == 0:
            return 1.0
        return self._successful_requests / self._total_requests

    @property
    def buffer_size(self) -> int:
        """Current number of requests in the buffer."""
        return len(self._requests)

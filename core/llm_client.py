"""
LLM Client
==========

Async client for LLM API calls (Anthropic Claude / OpenAI).
Provides sentiment analysis for news and market commentary.

Features:
- Rate limiting to avoid API throttling
- Retry with exponential backoff
- Response caching
- Graceful degradation when API unavailable
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    sentiment: float  # -1.0 (very bearish) to 1.0 (very bullish)
    confidence: float  # 0.0 to 1.0
    rationale: str  # LLM explanation
    symbol: str
    analyzed_at: datetime
    source: str = "llm_sentiment"
    cached: bool = False
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        """Check if result is valid (no error)."""
        return self.error is None


class LRUCache:
    """Simple LRU cache for sentiment results."""

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[str, tuple[SentimentResult, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = 300  # 5 minutes

    def get(self, key: str) -> SentimentResult | None:
        """Get cached result if exists and not expired."""
        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl_seconds:
            del self._cache[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return result

    def set(self, key: str, result: SentimentResult) -> None:
        """Cache a result."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Remove oldest
        self._cache[key] = (result, time.time())


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, calls_per_minute: int = 20):
        self._calls_per_minute = calls_per_minute
        self._min_interval = 60.0 / calls_per_minute
        self._last_call_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until we can make another API call."""
        async with self._lock:
            now = time.time()
            time_since_last = now - self._last_call_time
            if time_since_last < self._min_interval:
                wait_time = self._min_interval - time_since_last
                await asyncio.sleep(wait_time)
            self._last_call_time = time.time()


class LLMClient:
    """
    Async LLM client for sentiment analysis.

    Supports:
    - Anthropic Claude (default)
    - OpenAI GPT (fallback)

    Usage:
        client = LLMClient(config={"llm_provider": "anthropic"})
        result = await client.analyze_sentiment("AAPL earnings beat...", "AAPL")
    """

    # System prompt for sentiment analysis
    SENTIMENT_SYSTEM_PROMPT = """You are a financial sentiment analyzer. Analyze the provided news/text for market sentiment about the specified stock symbol.

Respond with ONLY a JSON object in this exact format:
{
  "sentiment": <float from -1.0 to 1.0>,
  "confidence": <float from 0.0 to 1.0>,
  "rationale": "<brief explanation>"
}

Sentiment scale:
- -1.0: Extremely bearish (major negative news, bankruptcy, fraud)
- -0.5: Moderately bearish (earnings miss, downgrades, layoffs)
- 0.0: Neutral (routine news, no clear direction)
- 0.5: Moderately bullish (earnings beat, upgrades, new products)
- 1.0: Extremely bullish (major positive catalyst, breakthrough)

Confidence represents how clear the sentiment signal is:
- 0.0-0.3: Ambiguous or insufficient information
- 0.4-0.6: Moderate confidence, some mixed signals
- 0.7-0.9: High confidence, clear sentiment
- 1.0: Very high confidence, unambiguous

Be conservative with confidence scores. Only use >0.8 for very clear cases."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}

        self._provider = config.get("llm_provider", "anthropic")
        self._model = config.get("model", self._get_default_model())
        self._timeout = config.get("timeout_seconds", 10)
        self._max_retries = config.get("max_retries", 3)

        # Rate limiting
        calls_per_minute = config.get("calls_per_minute", 20)
        self._rate_limiter = RateLimiter(calls_per_minute)

        # Caching
        cache_size = config.get("cache_size", 100)
        self._cache = LRUCache(max_size=cache_size)

        # API keys from environment
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")

        # Session will be created lazily
        self._session: aiohttp.ClientSession | None = None

        self._initialized = False
        self._api_available = False

    def _get_default_model(self) -> str:
        """Get default model for provider."""
        if self._provider == "anthropic":
            return "claude-3-haiku-20240307"  # Fast and cost-effective
        return "gpt-3.5-turbo"

    async def initialize(self) -> None:
        """Initialize the client."""
        if self._initialized:
            return

        # Check API key availability
        if self._provider == "anthropic" and self._anthropic_key:
            self._api_available = True
            logger.info(f"LLMClient initialized with Anthropic ({self._model})")
        elif self._provider == "openai" and self._openai_key:
            self._api_available = True
            logger.info(f"LLMClient initialized with OpenAI ({self._model})")
        elif self._anthropic_key:
            # Fallback to Anthropic if available
            self._provider = "anthropic"
            self._api_available = True
            logger.info("LLMClient falling back to Anthropic")
        elif self._openai_key:
            # Fallback to OpenAI if available
            self._provider = "openai"
            self._api_available = True
            logger.info("LLMClient falling back to OpenAI")
        else:
            self._api_available = False
            logger.warning(
                "LLMClient: No API key found (ANTHROPIC_API_KEY or OPENAI_API_KEY). "
                "Sentiment analysis will return neutral signals."
            )

        self._initialized = True

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _cache_key(self, text: str, symbol: str) -> str:
        """Generate cache key for sentiment request."""
        content = f"{symbol}:{text[:500]}"  # Truncate for key
        return hashlib.md5(content.encode()).hexdigest()

    async def analyze_sentiment(
        self,
        text: str,
        symbol: str,
        use_cache: bool = True,
    ) -> SentimentResult:
        """
        Analyze sentiment of text for a given symbol.

        Args:
            text: News text or market commentary to analyze
            symbol: Stock/instrument symbol
            use_cache: Whether to use cached results

        Returns:
            SentimentResult with sentiment score, confidence, and rationale
        """
        if not self._initialized:
            await self.initialize()

        # Check cache
        cache_key = self._cache_key(text, symbol)
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached:
                cached.cached = True
                return cached

        # Return neutral if API not available
        if not self._api_available:
            return SentimentResult(
                sentiment=0.0,
                confidence=0.1,
                rationale="LLM API not configured - returning neutral",
                symbol=symbol,
                analyzed_at=datetime.now(timezone.utc),
                error="API not available",
            )

        # Rate limit
        await self._rate_limiter.acquire()

        # Make API call with retries
        last_error: str | None = None
        for attempt in range(self._max_retries):
            try:
                if self._provider == "anthropic":
                    result = await self._call_anthropic(text, symbol)
                else:
                    result = await self._call_openai(text, symbol)

                # Cache successful result
                if result.is_valid and use_cache:
                    self._cache.set(cache_key, result)

                return result

            except asyncio.TimeoutError:
                last_error = "API timeout"
                logger.warning(f"LLM API timeout (attempt {attempt + 1}/{self._max_retries})")
            except aiohttp.ClientError as e:
                last_error = f"API error: {e}"
                logger.warning(f"LLM API error (attempt {attempt + 1}): {e}")
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.exception(f"Unexpected LLM error: {e}")

            # Exponential backoff
            if attempt < self._max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        # All retries failed
        return SentimentResult(
            sentiment=0.0,
            confidence=0.0,
            rationale=f"API call failed after {self._max_retries} attempts",
            symbol=symbol,
            analyzed_at=datetime.now(timezone.utc),
            error=last_error,
        )

    async def _call_anthropic(self, text: str, symbol: str) -> SentimentResult:
        """Call Anthropic Claude API."""
        session = await self._get_session()

        user_prompt = f"Analyze the sentiment of this news for {symbol}:\n\n{text[:2000]}"

        payload = {
            "model": self._model,
            "max_tokens": 200,
            "system": self.SENTIMENT_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._anthropic_key,
            "anthropic-version": "2023-06-01",
        }

        async with session.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers=headers,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise aiohttp.ClientError(f"Anthropic API error {response.status}: {error_text}")

            data = await response.json()
            return self._parse_response(data, symbol, "anthropic")

    async def _call_openai(self, text: str, symbol: str) -> SentimentResult:
        """Call OpenAI API."""
        session = await self._get_session()

        user_prompt = f"Analyze the sentiment of this news for {symbol}:\n\n{text[:2000]}"

        payload = {
            "model": self._model,
            "max_tokens": 200,
            "messages": [
                {"role": "system", "content": self.SENTIMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._openai_key}",
        }

        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise aiohttp.ClientError(f"OpenAI API error {response.status}: {error_text}")

            data = await response.json()
            return self._parse_response(data, symbol, "openai")

    def _parse_response(
        self,
        data: dict[str, Any],
        symbol: str,
        provider: str,
    ) -> SentimentResult:
        """Parse LLM response into SentimentResult."""
        import json

        try:
            # Extract content based on provider
            if provider == "anthropic":
                content = data["content"][0]["text"]
            else:  # openai
                content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            # Handle potential markdown code blocks
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            parsed = json.loads(content)

            sentiment = float(parsed.get("sentiment", 0.0))
            confidence = float(parsed.get("confidence", 0.5))
            rationale = str(parsed.get("rationale", "No rationale provided"))

            # Clamp values to valid ranges
            sentiment = max(-1.0, min(1.0, sentiment))
            confidence = max(0.0, min(1.0, confidence))

            return SentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                rationale=rationale,
                symbol=symbol,
                analyzed_at=datetime.now(timezone.utc),
                source="llm_sentiment",
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return SentimentResult(
                sentiment=0.0,
                confidence=0.0,
                rationale="Failed to parse LLM response",
                symbol=symbol,
                analyzed_at=datetime.now(timezone.utc),
                error=f"Parse error: {e}",
            )

    async def health_check(self) -> bool:
        """Check if LLM API is available."""
        if not self._api_available:
            return False

        try:
            result = await self.analyze_sentiment(
                "Test message for health check",
                "TEST",
                use_cache=False,
            )
            return result.is_valid
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            return False

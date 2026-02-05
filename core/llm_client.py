"""
LLM Client
==========

Async client for LLM API calls (Anthropic Claude / OpenAI / Local LLMs).
Provides sentiment analysis for news and market commentary.

Features:
- Rate limiting to avoid API throttling
- Retry with exponential backoff
- Response caching
- Graceful degradation when API unavailable
- Local LLM support (Ollama, llama.cpp) for free, fast sentiment analysis

Phase E Enhancement: Local LLM backends for cost-free sentiment analysis.
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
from enum import Enum
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


class LLMBackend(Enum):
    """LLM backend options."""
    CLAUDE_API = "claude"      # Anthropic Claude API
    OPENAI_API = "openai"      # OpenAI API
    LOCAL_OLLAMA = "ollama"    # Local Ollama server
    LOCAL_LLAMA_CPP = "llama_cpp"  # Local llama.cpp Python bindings


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
    - Anthropic Claude (API)
    - OpenAI GPT (API)
    - Ollama (local) - Phase E
    - llama.cpp (local) - Phase E

    Usage:
        # API-based (default)
        client = LLMClient(config={"llm_provider": "anthropic"})

        # Local Ollama (free, fast)
        client = LLMClient(config={
            "backend": "ollama",
            "ollama_url": "http://localhost:11434",
            "ollama_model": "llama3:8b"
        })

        # Local llama.cpp (free, very fast with GPU)
        client = LLMClient(config={
            "backend": "llama_cpp",
            "llama_cpp_model_path": "models/llama-3-8b-q4.gguf",
            "llama_cpp_n_gpu_layers": 35
        })

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

    # Compact prompt for local models (shorter context)
    SENTIMENT_PROMPT_LOCAL = """Analyze financial sentiment for {symbol}. Output JSON only:
{{"sentiment": <-1.0 to 1.0>, "confidence": <0.0 to 1.0>, "rationale": "<brief>"}}

Text: {text}"""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}

        # Determine backend (Phase E: support local backends)
        backend_str = config.get("backend", config.get("llm_provider", "anthropic"))
        self._backend = self._parse_backend(backend_str)

        # Legacy provider mapping
        if self._backend in (LLMBackend.CLAUDE_API, LLMBackend.OPENAI_API):
            self._provider = "anthropic" if self._backend == LLMBackend.CLAUDE_API else "openai"
        else:
            self._provider = backend_str

        self._model = config.get("model", self._get_default_model())
        self._timeout = config.get("timeout_seconds", 10)
        self._max_retries = config.get("max_retries", 3)

        # Rate limiting (lower for local models)
        if self._backend in (LLMBackend.LOCAL_OLLAMA, LLMBackend.LOCAL_LLAMA_CPP):
            calls_per_minute = config.get("calls_per_minute", 60)  # Higher rate for local
        else:
            calls_per_minute = config.get("calls_per_minute", 20)
        self._rate_limiter = RateLimiter(calls_per_minute)

        # Caching
        cache_size = config.get("cache_size", 100)
        self._cache = LRUCache(max_size=cache_size)

        # API keys from environment
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")

        # Local Ollama configuration (Phase E)
        self._ollama_url = config.get("ollama_url", config.get("ollama", {}).get("url", "http://localhost:11434"))
        self._ollama_model = config.get("ollama_model", config.get("ollama", {}).get("model", "llama3:8b"))

        # Local llama.cpp configuration (Phase E)
        self._llama_cpp_model_path = config.get(
            "llama_cpp_model_path",
            config.get("llama_cpp", {}).get("model_path", "models/llama-3-8b-q4.gguf")
        )
        self._llama_cpp_n_gpu_layers = config.get(
            "llama_cpp_n_gpu_layers",
            config.get("llama_cpp", {}).get("n_gpu_layers", 35)
        )
        self._llama_cpp_instance: Any = None  # Lazy loaded

        # Session will be created lazily
        self._session: aiohttp.ClientSession | None = None

        self._initialized = False
        self._api_available = False

        # Track backend usage statistics (Phase E)
        self._backend_stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "avg_latency_ms": 0.0,
        }

    def _parse_backend(self, backend_str: str) -> LLMBackend:
        """Parse backend string to enum."""
        backend_map = {
            "anthropic": LLMBackend.CLAUDE_API,
            "claude": LLMBackend.CLAUDE_API,
            "openai": LLMBackend.OPENAI_API,
            "gpt": LLMBackend.OPENAI_API,
            "ollama": LLMBackend.LOCAL_OLLAMA,
            "llama_cpp": LLMBackend.LOCAL_LLAMA_CPP,
            "llamacpp": LLMBackend.LOCAL_LLAMA_CPP,
        }
        return backend_map.get(backend_str.lower(), LLMBackend.CLAUDE_API)

    def _get_default_model(self) -> str:
        """Get default model for provider/backend."""
        if self._backend == LLMBackend.CLAUDE_API:
            return "claude-3-haiku-20240307"  # Fast and cost-effective
        elif self._backend == LLMBackend.OPENAI_API:
            return "gpt-3.5-turbo"
        elif self._backend == LLMBackend.LOCAL_OLLAMA:
            return self._ollama_model
        elif self._backend == LLMBackend.LOCAL_LLAMA_CPP:
            return "local-llama"
        return "gpt-3.5-turbo"

    async def initialize(self) -> None:
        """Initialize the client."""
        if self._initialized:
            return

        # Phase E: Check backend availability
        if self._backend == LLMBackend.LOCAL_OLLAMA:
            # Check if Ollama is running
            ollama_available = await self._check_ollama_available()
            if ollama_available:
                self._api_available = True
                logger.info(f"LLMClient initialized with Ollama ({self._ollama_model}) at {self._ollama_url}")
            else:
                logger.warning(f"Ollama not available at {self._ollama_url}, falling back to API")
                self._backend = LLMBackend.CLAUDE_API
                # Continue to API check below

        elif self._backend == LLMBackend.LOCAL_LLAMA_CPP:
            # Try to initialize llama.cpp
            llama_available = self._initialize_llama_cpp()
            if llama_available:
                self._api_available = True
                logger.info(f"LLMClient initialized with llama.cpp ({self._llama_cpp_model_path})")
            else:
                logger.warning("llama.cpp not available, falling back to API")
                self._backend = LLMBackend.CLAUDE_API
                # Continue to API check below

        # Check API key availability for API backends
        if self._backend in (LLMBackend.CLAUDE_API, LLMBackend.OPENAI_API):
            if self._backend == LLMBackend.CLAUDE_API and self._anthropic_key:
                self._api_available = True
                self._provider = "anthropic"
                logger.info(f"LLMClient initialized with Anthropic ({self._model})")
            elif self._backend == LLMBackend.OPENAI_API and self._openai_key:
                self._api_available = True
                self._provider = "openai"
                logger.info(f"LLMClient initialized with OpenAI ({self._model})")
            elif self._anthropic_key:
                # Fallback to Anthropic if available
                self._provider = "anthropic"
                self._backend = LLMBackend.CLAUDE_API
                self._api_available = True
                logger.info("LLMClient falling back to Anthropic")
            elif self._openai_key:
                # Fallback to OpenAI if available
                self._provider = "openai"
                self._backend = LLMBackend.OPENAI_API
                self._api_available = True
                logger.info("LLMClient falling back to OpenAI")
            else:
                self._api_available = False
                logger.warning(
                    "LLMClient: No API key found (ANTHROPIC_API_KEY or OPENAI_API_KEY) "
                    "and no local backend available. "
                    "Sentiment analysis will return neutral signals."
                )

        self._initialized = True

    async def _check_ollama_available(self) -> bool:
        """Check if Ollama server is running and has the model."""
        try:
            session = await self._get_session()
            async with session.get(f"{self._ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    # Check if our model is available (handle versioned names)
                    model_base = self._ollama_model.split(":")[0]
                    return any(model_base in m for m in models) or len(models) > 0
                return False
        except Exception as e:
            logger.debug(f"Ollama check failed: {e}")
            return False

    def _initialize_llama_cpp(self) -> bool:
        """Initialize llama.cpp if available."""
        try:
            from llama_cpp import Llama

            if not os.path.exists(self._llama_cpp_model_path):
                logger.warning(f"llama.cpp model not found: {self._llama_cpp_model_path}")
                return False

            self._llama_cpp_instance = Llama(
                model_path=self._llama_cpp_model_path,
                n_gpu_layers=self._llama_cpp_n_gpu_layers,
                n_ctx=2048,
                verbose=False,
            )
            return True
        except ImportError:
            logger.debug("llama-cpp-python not installed")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize llama.cpp: {e}")
            return False

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

        # Make API/local call with retries
        last_error: str | None = None
        start_time = time.time()
        self._backend_stats["requests"] += 1

        for attempt in range(self._max_retries):
            try:
                # Route to appropriate backend (Phase E)
                if self._backend == LLMBackend.LOCAL_OLLAMA:
                    result = await self._call_ollama(text, symbol)
                elif self._backend == LLMBackend.LOCAL_LLAMA_CPP:
                    result = await self._call_llama_cpp(text, symbol)
                elif self._backend == LLMBackend.CLAUDE_API or self._provider == "anthropic":
                    result = await self._call_anthropic(text, symbol)
                else:
                    result = await self._call_openai(text, symbol)

                # Track latency
                latency_ms = (time.time() - start_time) * 1000
                self._update_latency_stats(latency_ms)

                # Cache successful result
                if result.is_valid and use_cache:
                    self._cache.set(cache_key, result)
                    self._backend_stats["successes"] += 1

                return result

            except asyncio.TimeoutError:
                last_error = "API timeout"
                logger.warning(f"LLM timeout (attempt {attempt + 1}/{self._max_retries})")
            except aiohttp.ClientError as e:
                last_error = f"API error: {e}"
                logger.warning(f"LLM error (attempt {attempt + 1}): {e}")
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.exception(f"Unexpected LLM error: {e}")

            # Exponential backoff (shorter for local)
            if attempt < self._max_retries - 1:
                backoff = 2 ** attempt if self._backend in (LLMBackend.CLAUDE_API, LLMBackend.OPENAI_API) else 0.5
                await asyncio.sleep(backoff)

        # All retries failed
        self._backend_stats["failures"] += 1
        return SentimentResult(
            sentiment=0.0,
            confidence=0.0,
            rationale=f"LLM call failed after {self._max_retries} attempts",
            symbol=symbol,
            analyzed_at=datetime.now(timezone.utc),
            error=last_error,
        )

    def _update_latency_stats(self, latency_ms: float) -> None:
        """Update running average of latency."""
        alpha = 0.2  # EMA smoothing
        if self._backend_stats["avg_latency_ms"] == 0:
            self._backend_stats["avg_latency_ms"] = latency_ms
        else:
            self._backend_stats["avg_latency_ms"] = (
                alpha * latency_ms + (1 - alpha) * self._backend_stats["avg_latency_ms"]
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

    async def _call_ollama(self, text: str, symbol: str) -> SentimentResult:
        """
        Call local Ollama server (Phase E).

        Ollama provides a simple REST API for running LLMs locally.
        Typical latency is <50ms on GPU.
        """
        session = await self._get_session()

        # Use compact prompt for local models
        prompt = self.SENTIMENT_PROMPT_LOCAL.format(symbol=symbol, text=text[:1000])

        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistency
                "num_predict": 150,  # Limit output tokens
            },
        }

        async with session.post(
            f"{self._ollama_url}/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise aiohttp.ClientError(f"Ollama error {response.status}: {error_text}")

            data = await response.json()
            content = data.get("response", "")
            return self._parse_local_response(content, symbol, "ollama")

    async def _call_llama_cpp(self, text: str, symbol: str) -> SentimentResult:
        """
        Call local llama.cpp model (Phase E).

        llama.cpp provides extremely fast inference with GPU acceleration.
        Typical latency is <30ms with proper GPU offloading.
        """
        if self._llama_cpp_instance is None:
            raise RuntimeError("llama.cpp not initialized")

        # Use compact prompt for local models
        prompt = self.SENTIMENT_PROMPT_LOCAL.format(symbol=symbol, text=text[:1000])

        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._llama_cpp_instance(
                prompt,
                max_tokens=150,
                temperature=0.1,
                stop=["\n\n", "Text:"],
            )
        )

        content = result.get("choices", [{}])[0].get("text", "")
        return self._parse_local_response(content, symbol, "llama_cpp")

    def _parse_local_response(
        self,
        content: str,
        symbol: str,
        provider: str,
    ) -> SentimentResult:
        """
        Parse response from local LLM (Phase E).

        Local models may be less reliable at JSON formatting,
        so we try multiple parsing strategies.
        """
        import json
        import re

        try:
            # Clean up response
            content = content.strip()

            # Try to find JSON object
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                content = json_match.group()

            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            parsed = json.loads(content)

            sentiment = float(parsed.get("sentiment", 0.0))
            confidence = float(parsed.get("confidence", 0.5))
            rationale = str(parsed.get("rationale", "Local model analysis"))

            # Clamp values
            sentiment = max(-1.0, min(1.0, sentiment))
            confidence = max(0.0, min(1.0, confidence))

            return SentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                rationale=rationale,
                symbol=symbol,
                analyzed_at=datetime.now(timezone.utc),
                source=f"llm_sentiment_{provider}",
            )

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            # Fallback: try to extract sentiment from text heuristically
            sentiment = self._extract_sentiment_heuristic(content)
            return SentimentResult(
                sentiment=sentiment,
                confidence=0.3,  # Low confidence for heuristic
                rationale=f"Heuristic extraction from: {content[:100]}",
                symbol=symbol,
                analyzed_at=datetime.now(timezone.utc),
                source=f"llm_sentiment_{provider}_heuristic",
            )

    def _extract_sentiment_heuristic(self, text: str) -> float:
        """
        Extract sentiment heuristically when JSON parsing fails.

        Simple keyword-based extraction as fallback.
        """
        text_lower = text.lower()

        # Look for explicit sentiment values
        import re
        number_match = re.search(r'sentiment["\s:]+(-?\d+\.?\d*)', text_lower)
        if number_match:
            try:
                return max(-1.0, min(1.0, float(number_match.group(1))))
            except ValueError:
                pass

        # Keyword-based sentiment
        bullish_words = ["bullish", "positive", "strong", "growth", "beat", "upgrade"]
        bearish_words = ["bearish", "negative", "weak", "decline", "miss", "downgrade"]

        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)

        if bullish_count > bearish_count:
            return 0.3
        elif bearish_count > bullish_count:
            return -0.3
        return 0.0

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

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about current backend configuration (Phase E)."""
        return {
            "backend": self._backend.value,
            "provider": self._provider,
            "model": self._model,
            "api_available": self._api_available,
            "is_local": self._backend in (LLMBackend.LOCAL_OLLAMA, LLMBackend.LOCAL_LLAMA_CPP),
            "ollama_url": self._ollama_url if self._backend == LLMBackend.LOCAL_OLLAMA else None,
            "llama_cpp_model": self._llama_cpp_model_path if self._backend == LLMBackend.LOCAL_LLAMA_CPP else None,
            "stats": dict(self._backend_stats),
        }

    def switch_backend(self, backend: str) -> bool:
        """
        Switch to a different backend at runtime (Phase E).

        Useful for fallback scenarios or testing.

        Args:
            backend: Backend name ("claude", "openai", "ollama", "llama_cpp")

        Returns:
            True if switch was successful
        """
        new_backend = self._parse_backend(backend)

        # Validate new backend is available
        if new_backend == LLMBackend.LOCAL_LLAMA_CPP and self._llama_cpp_instance is None:
            if not self._initialize_llama_cpp():
                logger.error("Cannot switch to llama_cpp: not available")
                return False

        self._backend = new_backend
        self._model = self._get_default_model()

        if new_backend in (LLMBackend.CLAUDE_API, LLMBackend.OPENAI_API):
            self._provider = "anthropic" if new_backend == LLMBackend.CLAUDE_API else "openai"

        logger.info(f"LLM backend switched to {new_backend.value}")
        return True

"""
FinBERT Client for Financial Sentiment Analysis
================================================

Local NLP inference using FinBERT transformer model
for financial news sentiment analysis.

Inspired by:
- sp500-finbert-lstm-hybrid (19% lower MAE vs baselines)
- ProsusAI/finbert model

Features:
- Local inference (no API costs)
- Financial domain-specific sentiment
- Batch processing support
- Confidence scores
- Caching for repeated texts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
import hashlib

# Optional imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


logger = logging.getLogger(__name__)


@dataclass
class FinBERTResult:
    """Result of FinBERT sentiment analysis."""
    text: str
    sentiment: str  # "positive", "negative", "neutral"
    score: float  # -1.0 to 1.0 (negative to positive)
    confidence: float  # 0.0 to 1.0
    probabilities: dict[str, float]  # Class probabilities
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sentiment": self.sentiment,
            "score": self.score,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class FinBERTConfig:
    """Configuration for FinBERT client."""
    model_name: str = "ProsusAI/finbert"
    device: str = "auto"  # "auto", "cuda", "cpu"
    max_length: int = 512
    batch_size: int = 8
    cache_size: int = 1000
    use_fp16: bool = True  # Half precision for faster inference


class FinBERTClient:
    """
    FinBERT Client for Financial Sentiment Analysis.

    Uses the ProsusAI/finbert model for domain-specific
    sentiment analysis of financial text.

    Example:
        client = FinBERTClient()
        await client.initialize()

        result = await client.analyze("Apple stock surges on strong earnings")
        print(result.sentiment)  # "positive"
        print(result.score)  # 0.85
    """

    def __init__(self, config: FinBERTConfig | None = None):
        self._config = config or FinBERTConfig()
        self._model = None
        self._tokenizer = None
        self._device = None
        self._initialized = False

        # Simple cache using dict
        self._cache: dict[str, FinBERTResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    async def initialize(self) -> bool:
        """
        Initialize the FinBERT model.

        Returns:
            True if initialization successful, False otherwise
        """
        if not HAS_TRANSFORMERS:
            logger.warning(
                "transformers library not installed. "
                "Install with: pip install transformers torch"
            )
            return False

        try:
            logger.info(f"Loading FinBERT model: {self._config.model_name}")

            # Determine device
            if self._config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self._config.device

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_name
            )

            # Load model
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._config.model_name
            )

            # Move to device
            self._model = self._model.to(self._device)

            # Use half precision if available
            if self._config.use_fp16 and self._device == "cuda":
                self._model = self._model.half()

            # Set eval mode
            self._model.eval()

            self._initialized = True
            logger.info(f"FinBERT loaded on {self._device}")
            return True

        except Exception as e:
            logger.exception(f"Failed to load FinBERT: {e}")
            return False

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    async def analyze(self, text: str) -> FinBERTResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            FinBERTResult with sentiment analysis
        """
        if not self._initialized:
            return self._get_neutral_result(text, "Model not initialized")

        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1

        # Analyze
        start_time = datetime.now(timezone.utc)

        try:
            result = self._analyze_single(text)
            result.processing_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Cache result
            if len(self._cache) >= self._config.cache_size:
                # Remove oldest entries (simple FIFO)
                keys_to_remove = list(self._cache.keys())[:100]
                for key in keys_to_remove:
                    del self._cache[key]
            self._cache[cache_key] = result

            return result

        except Exception as e:
            logger.exception(f"FinBERT analysis error: {e}")
            return self._get_neutral_result(text, str(e))

    def _analyze_single(self, text: str) -> FinBERTResult:
        """Analyze single text (internal)."""
        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.max_length,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        # Get probabilities
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # FinBERT labels: positive, negative, neutral
        labels = ["positive", "negative", "neutral"]
        probabilities = {label: float(prob) for label, prob in zip(labels, probs)}

        # Get prediction
        pred_idx = probs.argmax()
        sentiment = labels[pred_idx]
        confidence = float(probs[pred_idx])

        # Calculate score (-1 to 1)
        # positive contributes +, negative contributes -, neutral is 0
        score = probabilities["positive"] - probabilities["negative"]

        return FinBERTResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            sentiment=sentiment,
            score=score,
            confidence=confidence,
            probabilities=probabilities,
        )

    async def analyze_batch(self, texts: list[str]) -> list[FinBERTResult]:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of FinBERTResult
        """
        if not self._initialized:
            return [self._get_neutral_result(t, "Model not initialized") for t in texts]

        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                self._cache_hits += 1
                results.append((i, self._cache[cache_key]))
            else:
                self._cache_misses += 1
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Analyze uncached texts in batches
        if uncached_texts:
            batch_results = self._analyze_batch_internal(uncached_texts)
            for idx, result in zip(uncached_indices, batch_results):
                results.append((idx, result))
                # Cache
                cache_key = self._get_cache_key(uncached_texts[uncached_indices.index(idx)])
                self._cache[cache_key] = result

        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _analyze_batch_internal(self, texts: list[str]) -> list[FinBERTResult]:
        """Analyze batch of texts (internal)."""
        results = []
        batch_size = self._config.batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self._config.max_length,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

            # Get probabilities for batch
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            labels = ["positive", "negative", "neutral"]

            for j, text in enumerate(batch_texts):
                text_probs = probs[j]
                probabilities = {label: float(prob) for label, prob in zip(labels, text_probs)}

                pred_idx = text_probs.argmax()
                sentiment = labels[pred_idx]
                confidence = float(text_probs[pred_idx])
                score = probabilities["positive"] - probabilities["negative"]

                results.append(FinBERTResult(
                    text=text[:100] + "..." if len(text) > 100 else text,
                    sentiment=sentiment,
                    score=score,
                    confidence=confidence,
                    probabilities=probabilities,
                ))

        return results

    def _get_neutral_result(self, text: str, reason: str) -> FinBERTResult:
        """Return neutral result for error cases."""
        return FinBERTResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            sentiment="neutral",
            score=0.0,
            confidence=0.5,
            probabilities={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
        )

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "initialized": self._initialized,
            "device": self._device,
            "model_name": self._config.model_name,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / max(1, total),
        }

    @property
    def is_available(self) -> bool:
        """Check if FinBERT is available."""
        return self._initialized


# Convenience function for one-off analysis
async def analyze_financial_sentiment(text: str) -> FinBERTResult:
    """
    Convenience function for one-off sentiment analysis.

    Note: For production use, create and reuse a FinBERTClient instance
    to avoid reloading the model.
    """
    client = FinBERTClient()
    if await client.initialize():
        return await client.analyze(text)
    return client._get_neutral_result(text, "Failed to initialize")

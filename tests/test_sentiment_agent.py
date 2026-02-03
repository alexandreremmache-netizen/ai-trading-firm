"""
Tests for SentimentAgent and LLMClient
======================================
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.sentiment_agent import SentimentAgent, NewsItem
from core.agent_base import AgentConfig
from core.events import MarketDataEvent, SignalDirection
from core.llm_client import LLMClient, SentimentResult, LRUCache, RateLimiter


# =============================================================================
# LLMClient Tests
# =============================================================================


class TestLRUCache:
    """Tests for LRU cache."""

    def test_cache_set_and_get(self):
        """Test basic cache set and get."""
        cache = LRUCache(max_size=10)
        result = SentimentResult(
            sentiment=0.5,
            confidence=0.8,
            rationale="Test",
            symbol="AAPL",
            analyzed_at=datetime.now(timezone.utc),
        )
        cache.set("key1", result)
        retrieved = cache.get("key1")
        assert retrieved is not None
        assert retrieved.sentiment == 0.5

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = LRUCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_cache_eviction(self):
        """Test LRU eviction when max size reached."""
        cache = LRUCache(max_size=2)
        for i in range(3):
            result = SentimentResult(
                sentiment=i / 10,
                confidence=0.8,
                rationale=f"Test {i}",
                symbol="AAPL",
                analyzed_at=datetime.now(timezone.utc),
            )
            cache.set(f"key{i}", result)

        # First key should be evicted
        assert cache.get("key0") is None
        assert cache.get("key1") is not None
        assert cache.get("key2") is not None


class TestRateLimiter:
    """Tests for rate limiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_first_call(self):
        """Test that first call is not delayed."""
        limiter = RateLimiter(calls_per_minute=60)
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 0.1  # Should be almost instant

    @pytest.mark.asyncio
    async def test_rate_limiter_delays_rapid_calls(self):
        """Test that rapid calls are delayed."""
        limiter = RateLimiter(calls_per_minute=60)  # 1 per second
        await limiter.acquire()
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start
        # Should wait at least close to 1 second (minus timing tolerance)
        assert elapsed >= 0.8


class TestLLMClient:
    """Tests for LLM client."""

    @pytest.mark.asyncio
    async def test_init_without_api_key(self):
        """Test client initializes without API key."""
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient()
            await client.initialize()
            assert not client._api_available

    @pytest.mark.asyncio
    async def test_analyze_sentiment_without_api(self):
        """Test sentiment analysis returns neutral without API."""
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient()
            await client.initialize()
            result = await client.analyze_sentiment("Test news", "AAPL")
            assert result.sentiment == 0.0
            assert result.confidence == 0.1
            assert not result.is_valid

    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test cache key is consistent."""
        client = LLMClient()
        key1 = client._cache_key("Test text", "AAPL")
        key2 = client._cache_key("Test text", "AAPL")
        key3 = client._cache_key("Different text", "AAPL")
        assert key1 == key2
        assert key1 != key3

    def test_sentiment_result_is_valid(self):
        """Test SentimentResult validation."""
        valid = SentimentResult(
            sentiment=0.5,
            confidence=0.8,
            rationale="Bullish",
            symbol="AAPL",
            analyzed_at=datetime.now(timezone.utc),
        )
        assert valid.is_valid

        invalid = SentimentResult(
            sentiment=0.0,
            confidence=0.0,
            rationale="Error",
            symbol="AAPL",
            analyzed_at=datetime.now(timezone.utc),
            error="API failed",
        )
        assert not invalid.is_valid


# =============================================================================
# NewsItem Tests
# =============================================================================


class TestNewsItem:
    """Tests for NewsItem dataclass."""

    def test_full_text(self):
        """Test full_text property."""
        news = NewsItem(
            title="Apple beats earnings",
            summary="Apple reported better than expected Q4 results.",
            source="test",
            published=datetime.now(timezone.utc),
            symbols=["AAPL"],
        )
        assert "Apple beats earnings" in news.full_text
        assert "better than expected" in news.full_text

    def test_age_hours(self):
        """Test age_hours calculation."""
        news = NewsItem(
            title="Test",
            summary="Test",
            source="test",
            published=datetime.now(timezone.utc) - timedelta(hours=2),
            symbols=["AAPL"],
        )
        assert 1.9 < news.age_hours < 2.1


# =============================================================================
# SentimentAgent Tests
# =============================================================================


class TestSentimentAgent:
    """Tests for SentimentAgent."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = MagicMock()
        bus.publish_signal = AsyncMock()
        bus.publish = AsyncMock()
        return bus

    @pytest.fixture
    def mock_audit_logger(self):
        """Create mock audit logger."""
        logger = MagicMock()
        logger.log_event = MagicMock()
        return logger

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = MagicMock(spec=LLMClient)
        client.initialize = AsyncMock()
        client.close = AsyncMock()
        client.analyze_sentiment = AsyncMock(return_value=SentimentResult(
            sentiment=0.6,
            confidence=0.8,
            rationale="Bullish news",
            symbol="AAPL",
            analyzed_at=datetime.now(timezone.utc),
        ))
        return client

    @pytest.fixture
    def agent_config(self):
        """Create agent config."""
        return AgentConfig(
            name="SentimentAgent",
            enabled=True,
            parameters={
                "analysis_interval_seconds": 1,  # Fast for tests
                "min_confidence": 0.5,
            },
        )

    @pytest.mark.asyncio
    async def test_agent_initialization(
        self, agent_config, mock_event_bus, mock_audit_logger, mock_llm_client
    ):
        """Test agent initializes correctly."""
        agent = SentimentAgent(
            config=agent_config,
            event_bus=mock_event_bus,
            audit_logger=mock_audit_logger,
            llm_client=mock_llm_client,
        )

        assert agent.name == "SentimentAgent"
        assert agent._min_confidence == 0.5

    @pytest.mark.asyncio
    async def test_process_event_publishes_signal(
        self, agent_config, mock_event_bus, mock_audit_logger, mock_llm_client
    ):
        """Test that processing event publishes a signal."""
        agent = SentimentAgent(
            config=agent_config,
            event_bus=mock_event_bus,
            audit_logger=mock_audit_logger,
            llm_client=mock_llm_client,
        )

        # Mock news data
        agent._recent_news = {
            "AAPL": [NewsItem(
                title="Apple beats earnings",
                summary="Strong results",
                source="test",
                published=datetime.now(timezone.utc),
                symbols=["AAPL"],
            )]
        }
        agent._http_session = MagicMock()

        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=150.0,
            ask=150.10,
            last=150.05,
        )

        await agent.process_event(event)

        # Should publish a signal
        mock_event_bus.publish_signal.assert_called()

    @pytest.mark.asyncio
    async def test_extract_symbols(
        self, agent_config, mock_event_bus, mock_audit_logger, mock_llm_client
    ):
        """Test symbol extraction from text."""
        agent = SentimentAgent(
            config=agent_config,
            event_bus=mock_event_bus,
            audit_logger=mock_audit_logger,
            llm_client=mock_llm_client,
        )

        # Direct symbol
        symbols = agent._extract_symbols("AAPL stock rises")
        assert "AAPL" in symbols

        # Company name
        symbols = agent._extract_symbols("Apple announces new iPhone")
        assert "AAPL" in symbols

        # Ticker with $
        symbols = agent._extract_symbols("$MSFT hits new high")
        assert "MSFT" in symbols

        # Multiple symbols
        symbols = agent._extract_symbols("Tech rally: AAPL MSFT GOOGL all up")
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOGL" in symbols

    @pytest.mark.asyncio
    async def test_aggregate_sentiments(
        self, agent_config, mock_event_bus, mock_audit_logger, mock_llm_client
    ):
        """Test sentiment aggregation."""
        agent = SentimentAgent(
            config=agent_config,
            event_bus=mock_event_bus,
            audit_logger=mock_audit_logger,
            llm_client=mock_llm_client,
        )

        sentiments = [
            SentimentResult(
                sentiment=0.8,
                confidence=0.9,
                rationale="Very bullish",
                symbol="AAPL",
                analyzed_at=datetime.now(timezone.utc),
            ),
            SentimentResult(
                sentiment=0.4,
                confidence=0.7,
                rationale="Moderately bullish",
                symbol="AAPL",
                analyzed_at=datetime.now(timezone.utc),
            ),
        ]

        news_items = [
            NewsItem(
                title="News 1",
                summary="Summary 1",
                source="test",
                published=datetime.now(timezone.utc),
                symbols=["AAPL"],
            ),
            NewsItem(
                title="News 2",
                summary="Summary 2",
                source="test",
                published=datetime.now(timezone.utc) - timedelta(hours=6),
                symbols=["AAPL"],
            ),
        ]

        result = agent._aggregate_sentiments(sentiments, news_items)

        # Aggregated sentiment should be positive (weighted toward more confident/recent)
        assert result["sentiment"] > 0
        assert result["confidence"] > 0
        assert result["num_sources"] == 2

    @pytest.mark.asyncio
    async def test_should_analyze_respects_interval(
        self, agent_config, mock_event_bus, mock_audit_logger, mock_llm_client
    ):
        """Test analysis interval is respected."""
        agent = SentimentAgent(
            config=agent_config,
            event_bus=mock_event_bus,
            audit_logger=mock_audit_logger,
            llm_client=mock_llm_client,
        )
        agent._analysis_interval = 60  # 1 minute

        # First check should allow analysis
        assert agent._should_analyze("AAPL")

        # Mark as analyzed
        agent._last_analysis["AAPL"] = datetime.now(timezone.utc)

        # Second check should not allow (too soon)
        assert not agent._should_analyze("AAPL")

        # Different symbol should allow
        assert agent._should_analyze("MSFT")

    @pytest.mark.asyncio
    async def test_build_rationale(
        self, agent_config, mock_event_bus, mock_audit_logger, mock_llm_client
    ):
        """Test rationale building."""
        agent = SentimentAgent(
            config=agent_config,
            event_bus=mock_event_bus,
            audit_logger=mock_audit_logger,
            llm_client=mock_llm_client,
        )

        aggregated = {
            "sentiment": 0.7,
            "confidence": 0.85,
            "num_sources": 3,
        }

        news_items = [
            NewsItem(
                title="Apple beats earnings expectations with strong iPhone sales",
                summary="Summary",
                source="test",
                published=datetime.now(timezone.utc),
                symbols=["AAPL"],
            ),
        ]

        rationale = agent._build_rationale(aggregated, news_items)

        assert "bullish" in rationale.lower()
        assert "0.70" in rationale or "0.7" in rationale
        assert "Apple beats" in rationale

    @pytest.mark.asyncio
    async def test_neutral_signal_on_low_confidence(
        self, agent_config, mock_event_bus, mock_audit_logger
    ):
        """Test that low confidence results in neutral signal."""
        # Create LLM client that returns low confidence
        low_conf_client = MagicMock(spec=LLMClient)
        low_conf_client.initialize = AsyncMock()
        low_conf_client.close = AsyncMock()
        low_conf_client.analyze_sentiment = AsyncMock(return_value=SentimentResult(
            sentiment=0.6,
            confidence=0.3,  # Below threshold
            rationale="Uncertain",
            symbol="AAPL",
            analyzed_at=datetime.now(timezone.utc),
        ))

        agent = SentimentAgent(
            config=agent_config,
            event_bus=mock_event_bus,
            audit_logger=mock_audit_logger,
            llm_client=low_conf_client,
        )

        agent._recent_news = {
            "AAPL": [NewsItem(
                title="Test",
                summary="Test",
                source="test",
                published=datetime.now(timezone.utc),
                symbols=["AAPL"],
            )]
        }
        agent._http_session = MagicMock()

        event = MarketDataEvent(
            source_agent="test",
            symbol="AAPL",
            bid=150.0,
            ask=150.10,
            last=150.05,
        )

        await agent.process_event(event)

        # Check that a signal was published
        mock_event_bus.publish_signal.assert_called()

        # Get the signal that was published
        call_args = mock_event_bus.publish_signal.call_args
        signal = call_args[0][0]

        # Low confidence should result in FLAT direction
        assert signal.direction == SignalDirection.FLAT


# =============================================================================
# Integration Tests
# =============================================================================


class TestSentimentAgentIntegration:
    """Integration tests for full sentiment analysis flow."""

    @pytest.mark.asyncio
    async def test_full_analysis_flow(self):
        """Test complete sentiment analysis flow."""
        # Setup
        event_bus = MagicMock()
        event_bus.publish_signal = AsyncMock()

        audit_logger = MagicMock()
        audit_logger.log_event = MagicMock()

        llm_client = MagicMock(spec=LLMClient)
        llm_client.initialize = AsyncMock()
        llm_client.close = AsyncMock()
        llm_client.analyze_sentiment = AsyncMock(return_value=SentimentResult(
            sentiment=0.75,
            confidence=0.85,
            rationale="Strong earnings beat",
            symbol="AAPL",
            analyzed_at=datetime.now(timezone.utc),
        ))

        config = AgentConfig(
            name="SentimentAgent",
            enabled=True,
            parameters={"analysis_interval_seconds": 0, "min_confidence": 0.5},
        )

        agent = SentimentAgent(
            config=config,
            event_bus=event_bus,
            audit_logger=audit_logger,
            llm_client=llm_client,
        )

        # Add mock news
        agent._recent_news = {
            "AAPL": [NewsItem(
                title="Apple beats Q4 earnings",
                summary="Apple reported revenue of $95B, beating estimates.",
                source="test",
                published=datetime.now(timezone.utc),
                symbols=["AAPL"],
            )]
        }
        agent._http_session = MagicMock()

        # Process market data event
        event = MarketDataEvent(
            source_agent="IB",
            symbol="AAPL",
            bid=175.0,
            ask=175.05,
            last=175.02,
            volume=1000000,
        )

        await agent.process_event(event)

        # Verify signal was published
        event_bus.publish_signal.assert_called_once()

        # Get the published signal
        signal = event_bus.publish_signal.call_args[0][0]

        # Verify signal properties
        assert signal.source_agent == "SentimentAgent"
        assert signal.symbol == "AAPL"
        assert signal.direction == SignalDirection.LONG  # Positive sentiment
        assert signal.strength > 0
        assert signal.confidence > 0.5
        assert "llm_sentiment" in signal.data_sources

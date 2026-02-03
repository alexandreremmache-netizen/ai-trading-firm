"""
Sentiment Analysis Agent
========================

Generates signals based on news sentiment analysis using LLM.
Monitors financial news feeds and analyzes sentiment for portfolio symbols.

Responsibility: Sentiment signal generation ONLY.
Does NOT make trading decisions or send orders.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Any

import aiohttp

from core.agent_base import SignalAgent, AgentConfig
from core.events import (
    Event,
    EventType,
    MarketDataEvent,
    SignalEvent,
    SignalDirection,
)
from core.llm_client import LLMClient, SentimentResult

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger


logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """A news article or headline."""
    title: str
    summary: str
    source: str
    published: datetime
    symbols: list[str]  # Related symbols
    url: str = ""

    @property
    def full_text(self) -> str:
        """Get full text for analysis."""
        return f"{self.title}\n\n{self.summary}"

    @property
    def age_hours(self) -> float:
        """Get age of news in hours."""
        delta = datetime.now(timezone.utc) - self.published
        return delta.total_seconds() / 3600


class SentimentAgent(SignalAgent):
    """
    Sentiment Analysis Agent.

    Analyzes news sentiment using LLM to generate trading signals.

    Data sources:
    - RSS feeds (Reuters, Bloomberg, Yahoo Finance)
    - Financial news APIs

    Signal output:
    - Bullish/Bearish sentiment signals per symbol
    - Confidence based on news clarity and volume
    """

    # Default RSS feeds for financial news
    DEFAULT_NEWS_SOURCES = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
        # Reuters and Bloomberg require authentication, using Yahoo as fallback
    ]

    # Symbols to monitor (will be expanded from market data events)
    CORE_SYMBOLS = {"SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"}

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
        llm_client: LLMClient | None = None,
    ):
        super().__init__(config, event_bus, audit_logger)

        # LLM client for sentiment analysis
        self._llm_client = llm_client or LLMClient(config.parameters)

        # Configuration
        self._news_sources = config.parameters.get(
            "news_sources", self.DEFAULT_NEWS_SOURCES
        )
        self._analysis_interval = config.parameters.get(
            "analysis_interval_seconds", 300
        )
        self._max_news_age_hours = config.parameters.get("max_news_age_hours", 24)
        self._min_confidence = config.parameters.get("min_confidence", 0.5)

        # State
        self._last_analysis: dict[str, datetime] = {}
        self._recent_news: dict[str, list[NewsItem]] = {}  # symbol -> news items
        self._monitored_symbols: set[str] = set(self.CORE_SYMBOLS)

        # HTTP session for RSS fetching
        self._http_session: aiohttp.ClientSession | None = None

    async def initialize(self) -> None:
        """Initialize sentiment data feeds and LLM client."""
        logger.info(
            f"SentimentAgent initializing with {len(self._news_sources)} news sources"
        )

        # Initialize LLM client
        await self._llm_client.initialize()

        # Create HTTP session for RSS
        timeout = aiohttp.ClientTimeout(total=10)
        self._http_session = aiohttp.ClientSession(timeout=timeout)

        # Initial news fetch
        await self._refresh_news()

        logger.info(
            f"SentimentAgent ready - monitoring {len(self._monitored_symbols)} symbols"
        )

    async def shutdown(self) -> None:
        """Cleanup resources."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        await self._llm_client.close()
        logger.info("SentimentAgent shutdown complete")

    async def process_event(self, event: Event) -> None:
        """Process market data and generate sentiment signals."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol

        # Add symbol to monitoring list
        if symbol not in self._monitored_symbols:
            self._monitored_symbols.add(symbol)

        # Check if we should analyze this symbol
        if not self._should_analyze(symbol):
            # Still publish a signal to satisfy barrier
            await self._publish_neutral_signal(symbol, "Analysis interval not reached")
            return

        # Analyze sentiment for symbol
        signal = await self._analyze_symbol_sentiment(symbol)

        # Always publish a signal for barrier synchronization
        if signal is None:
            signal = SignalEvent(
                source_agent=self.name,
                strategy_name="sentiment_neutral",
                symbol=symbol,
                direction=SignalDirection.FLAT,
                strength=0.0,
                confidence=0.3,
                rationale=f"Sentiment: No actionable news for {symbol}",
                data_sources=("llm_sentiment", "news_sentiment"),
            )

        await self._event_bus.publish_signal(signal)
        self._audit_logger.log_event(signal)

    def _should_analyze(self, symbol: str) -> bool:
        """Check if enough time has passed since last analysis."""
        last = self._last_analysis.get(symbol)
        if last is None:
            return True

        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed >= self._analysis_interval

    async def _publish_neutral_signal(self, symbol: str, reason: str) -> None:
        """Publish a neutral signal for barrier satisfaction."""
        signal = SignalEvent(
            source_agent=self.name,
            strategy_name="sentiment_monitoring",
            symbol=symbol,
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.2,
            rationale=f"Sentiment: {reason}",
            data_sources=("llm_sentiment",),
        )
        await self._event_bus.publish_signal(signal)

    async def _analyze_symbol_sentiment(self, symbol: str) -> SignalEvent | None:
        """
        Analyze sentiment for a specific symbol.

        Returns:
            SignalEvent with sentiment-based direction, or None if no signal
        """
        self._last_analysis[symbol] = datetime.now(timezone.utc)

        # Get recent news for symbol
        news_items = await self._get_recent_news(symbol)

        if not news_items:
            return None

        # Analyze sentiment for each news item
        sentiments: list[SentimentResult] = []

        for news in news_items[:5]:  # Limit to 5 most recent
            try:
                result = await self._llm_client.analyze_sentiment(
                    news.full_text,
                    symbol,
                )
                if result.is_valid:
                    sentiments.append(result)
            except Exception as e:
                logger.warning(f"Sentiment analysis failed for {symbol}: {e}")

        if not sentiments:
            return None

        # Aggregate sentiments (weighted by confidence and recency)
        aggregated = self._aggregate_sentiments(sentiments, news_items)

        # Check confidence threshold
        if aggregated["confidence"] < self._min_confidence:
            return SignalEvent(
                source_agent=self.name,
                strategy_name="sentiment_low_confidence",
                symbol=symbol,
                direction=SignalDirection.FLAT,
                strength=0.0,
                confidence=aggregated["confidence"],
                rationale=f"Sentiment confidence too low: {aggregated['confidence']:.2f}",
                data_sources=("llm_sentiment", "news_sentiment"),
            )

        # Determine direction and strength
        sentiment = aggregated["sentiment"]
        confidence = aggregated["confidence"]

        if sentiment > 0.2:
            direction = SignalDirection.LONG
            strength = min(1.0, sentiment)
        elif sentiment < -0.2:
            direction = SignalDirection.SHORT
            strength = max(-1.0, sentiment)
        else:
            direction = SignalDirection.FLAT
            strength = 0.0

        rationale = self._build_rationale(aggregated, news_items)

        return SignalEvent(
            source_agent=self.name,
            strategy_name="sentiment_llm",
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            rationale=rationale,
            data_sources=("llm_sentiment", "news_sentiment", "financial_news"),
        )

    def _aggregate_sentiments(
        self,
        sentiments: list[SentimentResult],
        news_items: list[NewsItem],
    ) -> dict[str, Any]:
        """
        Aggregate multiple sentiment results.

        Uses confidence-weighted average with recency decay.
        """
        if not sentiments:
            return {"sentiment": 0.0, "confidence": 0.0, "rationale": "No data"}

        total_weight = 0.0
        weighted_sentiment = 0.0
        rationales: list[str] = []

        for i, result in enumerate(sentiments):
            # Recency weight (more recent = higher weight)
            if i < len(news_items):
                age_hours = news_items[i].age_hours
                recency_weight = 1.0 / (1.0 + age_hours / 12.0)  # Decay over 12 hours
            else:
                recency_weight = 0.5

            # Combined weight
            weight = result.confidence * recency_weight
            total_weight += weight
            weighted_sentiment += result.sentiment * weight
            rationales.append(result.rationale)

        if total_weight == 0:
            return {"sentiment": 0.0, "confidence": 0.0, "rationale": "Zero weight"}

        avg_sentiment = weighted_sentiment / total_weight
        avg_confidence = sum(s.confidence for s in sentiments) / len(sentiments)

        return {
            "sentiment": avg_sentiment,
            "confidence": avg_confidence,
            "rationale": "; ".join(rationales[:3]),  # Top 3 rationales
            "num_sources": len(sentiments),
        }

    def _build_rationale(
        self,
        aggregated: dict[str, Any],
        news_items: list[NewsItem],
    ) -> str:
        """Build human-readable rationale for the signal."""
        sentiment = aggregated["sentiment"]
        confidence = aggregated["confidence"]
        num_sources = aggregated.get("num_sources", len(news_items))

        if sentiment > 0.5:
            tone = "strongly bullish"
        elif sentiment > 0.2:
            tone = "moderately bullish"
        elif sentiment < -0.5:
            tone = "strongly bearish"
        elif sentiment < -0.2:
            tone = "moderately bearish"
        else:
            tone = "neutral"

        headlines = [n.title[:50] for n in news_items[:2]]
        headlines_str = "; ".join(headlines) if headlines else "No headlines"

        return (
            f"LLM Sentiment: {tone} (score={sentiment:.2f}, conf={confidence:.2f}) "
            f"based on {num_sources} news sources. Headlines: {headlines_str}"
        )

    async def _get_recent_news(self, symbol: str) -> list[NewsItem]:
        """Get recent news for a symbol."""
        # Check cache first
        if symbol in self._recent_news:
            cached = self._recent_news[symbol]
            # Filter for max age
            fresh = [
                n for n in cached
                if n.age_hours <= self._max_news_age_hours
            ]
            if fresh:
                return fresh

        # Fetch fresh news
        await self._refresh_news()

        return self._recent_news.get(symbol, [])

    async def _refresh_news(self) -> None:
        """Refresh news from all sources."""
        if not self._http_session:
            return

        all_news: list[NewsItem] = []

        for source_url in self._news_sources:
            try:
                news_items = await self._fetch_rss(source_url)
                all_news.extend(news_items)
            except Exception as e:
                logger.warning(f"Failed to fetch news from {source_url}: {e}")

        # Organize by symbol
        self._recent_news.clear()
        for news in all_news:
            for symbol in news.symbols:
                if symbol not in self._recent_news:
                    self._recent_news[symbol] = []
                self._recent_news[symbol].append(news)

        # Sort by recency
        for symbol in self._recent_news:
            self._recent_news[symbol].sort(key=lambda n: n.published, reverse=True)

        logger.debug(f"Refreshed news: {len(all_news)} items for {len(self._recent_news)} symbols")

    async def _fetch_rss(self, url: str) -> list[NewsItem]:
        """Fetch and parse RSS feed."""
        if not self._http_session:
            return []

        try:
            async with self._http_session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"RSS fetch failed for {url}: HTTP {response.status}")
                    return []

                content = await response.text()
                return self._parse_rss(content, url)

        except asyncio.TimeoutError:
            logger.warning(f"RSS fetch timeout for {url}")
            return []
        except aiohttp.ClientError as e:
            logger.warning(f"RSS fetch error for {url}: {e}")
            return []

    def _parse_rss(self, content: str, source: str) -> list[NewsItem]:
        """Parse RSS XML content into NewsItem list."""
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed - RSS parsing disabled")
            return self._parse_rss_simple(content, source)

        feed = feedparser.parse(content)
        news_items: list[NewsItem] = []

        for entry in feed.entries[:20]:  # Limit to 20 items
            try:
                # Extract published date
                published = datetime.now(timezone.utc)
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    from time import mktime
                    published = datetime.fromtimestamp(
                        mktime(entry.published_parsed),
                        tz=timezone.utc,
                    )

                # Extract symbols from title/summary
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                symbols = self._extract_symbols(f"{title} {summary}")

                if not symbols:
                    # Assign to general market if no specific symbols
                    symbols = ["SPY"]

                news_items.append(NewsItem(
                    title=title,
                    summary=summary[:500],  # Truncate
                    source=source,
                    published=published,
                    symbols=symbols,
                    url=entry.get("link", ""),
                ))

            except Exception as e:
                logger.debug(f"Failed to parse RSS entry: {e}")

        return news_items

    def _parse_rss_simple(self, content: str, source: str) -> list[NewsItem]:
        """Simple XML parsing fallback when feedparser not available."""
        import re

        news_items: list[NewsItem] = []

        # Simple regex extraction
        titles = re.findall(r"<title>([^<]+)</title>", content)
        descriptions = re.findall(r"<description>([^<]+)</description>", content)

        for i, title in enumerate(titles[:10]):
            summary = descriptions[i] if i < len(descriptions) else ""
            symbols = self._extract_symbols(f"{title} {summary}")

            if not symbols:
                symbols = ["SPY"]

            news_items.append(NewsItem(
                title=title,
                summary=summary[:500],
                source=source,
                published=datetime.now(timezone.utc),
                symbols=symbols,
            ))

        return news_items

    def _extract_symbols(self, text: str) -> list[str]:
        """Extract stock symbols from text."""
        import re

        # Known symbols to look for
        known_symbols = {
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
            "JPM", "V", "JNJ", "UNH", "HD", "PG", "MA", "DIS", "PYPL",
            "SPY", "QQQ", "IWM", "DIA", "GLD", "TLT",
            "ES", "NQ", "YM", "CL", "GC",  # Futures
        }

        # Company name to symbol mapping
        company_map = {
            "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
            "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META",
            "facebook": "META", "nvidia": "NVDA", "tesla": "TSLA",
            "jpmorgan": "JPM", "visa": "V", "johnson": "JNJ",
        }

        text_upper = text.upper()
        text_lower = text.lower()
        found: set[str] = set()

        # Direct symbol matches
        for symbol in known_symbols:
            if re.search(rf"\b{symbol}\b", text_upper):
                found.add(symbol)

        # Company name matches
        for company, symbol in company_map.items():
            if company in text_lower:
                found.add(symbol)

        # Ticker pattern: $SYMBOL
        ticker_matches = re.findall(r"\$([A-Z]{1,5})\b", text_upper)
        for match in ticker_matches:
            if match in known_symbols or len(match) <= 4:
                found.add(match)

        return list(found)

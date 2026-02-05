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
from collections import deque
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

        # Phase 2: VIX Contrarian Signal Support
        self._vix_current: float | None = None
        self._vix_ma: float | None = None  # 20-day moving average
        self._vix_history: deque[float] = deque(maxlen=50)  # For MA calculation (bounded)
        self._enable_vix_contrarian = config.parameters.get("enable_vix_contrarian", True)
        self._vix_extreme_high = config.parameters.get("vix_extreme_high", 30.0)
        self._vix_extreme_low = config.parameters.get("vix_extreme_low", 12.0)
        self._vix_contrarian_weight = config.parameters.get("vix_contrarian_weight", 0.3)

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
        # Extract current price for stop_loss/target_price calculations
        current_price = event.last or event.bid or event.ask or 100.0

        # Add symbol to monitoring list
        if symbol not in self._monitored_symbols:
            self._monitored_symbols.add(symbol)

        # Check if we should analyze this symbol
        if not self._should_analyze(symbol):
            # Still publish a signal to satisfy barrier
            await self._publish_neutral_signal(symbol, "Analysis interval not reached")
            return

        # Analyze sentiment for symbol
        signal = await self._analyze_symbol_sentiment(symbol, current_price)

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
                target_price=None,
                stop_loss=None,
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
            target_price=None,
            stop_loss=None,
        )
        await self._event_bus.publish_signal(signal)

    async def _analyze_symbol_sentiment(
        self, symbol: str, current_price: float
    ) -> SignalEvent | None:
        """
        Analyze sentiment for a specific symbol.

        Args:
            symbol: The symbol to analyze
            current_price: Current market price for stop_loss/target_price calculations

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

        # Phase 2: Combine with VIX contrarian signal
        if self._enable_vix_contrarian and aggregated["sentiment"] is not None:
            combined_sentiment, combined_conf, vix_rationale = self._combine_with_vix_signal(
                aggregated["sentiment"],
                aggregated["confidence"],
                symbol,
            )
            # Update aggregated with combined values
            aggregated["original_sentiment"] = aggregated["sentiment"]
            aggregated["sentiment"] = combined_sentiment
            aggregated["confidence"] = combined_conf
            aggregated["vix_contribution"] = vix_rationale

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
                target_price=None,
                stop_loss=None,
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

        # Calculate stop_loss and target_price based on direction
        # 2% stop_loss distance, 4% target_price distance (2:1 reward/risk ratio)
        if direction == SignalDirection.LONG:
            stop_loss = current_price * 0.98
            target_price = current_price * 1.04
        elif direction == SignalDirection.SHORT:
            stop_loss = current_price * 1.02
            target_price = current_price * 0.96
        else:  # FLAT
            stop_loss = None
            target_price = None

        return SignalEvent(
            source_agent=self.name,
            strategy_name="sentiment_llm",
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            rationale=rationale,
            data_sources=("llm_sentiment", "news_sentiment", "financial_news"),
            target_price=target_price,
            stop_loss=stop_loss,
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

    # =========================================================================
    # Phase 2: VIX Contrarian Signal Support
    # =========================================================================

    def update_vix(self, vix_value: float) -> None:
        """
        Update VIX level for contrarian signal generation.

        Call this method when VIX data is received (e.g., from VIX futures
        or ^VIX index data).

        Args:
            vix_value: Current VIX level
        """
        self._vix_current = vix_value
        self._vix_history.append(vix_value)

        # Keep last 20 days for MA calculation
        if len(self._vix_history) > 20:
            self._vix_history = self._vix_history[-20:]

        # Calculate 20-day MA
        if len(self._vix_history) >= 5:  # Minimum for meaningful MA
            self._vix_ma = sum(self._vix_history) / len(self._vix_history)

        logger.debug(f"VIX updated: current={vix_value:.1f}, MA={self._vix_ma:.1f if self._vix_ma else 'N/A'}")

    def get_vix_contrarian_signal(
        self,
        symbol: str,
        asset_class: str = "equity",
    ) -> tuple[float, float, str]:
        """
        Generate VIX-based contrarian signal for equity indices.

        Research basis:
        - VIX > 30: Extreme fear, historically contrarian bullish
        - VIX < 12: Extreme complacency, can precede corrections
        - VIX spike (current >> MA): Short-term oversold conditions

        Args:
            symbol: Symbol to generate signal for
            asset_class: Asset class (contrarian only for equity/index)

        Returns:
            (signal_strength, confidence, rationale) tuple
            signal_strength: -1 to 1 (positive = bullish, negative = bearish)
            confidence: 0 to 1
        """
        if not self._enable_vix_contrarian:
            return 0.0, 0.0, "VIX contrarian disabled"

        if self._vix_current is None:
            return 0.0, 0.0, "No VIX data available"

        # Only apply to equity/index instruments
        equity_symbols = {
            "SPY", "QQQ", "IWM", "DIA", "ES", "NQ", "YM", "RTY",
            "MES", "MNQ", "MYM", "M2K",
        }

        # Also apply to major stocks (they correlate with VIX)
        major_stocks = {
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"
        }

        is_applicable = (
            symbol in equity_symbols or
            symbol in major_stocks or
            asset_class in ("equity", "future", "index")
        )

        if not is_applicable:
            return 0.0, 0.0, f"VIX contrarian not applicable to {symbol}"

        vix = self._vix_current
        vix_ma = self._vix_ma or vix  # Use current if no MA

        signal = 0.0
        confidence = 0.0
        reasons = []

        # Extreme fear (VIX > 30) - contrarian bullish
        if vix >= self._vix_extreme_high:
            # Scale signal from 0.3 to 1.0 based on how extreme
            fear_excess = (vix - self._vix_extreme_high) / 20.0  # Normalize by 20 points
            signal = min(1.0, 0.3 + fear_excess * 0.7)
            confidence = min(0.8, 0.5 + fear_excess * 0.3)
            reasons.append(f"VIX extreme fear ({vix:.1f} >= {self._vix_extreme_high}): contrarian bullish")

        # Extreme complacency (VIX < 12) - warning signal
        elif vix <= self._vix_extreme_low:
            # Mild bearish signal for complacency
            complacency = (self._vix_extreme_low - vix) / 4.0  # Normalize
            signal = max(-0.5, -0.2 - complacency * 0.3)
            confidence = min(0.5, 0.3 + complacency * 0.2)
            reasons.append(f"VIX complacency ({vix:.1f} <= {self._vix_extreme_low}): caution")

        # VIX spike relative to MA (current >> MA)
        if vix_ma > 0:
            vix_ratio = vix / vix_ma
            if vix_ratio >= 1.5:
                # Significant spike - contrarian opportunity
                spike_signal = min(0.5, (vix_ratio - 1.5) * 0.5)
                signal = max(signal, spike_signal)
                confidence = max(confidence, 0.4)
                reasons.append(f"VIX spike (ratio={vix_ratio:.2f}x MA): contrarian bullish")
            elif vix_ratio <= 0.7:
                # VIX unusually low vs MA - caution
                reasons.append(f"VIX low vs MA (ratio={vix_ratio:.2f}): potential complacency")

        rationale = "; ".join(reasons) if reasons else f"VIX normal ({vix:.1f})"

        logger.debug(f"VIX contrarian signal for {symbol}: signal={signal:.2f}, conf={confidence:.2f}, {rationale}")

        return signal, confidence, rationale

    def _combine_with_vix_signal(
        self,
        news_sentiment: float,
        news_confidence: float,
        symbol: str,
        asset_class: str = "equity",
    ) -> tuple[float, float, str]:
        """
        Combine news sentiment with VIX contrarian signal.

        Uses weighted combination:
        - News sentiment: (1 - vix_weight) weight
        - VIX contrarian: vix_weight weight

        Args:
            news_sentiment: Sentiment from news analysis (-1 to 1)
            news_confidence: Confidence from news analysis (0 to 1)
            symbol: Trading symbol
            asset_class: Asset class

        Returns:
            (combined_sentiment, combined_confidence, rationale) tuple
        """
        vix_signal, vix_conf, vix_rationale = self.get_vix_contrarian_signal(
            symbol, asset_class
        )

        # If no VIX signal, return news sentiment unchanged
        if vix_conf == 0.0:
            return news_sentiment, news_confidence, f"News only: {vix_rationale}"

        # Weighted combination
        vix_weight = self._vix_contrarian_weight * vix_conf
        news_weight = (1.0 - self._vix_contrarian_weight) * news_confidence

        total_weight = vix_weight + news_weight
        if total_weight == 0:
            return 0.0, 0.0, "No signal"

        combined_sentiment = (
            vix_signal * vix_weight + news_sentiment * news_weight
        ) / total_weight

        # Confidence is max of the two, boosted slightly if they agree
        agreement_boost = 0.1 if (vix_signal * news_sentiment > 0) else 0.0
        combined_confidence = min(1.0, max(vix_conf, news_confidence) + agreement_boost)

        rationale = f"Combined: VIX={vix_signal:.2f} ({vix_rationale}), News={news_sentiment:.2f}"

        return combined_sentiment, combined_confidence, rationale

    def get_vix_status(self) -> dict[str, Any]:
        """Get current VIX tracking status."""
        return {
            "enabled": self._enable_vix_contrarian,
            "current": self._vix_current,
            "ma_20": self._vix_ma,
            "history_length": len(self._vix_history),
            "extreme_high_threshold": self._vix_extreme_high,
            "extreme_low_threshold": self._vix_extreme_low,
            "contrarian_weight": self._vix_contrarian_weight,
            "regime": self._get_vix_regime(),
        }

    def _get_vix_regime(self) -> str:
        """Classify current VIX regime."""
        if self._vix_current is None:
            return "unknown"

        vix = self._vix_current
        if vix >= 40:
            return "crisis"
        elif vix >= 30:
            return "fear"
        elif vix >= 20:
            return "elevated"
        elif vix >= 15:
            return "normal"
        elif vix >= 12:
            return "low"
        else:
            return "complacency"

    # =========================================================================
    # Phase 4: Advanced Composite Sentiment Scoring
    # =========================================================================

    def composite_sentiment_signal(
        self,
        vix: float | None = None,
        fear_greed: float | None = None,
        put_call_ratio: float | None = None,
        aaii_bullish_pct: float | None = None,
        ig_long_pct: float | None = None,
        cot_net_long: float | None = None,
    ) -> dict[str, Any]:
        """
        Generate composite contrarian signal from multiple sentiment indicators.

        Based on research finding: When 3+ indicators agree at extremes,
        contrarian signals achieve 65-75% win rate over 1-3 month horizons.

        Indicator thresholds based on historical extremes:
        - VIX: >30 = fear (buy), <12 = complacency (sell)
        - Fear & Greed: <25 = extreme fear (buy), >75 = extreme greed (sell)
        - Put/Call Ratio: >1.2 = bearish sentiment (buy), <0.5 = bullish (sell)
        - AAII Bullish%: <25% = extreme bearish (buy), >55% = extreme bullish (sell)
        - IG Client Sentiment: >80% long = crowded (sell), <20% long = contrarian (buy)
        - COT Net Long: Extreme readings indicate potential reversal

        Args:
            vix: VIX index value
            fear_greed: CNN Fear & Greed Index (0-100)
            put_call_ratio: CBOE Put/Call ratio
            aaii_bullish_pct: AAII survey bullish percentage (0-100)
            ig_long_pct: IG client long percentage (0-100)
            cot_net_long: COT report net long positioning (-1 to 1)

        Returns:
            Dict with composite signal, confidence, and component breakdown
        """
        signals = []
        indicator_details = {}

        # Use internal VIX if not provided
        if vix is None:
            vix = self._vix_current

        # VIX-based signal (most reliable contrarian indicator)
        if vix is not None:
            if vix >= 40:
                signals.append(("VIX", "BUY", 0.9, "Extreme fear - VIX > 40"))
                indicator_details["vix"] = {"signal": "BUY", "value": vix, "weight": 0.9}
            elif vix >= 30:
                signals.append(("VIX", "BUY", 0.7, "High fear - VIX > 30"))
                indicator_details["vix"] = {"signal": "BUY", "value": vix, "weight": 0.7}
            elif vix <= 12:
                signals.append(("VIX", "SELL", 0.5, "Complacency - VIX < 12"))
                indicator_details["vix"] = {"signal": "SELL", "value": vix, "weight": 0.5}
            elif vix <= 14:
                signals.append(("VIX", "SELL", 0.3, "Low vol warning - VIX < 14"))
                indicator_details["vix"] = {"signal": "SELL", "value": vix, "weight": 0.3}

        # Fear & Greed Index
        if fear_greed is not None:
            if fear_greed <= 20:
                signals.append(("FEAR_GREED", "BUY", 0.85, "Extreme fear"))
                indicator_details["fear_greed"] = {"signal": "BUY", "value": fear_greed, "weight": 0.85}
            elif fear_greed <= 25:
                signals.append(("FEAR_GREED", "BUY", 0.7, "Fear zone"))
                indicator_details["fear_greed"] = {"signal": "BUY", "value": fear_greed, "weight": 0.7}
            elif fear_greed >= 80:
                signals.append(("FEAR_GREED", "SELL", 0.75, "Extreme greed"))
                indicator_details["fear_greed"] = {"signal": "SELL", "value": fear_greed, "weight": 0.75}
            elif fear_greed >= 75:
                signals.append(("FEAR_GREED", "SELL", 0.6, "Greed zone"))
                indicator_details["fear_greed"] = {"signal": "SELL", "value": fear_greed, "weight": 0.6}

        # Put/Call Ratio (contrarian - high PCR = bearish sentiment = buy signal)
        if put_call_ratio is not None:
            if put_call_ratio >= 1.3:
                signals.append(("PCR", "BUY", 0.7, "Extreme bearish hedging"))
                indicator_details["pcr"] = {"signal": "BUY", "value": put_call_ratio, "weight": 0.7}
            elif put_call_ratio >= 1.2:
                signals.append(("PCR", "BUY", 0.6, "Elevated hedging"))
                indicator_details["pcr"] = {"signal": "BUY", "value": put_call_ratio, "weight": 0.6}
            elif put_call_ratio <= 0.5:
                signals.append(("PCR", "SELL", 0.65, "Extreme complacency"))
                indicator_details["pcr"] = {"signal": "SELL", "value": put_call_ratio, "weight": 0.65}
            elif put_call_ratio <= 0.6:
                signals.append(("PCR", "SELL", 0.5, "Low hedging"))
                indicator_details["pcr"] = {"signal": "SELL", "value": put_call_ratio, "weight": 0.5}

        # AAII Sentiment Survey (highest historical edge at extremes)
        if aaii_bullish_pct is not None:
            if aaii_bullish_pct <= 20:
                signals.append(("AAII", "BUY", 0.9, "Extreme bearishness - highest edge"))
                indicator_details["aaii"] = {"signal": "BUY", "value": aaii_bullish_pct, "weight": 0.9}
            elif aaii_bullish_pct <= 25:
                signals.append(("AAII", "BUY", 0.75, "High bearishness"))
                indicator_details["aaii"] = {"signal": "BUY", "value": aaii_bullish_pct, "weight": 0.75}
            elif aaii_bullish_pct >= 60:
                signals.append(("AAII", "SELL", 0.6, "Extreme bullishness"))
                indicator_details["aaii"] = {"signal": "SELL", "value": aaii_bullish_pct, "weight": 0.6}
            elif aaii_bullish_pct >= 55:
                signals.append(("AAII", "SELL", 0.5, "Elevated bullishness"))
                indicator_details["aaii"] = {"signal": "SELL", "value": aaii_bullish_pct, "weight": 0.5}

        # IG Client Sentiment (retail contrarian indicator)
        if ig_long_pct is not None:
            if ig_long_pct >= 85:
                signals.append(("IG_CLIENT", "SELL", 0.7, "Extreme retail long crowding"))
                indicator_details["ig_client"] = {"signal": "SELL", "value": ig_long_pct, "weight": 0.7}
            elif ig_long_pct >= 80:
                signals.append(("IG_CLIENT", "SELL", 0.6, "Retail long crowding"))
                indicator_details["ig_client"] = {"signal": "SELL", "value": ig_long_pct, "weight": 0.6}
            elif ig_long_pct <= 15:
                signals.append(("IG_CLIENT", "BUY", 0.7, "Retail extreme short"))
                indicator_details["ig_client"] = {"signal": "BUY", "value": ig_long_pct, "weight": 0.7}
            elif ig_long_pct <= 20:
                signals.append(("IG_CLIENT", "BUY", 0.6, "Retail short"))
                indicator_details["ig_client"] = {"signal": "BUY", "value": ig_long_pct, "weight": 0.6}

        # COT Report Net Long (commercial hedger positioning)
        if cot_net_long is not None:
            # COT in normalized -1 to 1 range
            if cot_net_long <= -0.8:
                signals.append(("COT", "BUY", 0.65, "Commercials extreme short (they hedge)"))
                indicator_details["cot"] = {"signal": "BUY", "value": cot_net_long, "weight": 0.65}
            elif cot_net_long >= 0.8:
                signals.append(("COT", "SELL", 0.65, "Commercials extreme long"))
                indicator_details["cot"] = {"signal": "SELL", "value": cot_net_long, "weight": 0.65}

        # Aggregate signals
        buy_signals = [s for s in signals if s[1] == "BUY"]
        sell_signals = [s for s in signals if s[1] == "SELL"]

        n_buy = len(buy_signals)
        n_sell = len(sell_signals)
        n_total = len(signals)

        # Calculate weighted scores
        buy_score = sum(s[2] for s in buy_signals) if buy_signals else 0
        sell_score = sum(s[2] for s in sell_signals) if sell_signals else 0

        # Determine composite signal
        if n_buy >= 3 and n_buy > n_sell:
            composite_signal = "STRONG_BUY"
            confidence = min(0.9, buy_score / n_buy)
            rationale = f"Strong contrarian buy: {n_buy} indicators at extremes"
        elif n_sell >= 3 and n_sell > n_buy:
            composite_signal = "STRONG_SELL"
            confidence = min(0.9, sell_score / n_sell)
            rationale = f"Strong contrarian sell: {n_sell} indicators at extremes"
        elif n_buy >= 2 and n_buy > n_sell:
            composite_signal = "BUY"
            confidence = min(0.7, buy_score / n_buy)
            rationale = f"Moderate buy: {n_buy} buy signals"
        elif n_sell >= 2 and n_sell > n_buy:
            composite_signal = "SELL"
            confidence = min(0.7, sell_score / n_sell)
            rationale = f"Moderate sell: {n_sell} sell signals"
        elif n_buy == 1 and n_sell == 0:
            composite_signal = "WEAK_BUY"
            confidence = buy_signals[0][2] * 0.5
            rationale = f"Weak buy signal from {buy_signals[0][0]}"
        elif n_sell == 1 and n_buy == 0:
            composite_signal = "WEAK_SELL"
            confidence = sell_signals[0][2] * 0.5
            rationale = f"Weak sell signal from {sell_signals[0][0]}"
        else:
            composite_signal = "NEUTRAL"
            confidence = 0.0
            rationale = "No clear signal or conflicting indicators"

        return {
            "signal": composite_signal,
            "confidence": confidence,
            "rationale": rationale,
            "n_buy_signals": n_buy,
            "n_sell_signals": n_sell,
            "n_total_signals": n_total,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "indicators": indicator_details,
            "components": [
                {"source": s[0], "signal": s[1], "weight": s[2], "reason": s[3]}
                for s in signals
            ],
        }

    def get_composite_sentiment_status(self) -> dict[str, Any]:
        """Get status of composite sentiment system."""
        # Use current VIX for basic status
        current_composite = self.composite_sentiment_signal(vix=self._vix_current)

        return {
            "enabled": True,
            "current_signal": current_composite["signal"],
            "current_confidence": current_composite["confidence"],
            "n_active_indicators": current_composite["n_total_signals"],
            "vix_value": self._vix_current,
            "vix_regime": self._get_vix_regime(),
            "note": "Pass additional indicators for full composite scoring",
        }

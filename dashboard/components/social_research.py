"""
Social Research Tracker
=======================

Track and analyze social media sentiment and research for trading decisions.

Integrates with:
- Twitter/X API (posts, sentiment, influencer activity)
- Reddit API (posts, comments, subreddit activity)
- StockTwits API (messages, sentiment indicators)

Features:
- Aggregate sentiment by symbol across platforms
- Detect unusual social activity (volume spikes, sentiment shifts)
- Track influencer activity and impact
- Export for dashboard charts and feeds
"""

from __future__ import annotations

import logging
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Protocol
import uuid


logger = logging.getLogger(__name__)


class Platform(Enum):
    """Social media platform identifiers."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"


class SentimentCategory(Enum):
    """Sentiment classification categories."""
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


@dataclass
class SocialPost:
    """
    Represents a single social media post.

    Captures content, engagement metrics, and sentiment analysis.
    """
    post_id: str
    platform: Platform
    author: str
    content: str
    timestamp: datetime
    symbol_mentions: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0  # -1.0 (bearish) to 1.0 (bullish)
    engagement: dict[str, int] = field(default_factory=dict)  # likes, retweets, comments
    relevance_score: float = 0.0  # 0.0 to 1.0
    # Additional metadata
    url: str = ""
    is_verified_author: bool = False
    author_follower_count: int = 0
    hashtags: list[str] = field(default_factory=list)
    cashtags: list[str] = field(default_factory=list)
    reply_to_id: str | None = None
    is_repost: bool = False
    original_post_id: str | None = None

    @property
    def total_engagement(self) -> int:
        """Calculate total engagement count."""
        return sum(self.engagement.values())

    @property
    def sentiment_category(self) -> SentimentCategory:
        """Categorize sentiment score."""
        if self.sentiment_score <= -0.6:
            return SentimentCategory.VERY_BEARISH
        elif self.sentiment_score <= -0.2:
            return SentimentCategory.BEARISH
        elif self.sentiment_score <= 0.2:
            return SentimentCategory.NEUTRAL
        elif self.sentiment_score <= 0.6:
            return SentimentCategory.BULLISH
        else:
            return SentimentCategory.VERY_BULLISH

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "post_id": self.post_id,
            "platform": self.platform.value,
            "author": self.author,
            "content": self.content[:500] if len(self.content) > 500 else self.content,
            "timestamp": self.timestamp.isoformat(),
            "symbol_mentions": self.symbol_mentions,
            "sentiment_score": round(self.sentiment_score, 3),
            "sentiment_category": self.sentiment_category.value,
            "engagement": self.engagement,
            "total_engagement": self.total_engagement,
            "relevance_score": round(self.relevance_score, 3),
            "url": self.url,
            "is_verified_author": self.is_verified_author,
            "author_follower_count": self.author_follower_count,
            "hashtags": self.hashtags,
            "cashtags": self.cashtags,
        }


@dataclass
class SocialSentiment:
    """
    Aggregated sentiment for a symbol on a specific platform.

    Provides summary metrics for dashboard display.
    """
    symbol: str
    platform: Platform
    sentiment_score: float = 0.0  # Weighted average sentiment
    post_count: int = 0
    trending_score: float = 0.0  # 0.0 to 1.0 based on activity vs baseline
    key_themes: list[str] = field(default_factory=list)
    # Time-based metrics
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Engagement metrics
    total_engagement: int = 0
    avg_engagement_per_post: float = 0.0
    # Sentiment distribution
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    # Top contributors
    top_authors: list[str] = field(default_factory=list)

    @property
    def sentiment_category(self) -> SentimentCategory:
        """Categorize aggregated sentiment."""
        if self.sentiment_score <= -0.6:
            return SentimentCategory.VERY_BEARISH
        elif self.sentiment_score <= -0.2:
            return SentimentCategory.BEARISH
        elif self.sentiment_score <= 0.2:
            return SentimentCategory.NEUTRAL
        elif self.sentiment_score <= 0.6:
            return SentimentCategory.BULLISH
        else:
            return SentimentCategory.VERY_BULLISH

    @property
    def bullish_ratio(self) -> float:
        """Calculate ratio of bullish posts."""
        total = self.bullish_count + self.bearish_count + self.neutral_count
        return self.bullish_count / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "platform": self.platform.value,
            "sentiment_score": round(self.sentiment_score, 3),
            "sentiment_category": self.sentiment_category.value,
            "post_count": self.post_count,
            "trending_score": round(self.trending_score, 3),
            "key_themes": self.key_themes[:10],
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_engagement": self.total_engagement,
            "avg_engagement_per_post": round(self.avg_engagement_per_post, 2),
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "bullish_ratio": round(self.bullish_ratio, 3),
            "top_authors": self.top_authors[:5],
        }


@dataclass
class TrendingTopic:
    """
    Represents a trending topic or theme in social media.
    """
    topic: str
    mentions: int = 0
    sentiment_score: float = 0.0
    related_symbols: list[str] = field(default_factory=list)
    platforms: list[Platform] = field(default_factory=list)
    velocity: float = 0.0  # Rate of increase in mentions
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "topic": self.topic,
            "mentions": self.mentions,
            "sentiment_score": round(self.sentiment_score, 3),
            "related_symbols": self.related_symbols,
            "platforms": [p.value for p in self.platforms],
            "velocity": round(self.velocity, 3),
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }


@dataclass
class InfluencerActivity:
    """
    Tracks activity of influential social media users.
    """
    author: str
    platform: Platform
    follower_count: int = 0
    is_verified: bool = False
    recent_posts: int = 0
    avg_sentiment: float = 0.0
    symbols_mentioned: list[str] = field(default_factory=list)
    avg_engagement: float = 0.0
    influence_score: float = 0.0  # 0.0 to 1.0
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "author": self.author,
            "platform": self.platform.value,
            "follower_count": self.follower_count,
            "is_verified": self.is_verified,
            "recent_posts": self.recent_posts,
            "avg_sentiment": round(self.avg_sentiment, 3),
            "symbols_mentioned": self.symbols_mentioned,
            "avg_engagement": round(self.avg_engagement, 2),
            "influence_score": round(self.influence_score, 3),
            "last_active": self.last_active.isoformat(),
        }


@dataclass
class SocialAlert:
    """
    Alert for unusual social media activity.
    """
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: str = ""  # "volume_spike", "sentiment_shift", "influencer_post", "trending"
    symbol: str | None = None
    platform: Platform | None = None
    severity: str = "info"  # "info", "warning", "critical"
    message: str = ""
    current_value: float = 0.0
    baseline_value: float = 0.0
    threshold: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    related_posts: list[str] = field(default_factory=list)  # Post IDs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "symbol": self.symbol,
            "platform": self.platform.value if self.platform else None,
            "severity": self.severity,
            "message": self.message,
            "current_value": round(self.current_value, 3),
            "baseline_value": round(self.baseline_value, 3),
            "threshold": round(self.threshold, 3),
            "timestamp": self.timestamp.isoformat(),
            "related_posts": self.related_posts[:10],
        }


# =============================================================================
# API INTERFACES (Protocols)
# =============================================================================


class SocialMediaAPI(Protocol):
    """Protocol for social media API implementations."""

    def get_posts(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[SocialPost]:
        """Fetch posts from the platform."""
        ...

    def get_user_info(self, username: str) -> dict[str, Any]:
        """Get user information."""
        ...

    def search(self, query: str, limit: int = 100) -> list[SocialPost]:
        """Search for posts matching a query."""
        ...


# =============================================================================
# MOCK API IMPLEMENTATIONS
# =============================================================================


class MockTwitterAPI:
    """
    Mock Twitter/X API implementation.

    Provides interface-compatible mock for development and testing.
    Replace with real API calls in production.
    """

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self._connected = False
        logger.info("MockTwitterAPI initialized")

    def connect(self) -> bool:
        """Simulate API connection."""
        self._connected = True
        logger.info("MockTwitterAPI connected (mock)")
        return True

    def get_posts(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[SocialPost]:
        """
        Mock implementation - returns empty list.

        In production, this would call Twitter API v2 endpoints:
        - GET /2/tweets/search/recent for recent tweets
        - GET /2/tweets/search/all for historical tweets (Academic access)
        """
        logger.debug(f"MockTwitterAPI.get_posts called: symbol={symbol}, limit={limit}")
        return []

    def get_user_info(self, username: str) -> dict[str, Any]:
        """Mock user info retrieval."""
        return {
            "username": username,
            "platform": Platform.TWITTER.value,
            "follower_count": 0,
            "is_verified": False,
            "exists": False,
        }

    def search(self, query: str, limit: int = 100) -> list[SocialPost]:
        """Mock search implementation."""
        logger.debug(f"MockTwitterAPI.search called: query={query}, limit={limit}")
        return []

    def get_trending(self, woeid: int = 1) -> list[str]:
        """Mock trending topics. WOEID 1 = Worldwide."""
        return []


class MockRedditAPI:
    """
    Mock Reddit API implementation.

    Provides interface-compatible mock for development and testing.
    Replace with PRAW (Python Reddit API Wrapper) in production.
    """

    def __init__(self, client_id: str = "", client_secret: str = ""):
        self.client_id = client_id
        self.client_secret = client_secret
        self._connected = False
        logger.info("MockRedditAPI initialized")

    def connect(self) -> bool:
        """Simulate API connection."""
        self._connected = True
        logger.info("MockRedditAPI connected (mock)")
        return True

    def get_posts(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
        subreddits: list[str] | None = None,
    ) -> list[SocialPost]:
        """
        Mock implementation - returns empty list.

        In production, this would use PRAW to fetch from:
        - r/wallstreetbets
        - r/stocks
        - r/investing
        - r/options
        - r/SecurityAnalysis
        """
        if subreddits is None:
            subreddits = ["wallstreetbets", "stocks", "investing"]
        logger.debug(f"MockRedditAPI.get_posts called: symbol={symbol}, subreddits={subreddits}")
        return []

    def get_user_info(self, username: str) -> dict[str, Any]:
        """Mock user info retrieval."""
        return {
            "username": username,
            "platform": Platform.REDDIT.value,
            "karma": 0,
            "account_age_days": 0,
            "exists": False,
        }

    def search(self, query: str, limit: int = 100) -> list[SocialPost]:
        """Mock search implementation."""
        logger.debug(f"MockRedditAPI.search called: query={query}, limit={limit}")
        return []

    def get_subreddit_stats(self, subreddit: str) -> dict[str, Any]:
        """Mock subreddit statistics."""
        return {
            "subreddit": subreddit,
            "subscribers": 0,
            "active_users": 0,
        }


class MockStockTwitsAPI:
    """
    Mock StockTwits API implementation.

    Provides interface-compatible mock for development and testing.
    """

    def __init__(self, access_token: str = ""):
        self.access_token = access_token
        self._connected = False
        logger.info("MockStockTwitsAPI initialized")

    def connect(self) -> bool:
        """Simulate API connection."""
        self._connected = True
        logger.info("MockStockTwitsAPI connected (mock)")
        return True

    def get_posts(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[SocialPost]:
        """
        Mock implementation - returns empty list.

        In production, this would call StockTwits API:
        - GET /api/2/streams/symbol/{symbol}.json
        - GET /api/2/streams/trending.json
        """
        logger.debug(f"MockStockTwitsAPI.get_posts called: symbol={symbol}, limit={limit}")
        return []

    def get_user_info(self, username: str) -> dict[str, Any]:
        """Mock user info retrieval."""
        return {
            "username": username,
            "platform": Platform.STOCKTWITS.value,
            "followers": 0,
            "following": 0,
            "ideas": 0,
            "exists": False,
        }

    def search(self, query: str, limit: int = 100) -> list[SocialPost]:
        """Mock search implementation."""
        logger.debug(f"MockStockTwitsAPI.search called: query={query}, limit={limit}")
        return []

    def get_symbol_sentiment(self, symbol: str) -> dict[str, Any]:
        """Mock symbol sentiment from StockTwits."""
        return {
            "symbol": symbol,
            "sentiment": None,  # "Bullish", "Bearish", or None
            "messages_today": 0,
        }

    def get_trending_symbols(self) -> list[str]:
        """Mock trending symbols."""
        return []


# =============================================================================
# SENTIMENT ANALYZER
# =============================================================================


class SentimentAnalyzer:
    """
    Basic sentiment analysis for social media posts.

    Uses keyword-based analysis as a baseline.
    In production, replace with ML model (FinBERT, etc.)
    """

    # Sentiment keywords (simplified)
    BULLISH_KEYWORDS = {
        "buy", "long", "bullish", "moon", "rocket", "pump", "calls",
        "breakout", "rally", "green", "gains", "profit", "upside",
        "strong", "growth", "beat", "outperform", "upgrade", "buy the dip",
        "hodl", "diamond hands", "to the moon", "ath", "all time high",
    }

    BEARISH_KEYWORDS = {
        "sell", "short", "bearish", "dump", "crash", "puts", "drop",
        "fall", "red", "loss", "downside", "weak", "decline", "miss",
        "underperform", "downgrade", "overvalued", "bubble", "correction",
        "paper hands", "bag holder", "dead cat", "drill", "tank",
    }

    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score from -1.0 (bearish) to 1.0 (bullish)
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        words = set(text_lower.split())

        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        score = (bullish_count - bearish_count) / total
        return max(-1.0, min(1.0, score))

    def extract_symbols(self, text: str) -> list[str]:
        """
        Extract stock symbols from text.

        Looks for $SYMBOL cashtag format.
        """
        import re
        cashtags = re.findall(r'\$([A-Z]{1,5})\b', text.upper())
        return list(set(cashtags))


# =============================================================================
# MAIN TRACKER CLASS
# =============================================================================


class SocialResearchTracker:
    """
    Main tracker for social media sentiment and research.

    Aggregates data from multiple platforms, detects anomalies,
    and exports for dashboard visualization.

    Usage:
        tracker = SocialResearchTracker()

        # Record posts (from API or manual)
        tracker.record_post(post)

        # Get sentiment for a symbol
        sentiment = tracker.get_sentiment_summary("AAPL")

        # Get trending topics
        topics = tracker.get_trending_topics()

        # Detect anomalies
        alerts = tracker.detect_anomalies()

        # Export for dashboard
        data = tracker.to_dict()
    """

    # Configuration defaults
    MAX_POSTS = 10000  # Maximum posts to keep in memory
    ANOMALY_VOLUME_THRESHOLD = 2.0  # 2x baseline = volume spike
    ANOMALY_SENTIMENT_THRESHOLD = 0.3  # 0.3 shift = sentiment change
    BASELINE_WINDOW_HOURS = 24  # Hours for baseline calculation
    RECENT_WINDOW_MINUTES = 60  # Minutes for "recent" activity

    def __init__(
        self,
        twitter_api: MockTwitterAPI | None = None,
        reddit_api: MockRedditAPI | None = None,
        stocktwits_api: MockStockTwitsAPI | None = None,
        max_posts: int = MAX_POSTS,
    ):
        """
        Initialize the social research tracker.

        Args:
            twitter_api: Twitter API client (mock or real)
            reddit_api: Reddit API client (mock or real)
            stocktwits_api: StockTwits API client (mock or real)
            max_posts: Maximum posts to keep in memory
        """
        self._max_posts = max_posts

        # API clients (use mocks if not provided)
        self._twitter_api = twitter_api or MockTwitterAPI()
        self._reddit_api = reddit_api or MockRedditAPI()
        self._stocktwits_api = stocktwits_api or MockStockTwitsAPI()

        # Sentiment analyzer
        self._sentiment_analyzer = SentimentAnalyzer()

        # Storage
        self._posts: deque[SocialPost] = deque(maxlen=max_posts)
        self._posts_by_id: dict[str, SocialPost] = {}
        self._posts_by_symbol: dict[str, list[str]] = defaultdict(list)
        self._posts_by_platform: dict[Platform, list[str]] = defaultdict(list)
        self._posts_by_author: dict[str, list[str]] = defaultdict(list)

        # Baseline tracking for anomaly detection
        self._hourly_volume: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=168))  # 1 week
        self._hourly_sentiment: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=168))

        # Trending topics
        self._topic_mentions: dict[str, int] = defaultdict(int)
        self._topic_first_seen: dict[str, datetime] = {}

        # Alerts
        self._alerts: deque[SocialAlert] = deque(maxlen=1000)

        # Statistics
        self._total_posts_recorded = 0

        logger.info(f"SocialResearchTracker initialized with max_posts={max_posts}")

    def record_post(self, post: SocialPost) -> None:
        """
        Record a social media post.

        Args:
            post: The post to record
        """
        # Avoid duplicates
        if post.post_id in self._posts_by_id:
            logger.debug(f"Duplicate post ignored: {post.post_id}")
            return

        # Analyze sentiment if not already set
        if post.sentiment_score == 0.0 and post.content:
            post = SocialPost(
                post_id=post.post_id,
                platform=post.platform,
                author=post.author,
                content=post.content,
                timestamp=post.timestamp,
                symbol_mentions=post.symbol_mentions or self._sentiment_analyzer.extract_symbols(post.content),
                sentiment_score=self._sentiment_analyzer.analyze(post.content),
                engagement=post.engagement,
                relevance_score=post.relevance_score,
                url=post.url,
                is_verified_author=post.is_verified_author,
                author_follower_count=post.author_follower_count,
                hashtags=post.hashtags,
                cashtags=post.cashtags,
                reply_to_id=post.reply_to_id,
                is_repost=post.is_repost,
                original_post_id=post.original_post_id,
            )

        # Store post
        self._posts.append(post)
        self._posts_by_id[post.post_id] = post

        # Index by symbol
        for symbol in post.symbol_mentions:
            self._posts_by_symbol[symbol].append(post.post_id)

        # Index by platform
        self._posts_by_platform[post.platform].append(post.post_id)

        # Index by author
        self._posts_by_author[post.author].append(post.post_id)

        # Track topics (hashtags)
        for hashtag in post.hashtags:
            topic = hashtag.lower()
            self._topic_mentions[topic] += 1
            if topic not in self._topic_first_seen:
                self._topic_first_seen[topic] = post.timestamp

        self._total_posts_recorded += 1

        logger.debug(
            f"Recorded post {post.post_id[:8]} from {post.platform.value} "
            f"by {post.author}: sentiment={post.sentiment_score:.2f}"
        )

    def get_posts_by_symbol(
        self,
        symbol: str,
        platform: Platform | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[SocialPost]:
        """
        Get posts mentioning a specific symbol.

        Args:
            symbol: Stock symbol to search for
            platform: Filter by platform (optional)
            since: Only posts after this time (optional)
            limit: Maximum posts to return

        Returns:
            List of matching SocialPost objects
        """
        symbol = symbol.upper()
        post_ids = self._posts_by_symbol.get(symbol, [])

        posts = []
        for post_id in reversed(post_ids):  # Most recent first
            post = self._posts_by_id.get(post_id)
            if not post:
                continue

            if platform and post.platform != platform:
                continue

            if since and post.timestamp < since:
                continue

            posts.append(post)

            if len(posts) >= limit:
                break

        return posts

    def get_sentiment_summary(
        self,
        symbol: str,
        platform: Platform | None = None,
        hours: int = 24,
    ) -> SocialSentiment | dict[Platform, SocialSentiment]:
        """
        Get aggregated sentiment for a symbol.

        Args:
            symbol: Stock symbol
            platform: Specific platform (None = all platforms)
            hours: Time window in hours

        Returns:
            SocialSentiment for specific platform, or dict of all platforms
        """
        symbol = symbol.upper()
        since = datetime.now(timezone.utc) - timedelta(hours=hours)

        if platform:
            return self._calculate_sentiment(symbol, platform, since)

        # Calculate for all platforms
        result = {}
        for p in Platform:
            sentiment = self._calculate_sentiment(symbol, p, since)
            if sentiment.post_count > 0:
                result[p] = sentiment

        return result

    def _calculate_sentiment(
        self,
        symbol: str,
        platform: Platform,
        since: datetime,
    ) -> SocialSentiment:
        """Calculate sentiment for a symbol on a specific platform."""
        posts = self.get_posts_by_symbol(symbol, platform=platform, since=since, limit=10000)

        if not posts:
            return SocialSentiment(
                symbol=symbol,
                platform=platform,
                period_start=since,
                period_end=datetime.now(timezone.utc),
            )

        # Calculate weighted sentiment (weight by engagement and relevance)
        total_weight = 0.0
        weighted_sentiment = 0.0
        total_engagement = 0
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        author_counts: dict[str, int] = defaultdict(int)
        themes: dict[str, int] = defaultdict(int)

        for post in posts:
            # Weight by engagement + relevance
            weight = 1.0 + (post.total_engagement * 0.01) + (post.relevance_score * 2.0)
            weighted_sentiment += post.sentiment_score * weight
            total_weight += weight

            total_engagement += post.total_engagement
            author_counts[post.author] += 1

            # Count sentiment categories
            if post.sentiment_score > 0.2:
                bullish_count += 1
            elif post.sentiment_score < -0.2:
                bearish_count += 1
            else:
                neutral_count += 1

            # Track themes (hashtags)
            for tag in post.hashtags:
                themes[tag.lower()] += 1

        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0

        # Calculate trending score (posts vs baseline)
        baseline_posts = self._get_baseline_volume(symbol, platform)
        trending_score = min(1.0, len(posts) / max(baseline_posts, 1)) if baseline_posts else 0.5

        # Get top themes and authors
        top_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
        top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)

        return SocialSentiment(
            symbol=symbol,
            platform=platform,
            sentiment_score=avg_sentiment,
            post_count=len(posts),
            trending_score=trending_score,
            key_themes=[t[0] for t in top_themes[:10]],
            period_start=since,
            period_end=datetime.now(timezone.utc),
            total_engagement=total_engagement,
            avg_engagement_per_post=total_engagement / len(posts) if posts else 0.0,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            top_authors=[a[0] for a in top_authors[:5]],
        )

    def get_aggregate_sentiment(
        self,
        symbol: str,
        hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get aggregated sentiment across all platforms for a symbol.

        Args:
            symbol: Stock symbol
            hours: Time window in hours

        Returns:
            Dictionary with combined sentiment metrics
        """
        sentiments = self.get_sentiment_summary(symbol, hours=hours)

        if not sentiments:
            return {
                "symbol": symbol.upper(),
                "overall_sentiment": 0.0,
                "total_posts": 0,
                "platforms": {},
            }

        # Combine sentiments weighted by post count
        total_posts = 0
        weighted_sentiment = 0.0
        platform_data = {}

        for platform, sentiment in sentiments.items():
            total_posts += sentiment.post_count
            weighted_sentiment += sentiment.sentiment_score * sentiment.post_count
            platform_data[platform.value] = sentiment.to_dict()

        overall_sentiment = weighted_sentiment / total_posts if total_posts > 0 else 0.0

        return {
            "symbol": symbol.upper(),
            "overall_sentiment": round(overall_sentiment, 3),
            "total_posts": total_posts,
            "platforms": platform_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_trending_topics(
        self,
        limit: int = 20,
        min_mentions: int = 5,
    ) -> list[TrendingTopic]:
        """
        Get current trending topics.

        Args:
            limit: Maximum topics to return
            min_mentions: Minimum mentions to be considered trending

        Returns:
            List of TrendingTopic objects
        """
        now = datetime.now(timezone.utc)
        one_hour_ago = now - timedelta(hours=1)

        topics = []
        for topic, mentions in sorted(
            self._topic_mentions.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit * 2]:  # Get extra to filter
            if mentions < min_mentions:
                continue

            # Find related posts to extract symbols and sentiment
            related_symbols = set()
            sentiments = []
            platforms_seen = set()

            for post in self._posts:
                if topic in [h.lower() for h in post.hashtags]:
                    related_symbols.update(post.symbol_mentions)
                    sentiments.append(post.sentiment_score)
                    platforms_seen.add(post.platform)

            first_seen = self._topic_first_seen.get(topic, now)
            hours_active = max(1, (now - first_seen).total_seconds() / 3600)
            velocity = mentions / hours_active

            topics.append(TrendingTopic(
                topic=topic,
                mentions=mentions,
                sentiment_score=statistics.mean(sentiments) if sentiments else 0.0,
                related_symbols=list(related_symbols)[:10],
                platforms=list(platforms_seen),
                velocity=velocity,
                first_seen=first_seen,
                last_seen=now,
            ))

            if len(topics) >= limit:
                break

        return topics

    def get_influencer_activity(
        self,
        platform: Platform | None = None,
        min_followers: int = 10000,
        hours: int = 24,
        limit: int = 20,
    ) -> list[InfluencerActivity]:
        """
        Get activity of influential users.

        Args:
            platform: Filter by platform (optional)
            min_followers: Minimum followers to be considered influencer
            hours: Time window in hours
            limit: Maximum influencers to return

        Returns:
            List of InfluencerActivity objects
        """
        since = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Aggregate activity by author
        author_data: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "posts": [],
            "symbols": set(),
            "engagement_total": 0,
            "followers": 0,
            "verified": False,
            "platform": None,
        })

        for post in self._posts:
            if post.timestamp < since:
                continue

            if platform and post.platform != platform:
                continue

            if post.author_follower_count < min_followers:
                continue

            data = author_data[post.author]
            data["posts"].append(post)
            data["symbols"].update(post.symbol_mentions)
            data["engagement_total"] += post.total_engagement
            data["followers"] = max(data["followers"], post.author_follower_count)
            data["verified"] = data["verified"] or post.is_verified_author
            data["platform"] = post.platform

        # Build influencer activity objects
        influencers = []
        for author, data in author_data.items():
            posts = data["posts"]
            if not posts:
                continue

            sentiments = [p.sentiment_score for p in posts]
            engagements = [p.total_engagement for p in posts]

            # Calculate influence score (normalized)
            follower_score = min(1.0, data["followers"] / 1000000)  # 1M followers = max
            engagement_score = min(1.0, data["engagement_total"] / 10000)  # 10K engagement = max
            activity_score = min(1.0, len(posts) / 10)  # 10 posts = max
            influence_score = (follower_score * 0.5 + engagement_score * 0.3 + activity_score * 0.2)

            influencers.append(InfluencerActivity(
                author=author,
                platform=data["platform"],
                follower_count=data["followers"],
                is_verified=data["verified"],
                recent_posts=len(posts),
                avg_sentiment=statistics.mean(sentiments) if sentiments else 0.0,
                symbols_mentioned=list(data["symbols"])[:10],
                avg_engagement=statistics.mean(engagements) if engagements else 0.0,
                influence_score=influence_score,
                last_active=max(p.timestamp for p in posts),
            ))

        # Sort by influence score
        influencers.sort(key=lambda x: x.influence_score, reverse=True)
        return influencers[:limit]

    def detect_anomalies(
        self,
        symbols: list[str] | None = None,
    ) -> list[SocialAlert]:
        """
        Detect unusual social media activity.

        Checks for:
        - Volume spikes (sudden increase in posts)
        - Sentiment shifts (rapid change in sentiment)
        - Influencer activity (high-impact posts)

        Args:
            symbols: Symbols to check (None = all tracked)

        Returns:
            List of SocialAlert objects
        """
        if symbols is None:
            symbols = list(self._posts_by_symbol.keys())

        alerts = []
        now = datetime.now(timezone.utc)

        for symbol in symbols:
            for platform in Platform:
                # Get current and baseline metrics
                current_volume = self._get_recent_volume(symbol, platform, minutes=60)
                baseline_volume = self._get_baseline_volume(symbol, platform)

                if baseline_volume > 0:
                    volume_ratio = current_volume / baseline_volume

                    # Volume spike detection
                    if volume_ratio >= self.ANOMALY_VOLUME_THRESHOLD:
                        alert = SocialAlert(
                            alert_type="volume_spike",
                            symbol=symbol,
                            platform=platform,
                            severity="warning" if volume_ratio < 3.0 else "critical",
                            message=f"{symbol} social volume spike: {volume_ratio:.1f}x baseline on {platform.value}",
                            current_value=current_volume,
                            baseline_value=baseline_volume,
                            threshold=self.ANOMALY_VOLUME_THRESHOLD,
                            related_posts=self._get_recent_post_ids(symbol, platform, limit=10),
                        )
                        alerts.append(alert)
                        self._alerts.append(alert)

                # Sentiment shift detection
                current_sentiment = self._get_recent_sentiment(symbol, platform, minutes=60)
                baseline_sentiment = self._get_baseline_sentiment(symbol, platform)

                if current_sentiment is not None and baseline_sentiment is not None:
                    sentiment_shift = abs(current_sentiment - baseline_sentiment)

                    if sentiment_shift >= self.ANOMALY_SENTIMENT_THRESHOLD:
                        direction = "bullish" if current_sentiment > baseline_sentiment else "bearish"
                        alert = SocialAlert(
                            alert_type="sentiment_shift",
                            symbol=symbol,
                            platform=platform,
                            severity="warning" if sentiment_shift < 0.5 else "critical",
                            message=f"{symbol} sentiment shift to {direction}: {sentiment_shift:.2f} change on {platform.value}",
                            current_value=current_sentiment,
                            baseline_value=baseline_sentiment,
                            threshold=self.ANOMALY_SENTIMENT_THRESHOLD,
                            related_posts=self._get_recent_post_ids(symbol, platform, limit=10),
                        )
                        alerts.append(alert)
                        self._alerts.append(alert)

        return alerts

    def _get_recent_volume(
        self,
        symbol: str,
        platform: Platform,
        minutes: int = 60,
    ) -> int:
        """Get post count for symbol in recent time window."""
        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        posts = self.get_posts_by_symbol(symbol, platform=platform, since=since, limit=10000)
        return len(posts)

    def _get_baseline_volume(
        self,
        symbol: str,
        platform: Platform,
    ) -> float:
        """Get baseline hourly volume for symbol."""
        key = f"{symbol}:{platform.value}"
        volumes = self._hourly_volume.get(key)
        if not volumes or len(volumes) < 24:
            # Not enough data, return a conservative estimate
            return 10.0
        return statistics.mean(volumes)

    def _get_recent_sentiment(
        self,
        symbol: str,
        platform: Platform,
        minutes: int = 60,
    ) -> float | None:
        """Get average sentiment for symbol in recent time window."""
        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        posts = self.get_posts_by_symbol(symbol, platform=platform, since=since, limit=10000)
        if not posts:
            return None
        return statistics.mean(p.sentiment_score for p in posts)

    def _get_baseline_sentiment(
        self,
        symbol: str,
        platform: Platform,
    ) -> float | None:
        """Get baseline sentiment for symbol."""
        key = f"{symbol}:{platform.value}"
        sentiments = self._hourly_sentiment.get(key)
        if not sentiments or len(sentiments) < 24:
            return None
        return statistics.mean(sentiments)

    def _get_recent_post_ids(
        self,
        symbol: str,
        platform: Platform,
        limit: int = 10,
    ) -> list[str]:
        """Get IDs of recent posts for a symbol."""
        posts = self.get_posts_by_symbol(symbol, platform=platform, limit=limit)
        return [p.post_id for p in posts]

    def update_baselines(self) -> None:
        """
        Update baseline metrics.

        Should be called hourly to maintain accurate baselines.
        """
        now = datetime.now(timezone.utc)
        one_hour_ago = now - timedelta(hours=1)

        for symbol in self._posts_by_symbol.keys():
            for platform in Platform:
                # Update volume baseline
                posts = self.get_posts_by_symbol(
                    symbol, platform=platform, since=one_hour_ago, limit=10000
                )
                key = f"{symbol}:{platform.value}"
                self._hourly_volume[key].append(len(posts))

                # Update sentiment baseline
                if posts:
                    avg_sentiment = statistics.mean(p.sentiment_score for p in posts)
                    self._hourly_sentiment[key].append(avg_sentiment)

        logger.info("Updated social research baselines")

    def get_alerts(
        self,
        limit: int = 100,
        severity: str | None = None,
    ) -> list[SocialAlert]:
        """
        Get recent alerts.

        Args:
            limit: Maximum alerts to return
            severity: Filter by severity (optional)

        Returns:
            List of SocialAlert objects
        """
        alerts = list(self._alerts)
        alerts.reverse()  # Most recent first

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts[:limit]

    def get_dashboard_summary(self) -> dict[str, Any]:
        """
        Get summary data for dashboard widgets.

        Returns:
            Dictionary with key metrics for dashboard display
        """
        now = datetime.now(timezone.utc)
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(hours=24)

        # Count recent posts
        posts_last_hour = sum(1 for p in self._posts if p.timestamp >= one_hour_ago)
        posts_last_day = sum(1 for p in self._posts if p.timestamp >= one_day_ago)

        # Get platform breakdown
        platform_counts = {}
        for platform in Platform:
            count = len(self._posts_by_platform.get(platform, []))
            platform_counts[platform.value] = count

        # Get top symbols by post count (last 24h)
        symbol_counts: dict[str, int] = defaultdict(int)
        for post in self._posts:
            if post.timestamp >= one_day_ago:
                for symbol in post.symbol_mentions:
                    symbol_counts[symbol] += 1

        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Get recent alerts
        recent_alerts = [a.to_dict() for a in self.get_alerts(limit=5)]

        return {
            "posts_last_hour": posts_last_hour,
            "posts_last_day": posts_last_day,
            "total_posts_tracked": len(self._posts),
            "total_posts_recorded": self._total_posts_recorded,
            "platform_breakdown": platform_counts,
            "top_symbols": [{"symbol": s, "count": c} for s, c in top_symbols],
            "trending_topics": [t.to_dict() for t in self.get_trending_topics(limit=5)],
            "recent_alerts": recent_alerts,
            "timestamp": now.isoformat(),
        }

    def export_for_chart(
        self,
        symbol: str,
        hours: int = 24,
        interval_minutes: int = 60,
    ) -> dict[str, Any]:
        """
        Export time-series data for chart visualization.

        Args:
            symbol: Stock symbol
            hours: Hours of data to export
            interval_minutes: Time bucket size in minutes

        Returns:
            Dictionary with time-series arrays
        """
        symbol = symbol.upper()
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours)

        # Initialize time buckets
        num_buckets = (hours * 60) // interval_minutes
        timestamps = []
        post_counts = []
        sentiment_scores = []
        engagement_totals = []

        for i in range(num_buckets):
            bucket_start = start + timedelta(minutes=i * interval_minutes)
            bucket_end = bucket_start + timedelta(minutes=interval_minutes)

            timestamps.append(bucket_start.isoformat())

            # Get posts in this bucket
            bucket_posts = [
                p for p in self._posts
                if symbol in p.symbol_mentions
                and bucket_start <= p.timestamp < bucket_end
            ]

            post_counts.append(len(bucket_posts))

            if bucket_posts:
                sentiment_scores.append(
                    round(statistics.mean(p.sentiment_score for p in bucket_posts), 3)
                )
                engagement_totals.append(sum(p.total_engagement for p in bucket_posts))
            else:
                sentiment_scores.append(None)
                engagement_totals.append(0)

        return {
            "symbol": symbol,
            "timestamps": timestamps,
            "post_counts": post_counts,
            "sentiment_scores": sentiment_scores,
            "engagement_totals": engagement_totals,
            "interval_minutes": interval_minutes,
            "generated_at": now.isoformat(),
        }

    def export_feed(
        self,
        symbol: str | None = None,
        platform: Platform | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Export posts as a feed for dashboard display.

        Args:
            symbol: Filter by symbol (optional)
            platform: Filter by platform (optional)
            limit: Maximum posts to return

        Returns:
            List of post dictionaries
        """
        if symbol:
            posts = self.get_posts_by_symbol(symbol, platform=platform, limit=limit)
        else:
            posts = []
            for post in reversed(list(self._posts)):
                if platform and post.platform != platform:
                    continue
                posts.append(post)
                if len(posts) >= limit:
                    break

        return [p.to_dict() for p in posts]

    def clear(self) -> None:
        """Clear all tracked data."""
        self._posts.clear()
        self._posts_by_id.clear()
        self._posts_by_symbol.clear()
        self._posts_by_platform.clear()
        self._posts_by_author.clear()
        self._hourly_volume.clear()
        self._hourly_sentiment.clear()
        self._topic_mentions.clear()
        self._topic_first_seen.clear()
        self._alerts.clear()
        self._total_posts_recorded = 0
        logger.info("SocialResearchTracker cleared")

    def to_dict(self) -> dict[str, Any]:
        """
        Export full tracker state for dashboard/WebSocket.

        Returns:
            Complete tracker state as dictionary
        """
        return {
            "summary": self.get_dashboard_summary(),
            "trending_topics": [t.to_dict() for t in self.get_trending_topics()],
            "recent_posts": self.export_feed(limit=20),
            "alerts": [a.to_dict() for a in self.get_alerts(limit=10)],
            "tracked_symbols": list(self._posts_by_symbol.keys()),
            "buffer_size": len(self._posts),
            "max_buffer_size": self._max_posts,
        }

    @property
    def total_posts(self) -> int:
        """Total posts currently tracked."""
        return len(self._posts)

    @property
    def tracked_symbols(self) -> list[str]:
        """List of symbols with tracked posts."""
        return list(self._posts_by_symbol.keys())

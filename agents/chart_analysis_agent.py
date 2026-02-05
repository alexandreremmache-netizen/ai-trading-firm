"""
Chart Analysis Agent
====================

Generates signals based on visual chart pattern analysis using Claude Vision.
Analyzes candlestick patterns, support/resistance levels, and trend structures.

Responsibility: Visual chart analysis signal generation ONLY.
Does NOT make trading decisions or send orders.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Any

import aiohttp
import numpy as np
import os

from core.agent_base import SignalAgent, AgentConfig
from core.events import (
    Event,
    MarketDataEvent,
    SignalEvent,
    SignalDirection,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger


logger = logging.getLogger(__name__)


@dataclass
class ChartAnalysisResult:
    """Result from Claude Vision chart analysis."""
    direction: SignalDirection
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    patterns_detected: list[str]
    support_levels: list[float]
    resistance_levels: list[float]
    trend: str  # "bullish", "bearish", "sideways"
    rationale: str
    analyzed_at: datetime
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.error is None


@dataclass
class CandleData:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


class ChartAnalysisAgent(SignalAgent):
    """
    Chart Analysis Agent using Claude Vision.

    Analyzes price charts using LLM vision capabilities to detect:
    - Candlestick patterns (doji, hammer, engulfing, etc.)
    - Support and resistance levels
    - Trend lines and channels
    - Chart formations (head & shoulders, triangles, etc.)

    Signal output:
    - Directional signals based on visual pattern recognition
    - Confidence based on pattern clarity and confluence
    """

    # Claude Vision analysis prompt
    CHART_ANALYSIS_PROMPT = """Analyze this candlestick chart and provide a trading signal.

Look for:
1. **Candlestick Patterns**: Doji, hammer, engulfing, morning/evening star, etc.
2. **Support/Resistance**: Key price levels where price has reacted
3. **Trend Structure**: Higher highs/lows (uptrend) or lower highs/lows (downtrend)
4. **Chart Patterns**: Triangles, head & shoulders, double tops/bottoms, flags
5. **Volume Analysis**: Volume confirmation of moves (if visible)

Respond with ONLY a JSON object in this exact format:
{
  "direction": "long" | "short" | "flat",
  "strength": <float from -1.0 to 1.0>,
  "confidence": <float from 0.0 to 1.0>,
  "patterns_detected": ["pattern1", "pattern2"],
  "support_levels": [price1, price2],
  "resistance_levels": [price1, price2],
  "trend": "bullish" | "bearish" | "sideways",
  "rationale": "<brief explanation of the analysis>"
}

Direction Guidelines:
- "long": Bullish patterns, support holding, uptrend continuation
- "short": Bearish patterns, resistance holding, downtrend continuation
- "flat": No clear signal, consolidation, conflicting patterns

Confidence Guidelines:
- 0.8-1.0: Multiple confirming patterns, clear trend, strong levels
- 0.5-0.7: Some patterns visible, moderate clarity
- 0.2-0.4: Ambiguous patterns, mixed signals
- 0.0-0.2: Unable to identify meaningful patterns"""

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Configuration
        self._analysis_interval = config.parameters.get("analysis_interval_seconds", 300)
        self._candle_count = config.parameters.get("candle_count", 50)
        self._min_confidence = config.parameters.get("min_confidence", 0.5)
        self._chart_timeframe = config.parameters.get("chart_timeframe", "15min")

        # API configuration
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = config.parameters.get("model", "claude-sonnet-4-20250514")
        self._timeout = config.parameters.get("timeout_seconds", 30)

        # State (bounded deque to prevent memory leak)
        self._price_history: dict[str, deque[CandleData]] = {}
        self._last_analysis: dict[str, datetime] = {}
        self._http_session: aiohttp.ClientSession | None = None

        # Tracked symbols
        self._tracked_symbols: set[str] = set(
            config.parameters.get("symbols", ["SPY", "QQQ", "AAPL", "MSFT"])
        )

    async def initialize(self) -> None:
        """Initialize the chart analysis agent."""
        if not self._anthropic_key:
            logger.warning(
                "ChartAnalysisAgent: No ANTHROPIC_API_KEY found. "
                "Chart analysis will return neutral signals."
            )
        else:
            logger.info(f"ChartAnalysisAgent initialized with model {self._model}")

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        self._http_session = aiohttp.ClientSession(timeout=timeout)

        logger.info(
            f"ChartAnalysisAgent ready - tracking {len(self._tracked_symbols)} symbols, "
            f"interval={self._analysis_interval}s"
        )

    async def shutdown(self) -> None:
        """Cleanup resources."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        logger.info("ChartAnalysisAgent shutdown complete")

    async def process_event(self, event: Event) -> None:
        """Process market data and generate chart analysis signals."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol

        # Only process tracked symbols
        if symbol not in self._tracked_symbols:
            return

        # Update price history
        self._update_price_history(symbol, event)

        # Check if analysis is due
        if not self._should_analyze(symbol):
            await self._publish_monitoring_signal(symbol)
            return

        # Perform chart analysis
        signal = await self._analyze_chart(symbol)

        if signal:
            await self._event_bus.publish_signal(signal)
            self._audit_logger.log_event(signal)
        else:
            await self._publish_monitoring_signal(symbol)

    def _update_price_history(self, symbol: str, event: MarketDataEvent) -> None:
        """Update candle history for symbol."""
        max_candles = self._candle_count * 2
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=max_candles)

        # Create candle from tick (simplified - in production would aggregate properly)
        price = event.last or event.mid
        if price <= 0:
            return

        candle = CandleData(
            timestamp=event.timestamp,
            open=price,
            high=max(event.high, price) if event.high > 0 else price,
            low=min(event.low, price) if event.low > 0 else price,
            close=price,
            volume=event.volume,
        )

        self._price_history[symbol].append(candle)

    def _should_analyze(self, symbol: str) -> bool:
        """Check if analysis interval has passed."""
        last = self._last_analysis.get(symbol)
        if last is None:
            return True

        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed >= self._analysis_interval

    async def _publish_monitoring_signal(self, symbol: str) -> None:
        """Publish neutral monitoring signal for barrier."""
        candle_count = len(self._price_history.get(symbol, []))
        signal = SignalEvent(
            source_agent=self.name,
            strategy_name="chart_monitoring",
            symbol=symbol,
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.2,
            rationale=f"Chart: Monitoring {symbol}, candles={candle_count}",
            data_sources=("chart_analysis", "ib_market_data"),
            target_price=None,
            stop_loss=None,
        )
        await self._event_bus.publish_signal(signal)

    async def _analyze_chart(self, symbol: str) -> SignalEvent | None:
        """Analyze chart using Claude Vision."""
        self._last_analysis[symbol] = datetime.now(timezone.utc)

        # Get candle data
        candles = self._price_history.get(symbol, [])
        if len(candles) < 10:
            return None

        # Generate chart image
        chart_image = self._generate_chart_image(symbol, candles[-self._candle_count:])
        if chart_image is None:
            logger.warning(f"ChartAnalysisAgent: Failed to generate chart for {symbol}")
            return None

        # Analyze with Claude Vision
        result = await self._call_claude_vision(symbol, chart_image)

        if not result.is_valid:
            logger.warning(f"ChartAnalysisAgent: Analysis failed for {symbol}: {result.error}")
            return None

        # Check confidence threshold
        if result.confidence < self._min_confidence:
            return SignalEvent(
                source_agent=self.name,
                strategy_name="chart_low_confidence",
                symbol=symbol,
                direction=SignalDirection.FLAT,
                strength=0.0,
                confidence=result.confidence,
                rationale=f"Chart confidence too low: {result.confidence:.2f}",
                data_sources=("chart_analysis", "claude_vision"),
                target_price=None,
                stop_loss=None,
            )

        # Calculate stop_loss and target_price based on direction
        # 2% stop_loss distance, 4% target_price distance (2:1 reward/risk ratio)
        current_price = candles[-1].close
        if result.direction == SignalDirection.LONG:
            stop_loss = current_price * 0.98  # 2% below current price
            target_price = current_price * 1.04  # 4% above current price
        elif result.direction == SignalDirection.SHORT:
            stop_loss = current_price * 1.02  # 2% above current price
            target_price = current_price * 0.96  # 4% below current price
        else:
            stop_loss = None
            target_price = None

        return SignalEvent(
            source_agent=self.name,
            strategy_name="chart_pattern_analysis",
            symbol=symbol,
            direction=result.direction,
            strength=result.strength,
            confidence=result.confidence,
            rationale=self._build_rationale(result),
            data_sources=("chart_analysis", "claude_vision", "pattern_recognition"),
            target_price=target_price,
            stop_loss=stop_loss,
        )

    def _generate_chart_image(
        self,
        symbol: str,
        candles: list[CandleData],
    ) -> bytes | None:
        """
        Generate a candlestick chart image.

        Uses matplotlib if available, otherwise creates ASCII representation.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.patches import Rectangle

            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot candlesticks
            width = 0.6
            for i, candle in enumerate(candles):
                color = 'green' if candle.close >= candle.open else 'red'

                # Body
                body_bottom = min(candle.open, candle.close)
                body_height = abs(candle.close - candle.open)
                rect = Rectangle(
                    (i - width/2, body_bottom),
                    width, body_height,
                    facecolor=color, edgecolor='black', linewidth=0.5
                )
                ax.add_patch(rect)

                # Wicks
                ax.plot([i, i], [candle.low, body_bottom], color='black', linewidth=0.5)
                ax.plot([i, i], [body_bottom + body_height, candle.high], color='black', linewidth=0.5)

            # Styling
            ax.set_xlim(-1, len(candles))
            prices = [c.high for c in candles] + [c.low for c in candles]
            price_range = max(prices) - min(prices)
            ax.set_ylim(min(prices) - price_range * 0.1, max(prices) + price_range * 0.1)

            ax.set_title(f"{symbol} - {self._chart_timeframe} Chart", fontsize=14)
            ax.set_ylabel("Price", fontsize=12)
            ax.grid(True, alpha=0.3)

            # Add price levels
            current_price = candles[-1].close
            ax.axhline(y=current_price, color='blue', linestyle='--', alpha=0.5)

            # Convert to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return buf.read()

        except ImportError:
            logger.warning("matplotlib not available, using text-based chart description")
            return self._generate_text_chart(symbol, candles)
        except Exception as e:
            logger.exception(f"Failed to generate chart: {e}")
            return None

    def _generate_text_chart(self, symbol: str, candles: list[CandleData]) -> bytes | None:
        """Generate text-based chart description as fallback."""
        try:
            # Create a simple price ladder visualization
            prices = [c.close for c in candles]
            highs = [c.high for c in candles]
            lows = [c.low for c in candles]

            min_price = min(lows)
            max_price = max(highs)
            current = prices[-1]

            # Calculate stats
            sma_10 = sum(prices[-10:]) / 10 if len(prices) >= 10 else current
            sma_20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else current

            # Build text description
            lines = [
                f"=== {symbol} Chart Analysis Data ===",
                f"Timeframe: {self._chart_timeframe}",
                f"Candles: {len(candles)}",
                f"",
                f"Price Range: {min_price:.2f} - {max_price:.2f}",
                f"Current Price: {current:.2f}",
                f"SMA(10): {sma_10:.2f}",
                f"SMA(20): {sma_20:.2f}",
                f"",
                "Recent Candles (last 10):",
            ]

            for i, c in enumerate(candles[-10:]):
                candle_type = "▲" if c.close >= c.open else "▼"
                body_size = abs(c.close - c.open)
                upper_wick = c.high - max(c.open, c.close)
                lower_wick = min(c.open, c.close) - c.low
                lines.append(
                    f"  {i+1}. {candle_type} O:{c.open:.2f} H:{c.high:.2f} "
                    f"L:{c.low:.2f} C:{c.close:.2f} (body={body_size:.2f})"
                )

            # Pattern hints
            lines.extend([
                "",
                "Pattern Indicators:",
                f"  Trend: {'Bullish' if sma_10 > sma_20 else 'Bearish' if sma_10 < sma_20 else 'Sideways'}",
                f"  Price vs SMA10: {'Above' if current > sma_10 else 'Below'}",
                f"  Price vs SMA20: {'Above' if current > sma_20 else 'Below'}",
            ])

            text = "\n".join(lines)
            return text.encode('utf-8')

        except Exception as e:
            logger.exception(f"Failed to generate text chart: {e}")
            return None

    async def _call_claude_vision(
        self,
        symbol: str,
        image_data: bytes,
    ) -> ChartAnalysisResult:
        """Call Claude Vision API to analyze the chart."""
        if not self._anthropic_key:
            return ChartAnalysisResult(
                direction=SignalDirection.FLAT,
                strength=0.0,
                confidence=0.1,
                patterns_detected=[],
                support_levels=[],
                resistance_levels=[],
                trend="sideways",
                rationale="API key not configured",
                analyzed_at=datetime.now(timezone.utc),
                error="API not available",
            )

        try:
            # Determine if image or text
            is_image = image_data[:4] == b'\x89PNG' or image_data[:2] == b'\xff\xd8'

            if is_image:
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Analyze this {symbol} chart:\n\n{self.CHART_ANALYSIS_PROMPT}",
                    },
                ]
            else:
                # Text-based chart
                chart_text = image_data.decode('utf-8')
                content = [
                    {
                        "type": "text",
                        "text": f"{chart_text}\n\n{self.CHART_ANALYSIS_PROMPT}",
                    },
                ]

            payload = {
                "model": self._model,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": content}],
            }

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self._anthropic_key,
                "anthropic-version": "2023-06-01",
            }

            async with self._http_session.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"API error {response.status}: {error_text}")

                data = await response.json()
                return self._parse_vision_response(data, symbol)

        except asyncio.TimeoutError:
            return ChartAnalysisResult(
                direction=SignalDirection.FLAT,
                strength=0.0,
                confidence=0.0,
                patterns_detected=[],
                support_levels=[],
                resistance_levels=[],
                trend="sideways",
                rationale="API timeout",
                analyzed_at=datetime.now(timezone.utc),
                error="Timeout",
            )
        except Exception as e:
            logger.exception(f"Claude Vision API error: {e}")
            return ChartAnalysisResult(
                direction=SignalDirection.FLAT,
                strength=0.0,
                confidence=0.0,
                patterns_detected=[],
                support_levels=[],
                resistance_levels=[],
                trend="sideways",
                rationale=f"API error: {e}",
                analyzed_at=datetime.now(timezone.utc),
                error=str(e),
            )

    def _parse_vision_response(
        self,
        data: dict[str, Any],
        symbol: str,
    ) -> ChartAnalysisResult:
        """Parse Claude Vision response."""
        import json

        try:
            content = data["content"][0]["text"]

            # Handle markdown code blocks
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            parsed = json.loads(content)

            # Parse direction
            direction_str = parsed.get("direction", "flat").lower()
            if direction_str == "long":
                direction = SignalDirection.LONG
            elif direction_str == "short":
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.FLAT

            return ChartAnalysisResult(
                direction=direction,
                strength=float(parsed.get("strength", 0.0)),
                confidence=float(parsed.get("confidence", 0.5)),
                patterns_detected=parsed.get("patterns_detected", []),
                support_levels=parsed.get("support_levels", []),
                resistance_levels=parsed.get("resistance_levels", []),
                trend=parsed.get("trend", "sideways"),
                rationale=parsed.get("rationale", "No rationale provided"),
                analyzed_at=datetime.now(timezone.utc),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse vision response: {e}")
            return ChartAnalysisResult(
                direction=SignalDirection.FLAT,
                strength=0.0,
                confidence=0.0,
                patterns_detected=[],
                support_levels=[],
                resistance_levels=[],
                trend="sideways",
                rationale="Failed to parse response",
                analyzed_at=datetime.now(timezone.utc),
                error=f"Parse error: {e}",
            )

    def _build_rationale(self, result: ChartAnalysisResult) -> str:
        """Build human-readable rationale."""
        patterns = ", ".join(result.patterns_detected[:3]) if result.patterns_detected else "none"
        supports = ", ".join(f"{p:.2f}" for p in result.support_levels[:2]) if result.support_levels else "N/A"
        resistances = ", ".join(f"{p:.2f}" for p in result.resistance_levels[:2]) if result.resistance_levels else "N/A"

        return (
            f"Chart Analysis: {result.trend} trend (conf={result.confidence:.2f}). "
            f"Patterns: {patterns}. Support: {supports}. Resistance: {resistances}. "
            f"{result.rationale[:100]}"
        )

"""
Forecasting Agent
=================

LLM-powered price forecasting agent.
Uses Claude/GPT to analyze market conditions and predict price movements.

Based on patterns from alphaswarm/tools/forecasting.

Features:
- Multi-timeframe forecasts (1h, 4h, 1d)
- Confidence intervals (upper/lower bounds)
- Supporting context from other signals
- Integration with barrier synchronization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from core.agent_base import SignalAgent, AgentConfig
from core.events import (
    Event,
    EventType,
    MarketDataEvent,
    SignalEvent,
    SignalDirection,
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger
    from core.llm_client import LLMClient


logger = logging.getLogger(__name__)


@dataclass
class PriceForecast:
    """Single price forecast with confidence bounds."""
    symbol: str
    timestamp: datetime  # When forecast was made
    forecast_horizon: str  # "1h", "4h", "1d"
    target_time: datetime  # When the price is forecasted for
    predicted_price: float
    lower_bound: float  # Lower confidence bound
    upper_bound: float  # Upper confidence bound
    confidence: float  # 0.0 to 1.0
    direction: SignalDirection
    rationale: str
    supporting_context: list[str] = field(default_factory=list)

    @property
    def expected_return_pct(self) -> float:
        """Expected return percentage from current price (estimated)."""
        # This should be calculated with actual current price
        return 0.0

    @property
    def range_width_pct(self) -> float:
        """Width of confidence interval as percentage."""
        if self.predicted_price <= 0:
            return 0.0
        return (self.upper_bound - self.lower_bound) / self.predicted_price * 100


@dataclass
class ForecastingState:
    """State for a symbol being forecasted."""
    symbol: str
    last_price: float = 0.0
    price_history: list[tuple[datetime, float]] = field(default_factory=list)
    recent_forecasts: list[PriceForecast] = field(default_factory=list)
    last_forecast_time: datetime | None = None
    forecast_accuracy: float = 0.5  # Rolling accuracy


class ForecastingAgent(SignalAgent):
    """
    LLM-Powered Forecasting Agent.

    Uses LLM to analyze price data and generate forecasts with confidence intervals.
    Produces SignalEvents based on forecast direction and confidence.

    Configuration:
        forecast_horizons: List of horizons to forecast ["1h", "4h", "1d"]
        min_confidence: Minimum confidence to generate signal (default: 0.6)
        forecast_interval_seconds: Minimum time between forecasts per symbol
        price_history_length: Number of price points to keep for context
    """

    # System prompt for price forecasting
    FORECAST_SYSTEM_PROMPT = """You are an expert quantitative analyst specializing in short-term price forecasting.
Analyze the provided price data and market context to forecast the future price.

Respond with ONLY a JSON object in this exact format:
{
  "predicted_price": <float>,
  "lower_bound": <float>,
  "upper_bound": <float>,
  "confidence": <float from 0.0 to 1.0>,
  "direction": "<LONG|SHORT|FLAT>",
  "rationale": "<brief technical analysis explanation>"
}

Guidelines:
- predicted_price: Your best estimate for the target time
- lower_bound/upper_bound: 80% confidence interval bounds
- confidence: How confident you are (0.3 = low, 0.5 = moderate, 0.7+ = high)
- direction: LONG if bullish, SHORT if bearish, FLAT if uncertain
- rationale: Key factors driving your forecast (trend, support/resistance, momentum)

Be conservative with confidence scores. Use >0.7 only for clear technical setups."""

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
        llm_client: LLMClient | None = None,
    ):
        super().__init__(config, event_bus, audit_logger)

        # LLM client (injected or created)
        self._llm_client = llm_client

        # Configuration
        params = config.parameters
        self._forecast_horizons = params.get("forecast_horizons", ["1h", "4h"])
        self._min_confidence = params.get("min_confidence", 0.6)
        self._forecast_interval = params.get("forecast_interval_seconds", 300)
        self._price_history_length = params.get("price_history_length", 100)
        self._symbols = params.get("symbols", [])

        # State per symbol
        self._states: dict[str, ForecastingState] = {}

        # Accuracy tracking
        self._total_forecasts = 0
        self._accurate_forecasts = 0

    async def initialize(self) -> None:
        """Initialize the forecasting agent."""
        logger.info(
            f"ForecastingAgent initializing with horizons: {self._forecast_horizons}, "
            f"symbols: {self._symbols}"
        )

        # Initialize LLM client if not injected
        if self._llm_client is None:
            from core.llm_client import LLMClient
            llm_config = self._config.parameters.get("llm", {})
            self._llm_client = LLMClient(config=llm_config)

        await self._llm_client.initialize()

        # Initialize states for configured symbols
        for symbol in self._symbols:
            self._states[symbol] = ForecastingState(symbol=symbol)

    async def process_event(self, event: Event) -> None:
        """Process market data and potentially generate forecasts."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol

        # Only process configured symbols
        if self._symbols and symbol not in self._symbols:
            return

        # Get or create state
        if symbol not in self._states:
            self._states[symbol] = ForecastingState(symbol=symbol)
        state = self._states[symbol]

        # Update price history
        price = event.last_price or event.close or 0.0
        if price > 0:
            state.last_price = price
            state.price_history.append((event.timestamp, price))
            # Trim history
            if len(state.price_history) > self._price_history_length:
                state.price_history = state.price_history[-self._price_history_length:]

        # Check if enough data for forecast
        if len(state.price_history) < 10:
            # Not enough data yet, publish flat signal
            await self._publish_flat_signal(symbol, "Insufficient price history for forecast")
            return

        # Check forecast interval
        now = datetime.now(timezone.utc)
        if state.last_forecast_time:
            elapsed = (now - state.last_forecast_time).total_seconds()
            if elapsed < self._forecast_interval:
                # Too soon for new forecast, publish flat signal
                await self._publish_flat_signal(
                    symbol,
                    f"Using previous forecast (next in {self._forecast_interval - elapsed:.0f}s)"
                )
                return

        # Generate forecast
        forecast = await self._generate_forecast(symbol, state)

        if forecast is None:
            await self._publish_flat_signal(symbol, "Forecast generation failed")
            return

        # Store forecast and update timing
        state.recent_forecasts.append(forecast)
        if len(state.recent_forecasts) > 10:
            state.recent_forecasts = state.recent_forecasts[-10:]
        state.last_forecast_time = now

        # Convert forecast to signal
        signal = self._forecast_to_signal(forecast)
        await self._event_bus.publish_signal(signal)
        self._audit_logger.log_event(signal)

    async def _generate_forecast(
        self,
        symbol: str,
        state: ForecastingState,
    ) -> PriceForecast | None:
        """Generate a price forecast using LLM."""
        if self._llm_client is None:
            return None

        # Build price context
        price_data = self._format_price_history(state)

        # Use first horizon
        horizon = self._forecast_horizons[0] if self._forecast_horizons else "1h"

        try:
            result = await self._call_llm_forecast(symbol, price_data, horizon)
            return result
        except Exception as e:
            logger.exception(f"Forecast generation error for {symbol}: {e}")
            return None

    def _format_price_history(self, state: ForecastingState) -> str:
        """Format price history for LLM context."""
        if not state.price_history:
            return "No price data available"

        lines = ["Recent price history (timestamp, price):"]
        # Take last 20 prices
        for ts, price in state.price_history[-20:]:
            lines.append(f"  {ts.isoformat()}: ${price:.2f}")

        # Add summary stats
        prices = [p for _, p in state.price_history]
        current = prices[-1]
        high = max(prices)
        low = min(prices)
        avg = sum(prices) / len(prices)

        lines.append("")
        lines.append(f"Current: ${current:.2f}")
        lines.append(f"High: ${high:.2f}")
        lines.append(f"Low: ${low:.2f}")
        lines.append(f"Average: ${avg:.2f}")

        # Calculate simple momentum
        if len(prices) >= 10:
            momentum = (prices[-1] / prices[-10] - 1) * 100
            lines.append(f"10-period momentum: {momentum:+.2f}%")

        return "\n".join(lines)

    async def _call_llm_forecast(
        self,
        symbol: str,
        price_data: str,
        horizon: str,
    ) -> PriceForecast | None:
        """Call LLM to generate forecast."""
        import json

        # Calculate target time
        horizon_deltas = {
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
        }
        delta = horizon_deltas.get(horizon, timedelta(hours=1))
        target_time = datetime.now(timezone.utc) + delta

        prompt = f"""Analyze the following price data for {symbol} and forecast the price in {horizon}.

{price_data}

Target forecast time: {target_time.isoformat()}

Provide your forecast in the specified JSON format."""

        # Use the LLM client's method to call the API
        session = await self._llm_client._get_session()

        if self._llm_client._provider == "anthropic":
            payload = {
                "model": self._llm_client._model,
                "max_tokens": 300,
                "system": self.FORECAST_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self._llm_client._anthropic_key,
                "anthropic-version": "2023-06-01",
            }
            url = "https://api.anthropic.com/v1/messages"
        else:  # OpenAI
            payload = {
                "model": self._llm_client._model,
                "max_tokens": 300,
                "messages": [
                    {"role": "system", "content": self.FORECAST_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._llm_client._openai_key}",
            }
            url = "https://api.openai.com/v1/chat/completions"

        await self._llm_client._rate_limiter.acquire()

        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"LLM API error {response.status}: {error_text}")
                return None

            data = await response.json()

            # Extract content
            if self._llm_client._provider == "anthropic":
                content = data["content"][0]["text"]
            else:
                content = data["choices"][0]["message"]["content"]

            # Parse JSON
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            parsed = json.loads(content)

            # Create forecast object
            direction_str = parsed.get("direction", "FLAT").upper()
            direction = {
                "LONG": SignalDirection.LONG,
                "SHORT": SignalDirection.SHORT,
                "FLAT": SignalDirection.FLAT,
            }.get(direction_str, SignalDirection.FLAT)

            return PriceForecast(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                forecast_horizon=horizon,
                target_time=target_time,
                predicted_price=float(parsed.get("predicted_price", 0)),
                lower_bound=float(parsed.get("lower_bound", 0)),
                upper_bound=float(parsed.get("upper_bound", 0)),
                confidence=float(parsed.get("confidence", 0.5)),
                direction=direction,
                rationale=str(parsed.get("rationale", "")),
            )

    def _forecast_to_signal(self, forecast: PriceForecast) -> SignalEvent:
        """Convert a forecast to a SignalEvent."""
        # Only generate actionable signal if confidence meets threshold
        if forecast.confidence < self._min_confidence:
            return SignalEvent(
                source_agent=self.name,
                strategy_name="llm_forecasting",
                symbol=forecast.symbol,
                direction=SignalDirection.FLAT,
                strength=forecast.confidence * 0.5,
                confidence=forecast.confidence,
                rationale=f"Forecast confidence {forecast.confidence:.0%} below threshold {self._min_confidence:.0%}. {forecast.rationale}",
                data_sources=("llm_forecast", "price_history"),
            )

        # Calculate signal strength based on confidence and range tightness
        strength = forecast.confidence
        if forecast.range_width_pct < 2.0:  # Tight range = higher conviction
            strength = min(1.0, strength * 1.2)

        return SignalEvent(
            source_agent=self.name,
            strategy_name="llm_forecasting",
            symbol=forecast.symbol,
            direction=forecast.direction,
            strength=strength,
            confidence=forecast.confidence,
            rationale=(
                f"LLM Forecast ({forecast.forecast_horizon}): "
                f"${forecast.predicted_price:.2f} "
                f"[${forecast.lower_bound:.2f}-${forecast.upper_bound:.2f}]. "
                f"{forecast.rationale}"
            ),
            data_sources=("llm_forecast", "price_history", "technical_analysis"),
            metadata={
                "forecast_horizon": forecast.forecast_horizon,
                "predicted_price": forecast.predicted_price,
                "lower_bound": forecast.lower_bound,
                "upper_bound": forecast.upper_bound,
                "target_time": forecast.target_time.isoformat(),
            },
        )

    async def _publish_flat_signal(self, symbol: str, reason: str) -> None:
        """Publish a FLAT signal to satisfy barrier synchronization."""
        signal = SignalEvent(
            source_agent=self.name,
            strategy_name="llm_forecasting",
            symbol=symbol,
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.3,
            rationale=f"Forecasting: {reason}",
            data_sources=("llm_forecast",),
        )
        await self._event_bus.publish_signal(signal)
        self._audit_logger.log_event(signal)

    def get_forecast_status(self) -> dict[str, Any]:
        """Get current forecasting status for monitoring."""
        accuracy = (
            self._accurate_forecasts / self._total_forecasts
            if self._total_forecasts > 0
            else 0.0
        )

        return {
            "total_forecasts": self._total_forecasts,
            "accurate_forecasts": self._accurate_forecasts,
            "accuracy": accuracy,
            "symbols_tracked": list(self._states.keys()),
            "states": {
                symbol: {
                    "last_price": state.last_price,
                    "history_length": len(state.price_history),
                    "recent_forecasts": len(state.recent_forecasts),
                    "last_forecast_time": (
                        state.last_forecast_time.isoformat()
                        if state.last_forecast_time
                        else None
                    ),
                }
                for symbol, state in self._states.items()
            },
        }

    async def evaluate_past_forecasts(self) -> dict[str, Any]:
        """Evaluate accuracy of past forecasts against actual prices."""
        results = {"evaluated": 0, "accurate": 0, "symbols": {}}
        now = datetime.now(timezone.utc)

        for symbol, state in self._states.items():
            symbol_results = []

            for forecast in state.recent_forecasts:
                # Check if target time has passed
                if forecast.target_time > now:
                    continue

                # Check if we have price data for that time
                actual_price = state.last_price  # Simplified - use current price

                # Determine if forecast was accurate
                # Accurate if: direction was correct OR price within bounds
                was_accurate = False

                if forecast.lower_bound <= actual_price <= forecast.upper_bound:
                    was_accurate = True
                elif forecast.direction == SignalDirection.LONG and actual_price > forecast.predicted_price * 0.995:
                    was_accurate = True
                elif forecast.direction == SignalDirection.SHORT and actual_price < forecast.predicted_price * 1.005:
                    was_accurate = True

                symbol_results.append({
                    "forecast_time": forecast.timestamp.isoformat(),
                    "target_time": forecast.target_time.isoformat(),
                    "predicted": forecast.predicted_price,
                    "actual": actual_price,
                    "accurate": was_accurate,
                })

                results["evaluated"] += 1
                if was_accurate:
                    results["accurate"] += 1
                    self._accurate_forecasts += 1
                self._total_forecasts += 1

            results["symbols"][symbol] = symbol_results

        results["accuracy"] = (
            results["accurate"] / results["evaluated"]
            if results["evaluated"] > 0
            else 0.0
        )

        return results

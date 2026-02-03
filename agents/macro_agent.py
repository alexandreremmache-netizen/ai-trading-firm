"""
Macro Strategy Agent
====================

Generates signals based on macroeconomic indicators.
Monitors yield curves, VIX, DXY, and other macro factors.

Responsibility: Macro signal generation ONLY.
Does NOT make trading decisions or send orders.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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


logger = logging.getLogger(__name__)


class MacroAgent(SignalAgent):
    """
    Macro Strategy Agent.

    Analyzes macroeconomic indicators to generate regime-based signals.

    Indicators monitored:
    - Yield curve (2s10s spread)
    - VIX (volatility index)
    - DXY (dollar index)
    - Credit spreads

    Signal output:
    - Risk-on / Risk-off regime
    - Sector rotation signals
    - Duration signals
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Configuration
        self._indicators = config.parameters.get("indicators", ["yield_curve", "vix", "dxy"])
        self._rebalance_frequency = config.parameters.get("rebalance_frequency", "daily")

        # State (minimal - agent should be mostly stateless)
        self._last_vix: float | None = None
        self._last_yield_spread: float | None = None
        self._current_regime: str = "neutral"

    async def initialize(self) -> None:
        """Initialize macro data feeds."""
        logger.info(f"MacroAgent initializing with indicators: {self._indicators}")
        # TODO: Initialize macro data subscriptions
        # - Subscribe to VIX futures
        # - Subscribe to treasury yields
        # - Subscribe to DXY

    async def process_event(self, event: Event) -> None:
        """Process market data and generate macro signals."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol
        macro_symbols = {"VIX", "TLT", "UUP", "SPY", "QQQ", "GLD", "USO", "IWM"}

        # Only process macro-relevant symbols
        if symbol not in macro_symbols:
            return

        signal = await self._analyze_macro_conditions(event)

        # Always publish a signal for macro symbols to satisfy barrier
        if signal is None:
            signal = SignalEvent(
                source_agent=self.name,
                strategy_name="macro_monitoring",
                symbol=symbol,
                direction=SignalDirection.FLAT,
                strength=0.0,
                confidence=0.3,
                rationale=f"Macro: Monitoring {symbol}, no actionable signal",
                data_sources=("ib_market_data", "macro_indicator"),
            )

        await self._event_bus.publish_signal(signal)
        self._audit_logger.log_event(signal)

    async def _analyze_macro_conditions(
        self,
        market_data: MarketDataEvent,
    ) -> SignalEvent | None:
        """
        Analyze macroeconomic conditions.

        TODO: Implement real macro models:
        1. Yield curve analysis (inversion = risk-off)
        2. VIX regime detection (>20 = elevated, >30 = crisis)
        3. Dollar strength impact on EM/commodities
        4. Credit spread analysis
        """

        # Placeholder logic - replace with actual macro models
        symbol = market_data.symbol

        # Process macro-relevant symbols
        # VIX for volatility regime, SPY/QQQ for broad market, GLD for risk-off, TLT for rates
        macro_symbols = {"VIX", "TLT", "UUP", "SPY", "QQQ", "GLD", "USO", "IWM"}

        if symbol not in macro_symbols:
            return None

        # VIX-based regime detection
        if symbol == "VIX":
            return await self._process_vix_signal(market_data)

        # SPY/QQQ momentum for regime
        if symbol in ("SPY", "QQQ"):
            return await self._process_equity_regime_signal(market_data)

        # GLD/TLT for risk-off detection
        if symbol in ("GLD", "TLT"):
            return await self._process_safe_haven_signal(market_data)

        return None

    async def _process_vix_signal(
        self,
        market_data: MarketDataEvent,
    ) -> SignalEvent | None:
        """
        Process VIX data for regime signal.

        TODO: Implement proper VIX regime model:
        - VIX < 15: Low vol, risk-on
        - VIX 15-20: Normal
        - VIX 20-30: Elevated, reduce risk
        - VIX > 30: Crisis, risk-off
        """
        vix_level = market_data.last

        if vix_level <= 0:
            return None

        self._last_vix = vix_level

        # Determine regime
        if vix_level < 15:
            direction = SignalDirection.LONG
            regime = "risk_on"
            strength = 0.7
        elif vix_level < 20:
            direction = SignalDirection.FLAT
            regime = "neutral"
            strength = 0.0
        elif vix_level < 30:
            direction = SignalDirection.SHORT
            regime = "risk_off"
            strength = -0.5
        else:
            direction = SignalDirection.SHORT
            regime = "crisis"
            strength = -1.0

        # Only signal on regime change
        if regime == self._current_regime:
            return None

        self._current_regime = regime

        return SignalEvent(
            source_agent=self.name,
            strategy_name="macro_vix_regime",
            symbol="SPY",  # Signal applies to broad market
            direction=direction,
            strength=strength,
            confidence=0.6,  # TODO: Calculate based on model confidence
            rationale=f"VIX regime change to {regime} (VIX={vix_level:.1f})",
            data_sources=("ib_market_data", "macro_indicator", "vix_indicator"),
        )

    async def _process_equity_regime_signal(
        self,
        market_data: MarketDataEvent,
    ) -> SignalEvent | None:
        """
        Process SPY/QQQ data for equity regime detection.

        Uses price momentum as a simple regime indicator.
        """
        symbol = market_data.symbol
        price = market_data.last or market_data.mid

        if price <= 0:
            return None

        # Store price history (simple approach using instance variable)
        if not hasattr(self, "_price_history"):
            self._price_history: dict[str, list[float]] = {}

        if symbol not in self._price_history:
            self._price_history[symbol] = []

        self._price_history[symbol].append(price)

        # Keep last 50 prices
        if len(self._price_history[symbol]) > 50:
            self._price_history[symbol] = self._price_history[symbol][-50:]

        # Need at least 5 prices for signal (fast startup)
        if len(self._price_history[symbol]) < 5:
            return None

        prices = self._price_history[symbol]
        n = len(prices)

        # Adaptive SMA based on available data
        sma_short = sum(prices[-min(5, n):]) / min(5, n)
        sma_long = sum(prices[-min(10, n):]) / min(10, n)

        # Simple momentum regime
        momentum = (sma_short - sma_long) / sma_long * 100 if sma_long > 0 else 0  # Percentage

        if momentum > 0.1:
            direction = SignalDirection.LONG
            regime = "bullish"
            strength = min(1.0, momentum / 2)
        elif momentum < -0.1:
            direction = SignalDirection.SHORT
            regime = "bearish"
            strength = max(-1.0, momentum / 2)
        else:
            # Neutral signal with low strength
            direction = SignalDirection.FLAT
            regime = "neutral"
            strength = 0.0

        return SignalEvent(
            source_agent=self.name,
            strategy_name="macro_equity_regime",
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=0.55,
            rationale=f"Equity regime {regime}: {symbol} SMA10/SMA20 momentum={momentum:.2f}%",
            data_sources=("ib_market_data", "macro_indicator", "momentum_indicator"),
        )

    async def _process_safe_haven_signal(
        self,
        market_data: MarketDataEvent,
    ) -> SignalEvent | None:
        """
        Process GLD/TLT for safe haven demand (risk-off indicator).

        Rising GLD/TLT suggests risk-off environment.
        """
        symbol = market_data.symbol
        price = market_data.last or market_data.mid

        if price <= 0:
            return None

        # Store price history
        if not hasattr(self, "_safe_haven_history"):
            self._safe_haven_history: dict[str, list[float]] = {}

        if symbol not in self._safe_haven_history:
            self._safe_haven_history[symbol] = []

        self._safe_haven_history[symbol].append(price)

        # Keep last 30 prices
        if len(self._safe_haven_history[symbol]) > 30:
            self._safe_haven_history[symbol] = self._safe_haven_history[symbol][-30:]

        # Need at least 4 prices (fast startup)
        if len(self._safe_haven_history[symbol]) < 4:
            return None

        prices = self._safe_haven_history[symbol]
        n = len(prices)
        half = max(2, n // 2)
        recent_avg = sum(prices[-half:]) / half
        older_avg = sum(prices[:-half]) / (n - half) if n > half else recent_avg

        # Safe haven demand rising = risk-off
        change_pct = (recent_avg - older_avg) / older_avg * 100

        if change_pct > 0.3:
            # Risk-off: safe havens rising
            return SignalEvent(
                source_agent=self.name,
                strategy_name="macro_safe_haven",
                symbol="SPY",  # Signal affects broad market
                direction=SignalDirection.SHORT,
                strength=-0.3,
                confidence=0.5,
                rationale=f"Risk-off: {symbol} rising {change_pct:.2f}% (safe haven demand)",
                data_sources=("ib_market_data", "macro_indicator", "safe_haven_indicator"),
            )

        return None

    async def _analyze_yield_curve(self) -> SignalEvent | None:
        """
        Analyze yield curve for recession signals.

        TODO: Implement yield curve analysis:
        - 2s10s spread
        - 3m10y spread (Fed preferred)
        - Full curve shape analysis
        """
        # Placeholder
        return None

    async def _analyze_dollar_strength(self) -> SignalEvent | None:
        """
        Analyze dollar strength for sector rotation.

        TODO: Implement DXY analysis:
        - Strong dollar = headwind for multinationals, EM
        - Weak dollar = tailwind for commodities, EM
        """
        # Placeholder
        return None

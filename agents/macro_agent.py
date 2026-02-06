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
from collections import deque
from typing import TYPE_CHECKING, Any

from core.agent_base import SignalAgent, AgentConfig
from core.events import (
    Event,
    EventType,
    MarketDataEvent,
    SignalEvent,
    SignalDirection,
)

# Import HMM regime detection
try:
    from core.hmm_regime import (
        HMMRegimeDetector,
        MarketState,
        create_hmm_detector,
    )
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    HMMRegimeDetector = None  # type: ignore
    MarketState = None  # type: ignore

# Import yield curve analyzer
try:
    from core.yield_curve import (
        YieldCurveAnalyzer,
        YieldCurveState,
    )
    HAS_YIELD_CURVE = True
except ImportError:
    HAS_YIELD_CURVE = False
    YieldCurveAnalyzer = None  # type: ignore
    YieldCurveState = None  # type: ignore

# Import DXY analyzer
try:
    from core.dxy_analyzer import (
        DXYAnalyzer,
        DXYState,
    )
    HAS_DXY = True
except ImportError:
    HAS_DXY = False
    DXYAnalyzer = None  # type: ignore
    DXYState = None  # type: ignore

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
        self._indicators = config.parameters.get("indicators", ["yield_curve", "vix", "dxy", "hmm"])
        self._rebalance_frequency = config.parameters.get("rebalance_frequency", "daily")
        self._hmm_min_samples = config.parameters.get("hmm_min_samples", 50)
        self._hmm_n_states = config.parameters.get("hmm_n_states", 3)

        # Minimum confidence threshold for signal generation
        self._min_confidence = config.parameters.get("min_confidence", 0.75)

        # State (minimal - agent should be mostly stateless)
        self._last_vix: float | None = None
        self._last_yield_spread: float | None = None
        self._current_regime: str = "neutral"

        # HMM regime detector (bounded to prevent memory leak)
        self._hmm_detector: HMMRegimeDetector | None = None
        self._hmm_returns: deque[float] = deque(maxlen=500)
        self._last_hmm_state: MarketState | None = None
        if HAS_HMM:
            self._hmm_detector = create_hmm_detector(
                n_states=self._hmm_n_states,
                min_samples=self._hmm_min_samples,
            )

        # Yield curve analyzer
        self._yield_curve_analyzer: YieldCurveAnalyzer | None = None
        if HAS_YIELD_CURVE:
            self._yield_curve_analyzer = YieldCurveAnalyzer()

        # DXY analyzer
        self._dxy_analyzer: DXYAnalyzer | None = None
        if HAS_DXY:
            self._dxy_analyzer = DXYAnalyzer()

        # Position-level risk parameters
        self._position_sl_pct = config.parameters.get("position_stop_loss_pct", 2.0) / 100  # 2% default
        self._position_tp_pct = config.parameters.get("position_take_profit_pct", 4.0) / 100  # 4% default
        self._min_rr_ratio = config.parameters.get("min_reward_risk_ratio", 2.0)  # 2:1 R:R minimum

    def _get_reference_price(self, symbol: str, default: float = 500.0) -> float:
        """
        Get reference price for a symbol from stored price history.

        Args:
            symbol: Symbol to get price for
            default: Default price if no history available

        Returns:
            Latest price from history or default
        """
        if hasattr(self, "_price_history") and symbol in self._price_history:
            prices = self._price_history[symbol]
            if prices:
                return prices[-1]
        return default

    def _calculate_position_stops(
        self,
        price: float,
        direction: SignalDirection,
        sl_pct: float | None = None,
        tp_pct: float | None = None,
    ) -> tuple[float | None, float | None, float, bool]:
        """
        Calculate position-level stop-loss and take-profit.

        Args:
            price: Entry/reference price
            direction: Signal direction (LONG/SHORT/FLAT)
            sl_pct: Stop-loss percentage (optional, uses default if None)
            tp_pct: Take-profit percentage (optional, uses default if None)

        Returns:
            Tuple of (stop_loss, target_price, rr_ratio, is_valid_rr)
        """
        if price <= 0 or direction == SignalDirection.FLAT:
            return None, None, 0.0, False

        sl = sl_pct if sl_pct is not None else self._position_sl_pct
        tp = tp_pct if tp_pct is not None else self._position_tp_pct

        if direction == SignalDirection.LONG:
            stop_loss = round(price * (1 - sl), 2)
            target_price = round(price * (1 + tp), 2)
            sl_distance = price - stop_loss
            tp_distance = target_price - price
        else:  # SHORT
            stop_loss = round(price * (1 + sl), 2)
            target_price = round(price * (1 - tp), 2)
            sl_distance = stop_loss - price
            tp_distance = price - target_price

        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0.0
        # Round to 2 decimals to avoid floating-point precision issues
        # (e.g., 1.9999999... should be treated as 2.0)
        rr_ratio = round(rr_ratio, 2)
        is_valid_rr = rr_ratio >= self._min_rr_ratio

        return stop_loss, target_price, rr_ratio, is_valid_rr

    async def _emit_warmup_heartbeat(self, symbol: str, reason: str = "Not a macro symbol") -> None:
        """Emit FLAT heartbeat signal to participate in barrier sync."""
        signal = SignalEvent(
            source_agent=self.name,
            strategy_name="macro_heartbeat",
            symbol=symbol,
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.1,
            rationale=f"Macro heartbeat: {reason}",
        )
        if self._event_bus:
            await self._event_bus.publish(signal)

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

        # Only process macro-relevant symbols; emit heartbeat for others to satisfy barrier
        if symbol not in macro_symbols:
            await self._emit_warmup_heartbeat(symbol, f"{symbol} not in macro universe")
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

        # Filter weak signals by confidence threshold
        if signal.direction != SignalDirection.FLAT and signal.confidence < self._min_confidence:
            logger.debug(
                f"Signal filtered: confidence {signal.confidence:.2f} < threshold {self._min_confidence}"
            )
            # Still publish a neutral signal for barrier satisfaction
            signal = SignalEvent(
                source_agent=self.name,
                strategy_name="macro_low_confidence",
                symbol=signal.symbol,
                direction=SignalDirection.FLAT,
                strength=0.0,
                confidence=signal.confidence,
                rationale=f"Macro signal confidence too low: {signal.confidence:.2f} < {self._min_confidence}",
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

        Implemented macro models:
        1. VIX regime detection (>20 = elevated, >30 = crisis)
        2. HMM-based regime detection (bull/bear/sideways)
        3. Yield curve analysis (via YieldCurveAnalyzer)
        4. Dollar strength analysis (via DXYAnalyzer)
        """
        symbol = market_data.symbol

        # Process macro-relevant symbols
        # VIX for volatility regime, SPY/QQQ for broad market, GLD for risk-off, TLT for rates
        # UUP for DXY tracking
        macro_symbols = {"VIX", "TLT", "UUP", "SPY", "QQQ", "GLD", "USO", "IWM", "DX"}

        if symbol not in macro_symbols:
            return None

        price = market_data.last or market_data.mid
        if price and price > 0:
            # Update HMM with SPY/QQQ data (broad market)
            if symbol in ("SPY", "QQQ"):
                self.update_hmm_regime(price)

            # Update DXY analyzer if we have dollar data
            if symbol in ("UUP", "DX") and HAS_DXY and self._dxy_analyzer:
                # UUP is dollar ETF, approximate DXY from it
                # UUP ~= 28 when DXY = 100, so multiply by ~3.5
                approx_dxy = price * 3.5 if symbol == "UUP" else price
                self._dxy_analyzer.update(approx_dxy)

        # VIX-based regime detection
        if symbol == "VIX":
            return await self._process_vix_signal(market_data)

        # SPY/QQQ momentum for regime + HMM signal
        if symbol in ("SPY", "QQQ"):
            # Try HMM signal first (more sophisticated)
            hmm_signal = self.get_hmm_regime_signal()
            if hmm_signal is not None:
                return hmm_signal

            # Fall back to simple momentum
            return await self._process_equity_regime_signal(market_data)

        # GLD/TLT for risk-off detection
        if symbol in ("GLD", "TLT"):
            return await self._process_safe_haven_signal(market_data)

        # UUP/DX for dollar strength
        if symbol in ("UUP", "DX"):
            return await self._analyze_dollar_strength()

        return None

    async def _process_vix_signal(
        self,
        market_data: MarketDataEvent,
    ) -> SignalEvent | None:
        """
        Process VIX data for regime signal.

        VIX-based regime detection:
        - VIX < 15: Low vol, risk-on
        - VIX 15-20: Normal
        - VIX 20-30: Elevated, reduce risk
        - VIX > 30: Crisis, risk-off

        Position-level risk controls:
        - 2% stop-loss per position (configurable)
        - 4% take-profit per position (configurable)
        - 2:1 minimum R:R ratio (configurable)
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

        # Get SPY price from history if available
        spy_price = self._get_reference_price("SPY", default=500.0)

        # Calculate position-level stop-loss and take-profit
        stop_loss, target_price, rr_ratio, is_valid_rr = self._calculate_position_stops(
            price=spy_price,
            direction=direction,
        )

        # RISK FILTER: Reject poor R:R trades
        if direction != SignalDirection.FLAT and not is_valid_rr:
            logger.info(
                f"MacroAgent VIX: Rejecting signal - R:R ratio {rr_ratio:.2f} "
                f"< minimum {self._min_rr_ratio:.2f}"
            )
            return None

        return SignalEvent(
            source_agent=self.name,
            strategy_name="macro_vix_regime",
            symbol="MES",  # FIX-30: System trades MES, not SPY
            direction=direction,
            strength=strength,
            confidence=0.6,
            rationale=f"VIX regime change to {regime} (VIX={vix_level:.1f}) | SL={stop_loss}, TP={target_price}, R:R={rr_ratio:.2f}",
            data_sources=("ib_market_data", "macro_indicator", "vix_indicator"),
            target_price=target_price,
            stop_loss=stop_loss,
        )

    async def _process_equity_regime_signal(
        self,
        market_data: MarketDataEvent,
    ) -> SignalEvent | None:
        """
        Process SPY/QQQ data for equity regime detection.

        Uses price momentum as a simple regime indicator.

        Position-level risk controls:
        - 2% stop-loss per position (configurable)
        - 4% take-profit per position (configurable)
        - 2:1 minimum R:R ratio (configurable)
        """
        symbol = market_data.symbol
        price = market_data.last or market_data.mid

        if price <= 0:
            return None

        # Store price history (bounded deque to prevent memory leak)
        if not hasattr(self, "_price_history"):
            self._price_history: dict[str, deque[float]] = {}

        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=50)

        self._price_history[symbol].append(price)

        # Need at least 5 prices for signal (fast startup)
        if len(self._price_history[symbol]) < 5:
            return None

        prices = list(self._price_history[symbol])  # Convert deque to list for slicing
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

        # Calculate position-level stop-loss and take-profit
        current_price = prices[-1]
        stop_loss, target_price, rr_ratio, is_valid_rr = self._calculate_position_stops(
            price=current_price,
            direction=direction,
        )

        # RISK FILTER: Reject poor R:R trades
        if direction != SignalDirection.FLAT and not is_valid_rr:
            logger.info(
                f"MacroAgent Equity: Rejecting {symbol} signal - R:R ratio {rr_ratio:.2f} "
                f"< minimum {self._min_rr_ratio:.2f}"
            )
            return None

        return SignalEvent(
            source_agent=self.name,
            strategy_name="macro_equity_regime",
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=0.55,
            rationale=f"Equity regime {regime}: {symbol} SMA5/SMA10 momentum={momentum:.2f}% | SL={stop_loss}, TP={target_price}, R:R={rr_ratio:.2f}",
            data_sources=("ib_market_data", "macro_indicator", "momentum_indicator"),
            target_price=target_price,
            stop_loss=stop_loss,
        )

    async def _process_safe_haven_signal(
        self,
        market_data: MarketDataEvent,
    ) -> SignalEvent | None:
        """
        Process GLD/TLT for safe haven demand (risk-off indicator).

        Rising GLD/TLT suggests risk-off environment.

        Position-level risk controls:
        - 2% stop-loss per position (configurable)
        - 4% take-profit per position (configurable)
        - 2:1 minimum R:R ratio (configurable)
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
            # Risk-off: safe havens rising - SHORT signal with SL/TP
            spy_price = self._get_reference_price("SPY", default=500.0)

            # Calculate position-level stop-loss and take-profit
            stop_loss, target_price, rr_ratio, is_valid_rr = self._calculate_position_stops(
                price=spy_price,
                direction=SignalDirection.SHORT,
            )

            # RISK FILTER: Reject poor R:R trades
            if not is_valid_rr:
                logger.info(
                    f"MacroAgent SafeHaven: Rejecting signal - R:R ratio {rr_ratio:.2f} "
                    f"< minimum {self._min_rr_ratio:.2f}"
                )
                return None

            return SignalEvent(
                source_agent=self.name,
                strategy_name="macro_safe_haven",
                symbol="MES",  # FIX-30: System trades MES, not SPY
                direction=SignalDirection.SHORT,
                strength=-0.3,
                confidence=0.5,
                rationale=f"Risk-off: {symbol} rising {change_pct:.2f}% (safe haven demand) | SL={stop_loss}, TP={target_price}, R:R={rr_ratio:.2f}",
                data_sources=("ib_market_data", "macro_indicator", "safe_haven_indicator"),
                target_price=target_price,
                stop_loss=stop_loss,
            )

        return None

    async def _analyze_yield_curve(self) -> SignalEvent | None:
        """
        Analyze yield curve for recession signals.

        Uses YieldCurveAnalyzer for:
        - 2s10s spread
        - 3m10y spread (Fed preferred)
        - Full curve shape analysis

        Position-level risk controls:
        - 2% stop-loss per position (configurable)
        - 4% take-profit per position (configurable)
        - 2:1 minimum R:R ratio (configurable)
        """
        if not HAS_YIELD_CURVE or self._yield_curve_analyzer is None:
            return None

        try:
            result = self._yield_curve_analyzer.analyze()
            signal, rationale = self._yield_curve_analyzer.get_trading_signal()

            if abs(signal) < 0.15:
                return None  # Not strong enough signal

            direction = SignalDirection.LONG if signal > 0 else SignalDirection.SHORT
            strength = signal

            # Get SPY price from history
            spy_price = self._get_reference_price("SPY", default=500.0)

            # Calculate position-level stop-loss and take-profit
            stop_loss, target_price, rr_ratio, is_valid_rr = self._calculate_position_stops(
                price=spy_price,
                direction=direction,
            )

            # RISK FILTER: Reject poor R:R trades
            if not is_valid_rr:
                logger.info(
                    f"MacroAgent YieldCurve: Rejecting signal - R:R ratio {rr_ratio:.2f} "
                    f"< minimum {self._min_rr_ratio:.2f}"
                )
                return None

            return SignalEvent(
                source_agent=self.name,
                strategy_name="macro_yield_curve",
                symbol="MES",  # FIX-30: System trades MES, not SPY
                direction=direction,
                strength=strength,
                confidence=0.6 if result.is_warning else 0.5,
                rationale=f"{rationale} | SL={stop_loss}, TP={target_price}, R:R={rr_ratio:.2f}",
                data_sources=("ib_market_data", "yield_curve_indicator"),
                target_price=target_price,
                stop_loss=stop_loss,
            )
        except Exception as e:
            logger.exception(f"Error in yield curve analysis: {e}")
            return None

    async def _analyze_dollar_strength(self) -> SignalEvent | None:
        """
        Analyze dollar strength for sector rotation.

        Uses DXYAnalyzer for:
        - Strong dollar = headwind for multinationals, EM
        - Weak dollar = tailwind for commodities, EM

        Position-level risk controls:
        - 2% stop-loss per position (configurable)
        - 4% take-profit per position (configurable)
        - 2:1 minimum R:R ratio (configurable)
        """
        if not HAS_DXY or self._dxy_analyzer is None:
            return None

        try:
            result = self._dxy_analyzer.analyze()
            signal = result.signal_for_risk_assets

            if abs(signal) < 0.2:
                return None  # Not strong enough

            direction = SignalDirection.LONG if signal > 0 else SignalDirection.SHORT
            strength = signal

            # Get SPY price from history
            spy_price = self._get_reference_price("SPY", default=500.0)

            # Calculate position-level stop-loss and take-profit
            stop_loss, target_price, rr_ratio, is_valid_rr = self._calculate_position_stops(
                price=spy_price,
                direction=direction,
            )

            # RISK FILTER: Reject poor R:R trades
            if not is_valid_rr:
                logger.info(
                    f"MacroAgent DXY: Rejecting signal - R:R ratio {rr_ratio:.2f} "
                    f"< minimum {self._min_rr_ratio:.2f}"
                )
                return None

            rationale = (
                f"DXY {result.state.value} ({result.current_level:.1f}), "
                f"momentum {result.momentum_score:+.2f} | SL={stop_loss}, TP={target_price}, R:R={rr_ratio:.2f}"
            )

            return SignalEvent(
                source_agent=self.name,
                strategy_name="macro_dxy",
                symbol="MES",  # FIX-30: System trades MES, not SPY
                direction=direction,
                strength=strength,
                confidence=0.55,
                rationale=rationale,
                data_sources=("ib_market_data", "dxy_indicator"),
                target_price=target_price,
                stop_loss=stop_loss,
            )
        except Exception as e:
            logger.exception(f"Error in DXY analysis: {e}")
            return None

    # =========================================================================
    # HMM REGIME DETECTION METHODS
    # =========================================================================

    def update_hmm_regime(self, price: float) -> None:
        """
        Update HMM regime detector with new price observation.

        Called when new price data arrives. Calculates returns and
        updates the HMM model.

        Args:
            price: Latest price observation
        """
        if not HAS_HMM or self._hmm_detector is None:
            return

        # Store price for return calculation (bounded deque)
        if not hasattr(self, "_hmm_prices"):
            self._hmm_prices: deque[float] = deque(maxlen=300)

        self._hmm_prices.append(price)

        # Calculate return if we have at least 2 prices
        if len(self._hmm_prices) >= 2:
            prev_price = self._hmm_prices[-2]
            if prev_price > 0:
                ret = (price - prev_price) / prev_price
                self._hmm_returns.append(ret)

                # Update HMM detector
                self._hmm_detector.update(ret)

        # FIX-29: Run HMM fit in thread to avoid blocking event loop (500-2000ms)
        if (
            len(self._hmm_returns) >= self._hmm_min_samples
            and not self._hmm_detector._is_fitted
        ):
            try:
                import asyncio
                import concurrent.futures
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    loop.run_in_executor(pool, self._hmm_detector.fit, list(self._hmm_returns))
                logger.info("HMM regime detector fit submitted to thread pool")
            except Exception as e:
                logger.warning(f"Could not fit HMM: {e}")

    def get_hmm_regime_signal(self) -> SignalEvent | None:
        """
        Get trading signal based on HMM regime detection.

        Position-level risk controls:
        - 2% stop-loss per position (configurable)
        - 4% take-profit per position (configurable)
        - 2:1 minimum R:R ratio (configurable)

        Returns:
            SignalEvent if regime signal is actionable, None otherwise
        """
        if not HAS_HMM or self._hmm_detector is None:
            return None

        if not self._hmm_detector._is_fitted:
            return None

        try:
            # Get regime analysis
            result = self._hmm_detector.analyze()
            signal_strength, rationale = self._hmm_detector.get_regime_signal()

            # Check for regime change
            current_state = result.current_state
            if current_state == self._last_hmm_state:
                return None  # No regime change

            self._last_hmm_state = current_state

            # Only signal on meaningful strength
            if abs(signal_strength) < 0.2:
                return None

            # Determine direction
            if signal_strength > 0:
                direction = SignalDirection.LONG
            elif signal_strength < 0:
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.FLAT

            # Get SPY price from history
            spy_price = self._get_reference_price("SPY", default=500.0)

            # Calculate position-level stop-loss and take-profit
            stop_loss, target_price, rr_ratio, is_valid_rr = self._calculate_position_stops(
                price=spy_price,
                direction=direction,
            )

            # RISK FILTER: Reject poor R:R trades
            if direction != SignalDirection.FLAT and not is_valid_rr:
                logger.info(
                    f"MacroAgent HMM: Rejecting signal - R:R ratio {rr_ratio:.2f} "
                    f"< minimum {self._min_rr_ratio:.2f}"
                )
                return None

            return SignalEvent(
                source_agent=self.name,
                strategy_name="macro_hmm_regime",
                symbol="MES",  # FIX-30: System trades MES, not SPY
                direction=direction,
                strength=signal_strength,
                confidence=result.confidence,
                rationale=f"HMM: {rationale} [state: {current_state.value}] | SL={stop_loss}, TP={target_price}, R:R={rr_ratio:.2f}",
                data_sources=("ib_market_data", "hmm_regime_detector"),
                target_price=target_price,
                stop_loss=stop_loss,
            )
        except Exception as e:
            logger.exception(f"Error in HMM regime signal: {e}")
            return None

    def get_hmm_regime_probabilities(self) -> dict[str, float] | None:
        """
        Get current HMM regime probabilities.

        Returns:
            Dictionary of regime -> probability, or None if not fitted
        """
        if not HAS_HMM or self._hmm_detector is None:
            return None

        if not self._hmm_detector._is_fitted:
            return None

        try:
            probs = self._hmm_detector.get_regime_probabilities()
            return {k.value: v for k, v in probs.items()}
        except Exception as e:
            logger.warning(f"Could not get HMM probabilities: {e}")
            return None

    def get_hmm_transition_matrix(self) -> dict[str, dict[str, float]] | None:
        """
        Get HMM state transition probability matrix.

        Returns:
            Nested dict of transition probabilities, or None if not fitted
        """
        if not HAS_HMM or self._hmm_detector is None:
            return None

        if not self._hmm_detector._is_fitted:
            return None

        try:
            return self._hmm_detector.get_transition_matrix()
        except Exception as e:
            logger.warning(f"Could not get transition matrix: {e}")
            return None

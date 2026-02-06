"""
Statistical Arbitrage Agent
===========================

Generates signals based on statistical relationships between instruments.
Implements pairs trading and mean reversion strategies.

Responsibility: Statistical arbitrage signal generation ONLY.
Does NOT make trading decisions or send orders.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

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


@dataclass
class PairState:
    """State for a trading pair."""
    symbol_a: str
    symbol_b: str
    price_a: deque  # Rolling window of prices
    price_b: deque
    spread_history: deque
    zscore: float = 0.0
    hedge_ratio: float = 1.0
    half_life: float = 0.0
    last_signal: SignalDirection = SignalDirection.FLAT
    # P2: Cointegration monitoring
    cointegration_score: float = 0.0  # ADF test statistic
    cointegration_pvalue: float = 1.0  # ADF p-value
    is_cointegrated: bool = False
    cointegration_last_check: datetime | None = None
    cointegration_breakdown_count: int = 0  # Times cointegration broke down
    # P2: Pair selection optimization
    pair_quality_score: float = 0.0  # Overall pair quality (0-1)
    spread_volatility: float = 0.0
    mean_reversion_strength: float = 0.0


@dataclass
class ZScoreAlert:
    """Alert for z-score threshold breach (P2)."""
    pair_key: str
    timestamp: datetime
    zscore: float
    alert_level: str  # "warning", "critical", "extreme"
    direction: str  # "high" or "low"
    message: str


@dataclass
class DivergenceState:
    """State for correlation divergence tracking."""
    symbol_a: str
    symbol_b: str
    returns_a: deque  # Rolling returns
    returns_b: deque
    correlation_history: deque  # Rolling correlation values
    current_correlation: float = 0.0
    baseline_correlation: float = 0.0
    divergence_zscore: float = 0.0
    last_signal: SignalDirection = SignalDirection.FLAT
    divergence_start: datetime | None = None


class StatArbAgent(SignalAgent):
    """
    Statistical Arbitrage Agent.

    Implements pairs trading strategy based on cointegration.

    Methodology:
    1. Identify cointegrated pairs
    2. Calculate hedge ratio via OLS or Kalman filter
    3. Compute spread z-score
    4. Generate mean reversion signals

    Signal output:
    - Long/short pair signals when z-score exceeds threshold
    - Exit signals when z-score reverts
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Configuration
        self._lookback_days = config.parameters.get("lookback_days", 60)
        self._zscore_entry = config.parameters.get("zscore_entry_threshold", 2.0)
        self._zscore_exit = config.parameters.get("zscore_exit_threshold", 0.5)
        self._pairs_config = config.parameters.get("pairs", [])

        # State for each pair
        self._pairs: dict[str, PairState] = {}
        self._lookback_size = self._lookback_days * 390  # Minute bars

        # P2: Z-score alert thresholds
        self._zscore_warning = config.parameters.get("zscore_warning_threshold", 2.5)
        self._zscore_critical = config.parameters.get("zscore_critical_threshold", 3.0)
        self._zscore_extreme = config.parameters.get("zscore_extreme_threshold", 4.0)
        self._zscore_alerts: list[ZScoreAlert] = []
        self._max_alerts = config.parameters.get("max_zscore_alerts", 100)

        # P2: Cointegration monitoring
        self._cointegration_check_interval = config.parameters.get("cointegration_check_interval", 100)  # bars
        self._cointegration_pvalue_threshold = config.parameters.get("cointegration_pvalue_threshold", 0.05)
        self._cointegration_breakdown_threshold = config.parameters.get("cointegration_breakdown_threshold", 3)
        self._bar_counter = 0

        # P2: Pair selection optimization
        self._min_pair_quality = config.parameters.get("min_pair_quality_score", 0.4)
        self._pair_ranking_enabled = config.parameters.get("pair_ranking_enabled", True)

        # Minimum confidence threshold for signal generation
        self._min_confidence = config.parameters.get("min_confidence", 0.75)

        # Correlation divergence detection (MoonDev-inspired)
        self._divergence_enabled = config.parameters.get("divergence_enabled", True)
        # FIX-11: Removed stock/ETF pairs - futures-only system
        self._divergence_pairs = config.parameters.get("divergence_pairs", [
            ["MES", "MNQ"],   # Micro index futures
            ["MGC", "MCL"],   # Micro metals/energy
            ["MYM", "M2K"],   # Micro Dow / Micro Russell
        ])
        self._divergence_lookback = config.parameters.get("divergence_lookback", 20)
        self._divergence_threshold = config.parameters.get("divergence_threshold", 2.0)
        self._correlation_breakdown_threshold = config.parameters.get("correlation_breakdown_threshold", 0.3)
        self._divergence_states: dict[str, DivergenceState] = {}

    async def initialize(self) -> None:
        """Initialize pairs state."""
        logger.info(f"StatArbAgent initializing with pairs: {self._pairs_config}")

        for pair in self._pairs_config:
            if len(pair) == 2:
                pair_key = f"{pair[0]}:{pair[1]}"
                self._pairs[pair_key] = PairState(
                    symbol_a=pair[0],
                    symbol_b=pair[1],
                    price_a=deque(maxlen=self._lookback_size),
                    price_b=deque(maxlen=self._lookback_size),
                    spread_history=deque(maxlen=self._lookback_size),
                )

        # Initialize divergence tracking (MoonDev-inspired correlation divergence)
        if self._divergence_enabled:
            logger.info(f"StatArbAgent initializing divergence tracking for: {self._divergence_pairs}")
            for pair in self._divergence_pairs:
                if len(pair) == 2:
                    div_key = f"DIV:{pair[0]}:{pair[1]}"
                    self._divergence_states[div_key] = DivergenceState(
                        symbol_a=pair[0],
                        symbol_b=pair[1],
                        returns_a=deque(maxlen=self._divergence_lookback * 2),
                        returns_b=deque(maxlen=self._divergence_lookback * 2),
                        correlation_history=deque(maxlen=50),
                    )

        # TODO: Load historical data to bootstrap cointegration estimates

    async def process_event(self, event: Event) -> None:
        """Process market data and generate stat arb signals."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol
        price = event.mid

        if price <= 0:
            return

        # FIX-28: Track last update time per leg to avoid stale data
        import time
        now = time.time()

        # Update relevant pairs
        signals = []
        for pair_key, pair_state in self._pairs.items():
            if symbol == pair_state.symbol_a:
                pair_state.price_a.append(price)
                if not hasattr(pair_state, '_last_update_a'):
                    pair_state._last_update_a = 0.0
                    pair_state._last_update_b = 0.0
                pair_state._last_update_a = now
                # FIX-28: Only compute signal when both legs updated within 5s
                if now - getattr(pair_state, '_last_update_b', 0.0) < 5.0:
                    signal = await self._check_pair_signal(pair_key, pair_state)
                    if signal:
                        signals.extend(signal if isinstance(signal, list) else [signal])

            elif symbol == pair_state.symbol_b:
                pair_state.price_b.append(price)
                if not hasattr(pair_state, '_last_update_b'):
                    pair_state._last_update_a = 0.0
                    pair_state._last_update_b = 0.0
                pair_state._last_update_b = now
                # FIX-28: Only compute signal when both legs updated within 5s
                if now - getattr(pair_state, '_last_update_a', 0.0) < 5.0:
                    signal = await self._check_pair_signal(pair_key, pair_state)
                    if signal:
                        signals.extend(signal if isinstance(signal, list) else [signal])

        # Publish signals (filter by confidence threshold)
        for signal in signals:
            if signal.confidence < self._min_confidence:
                logger.debug(
                    f"Signal filtered: confidence {signal.confidence:.2f} < threshold {self._min_confidence}"
                )
                continue
            await self._event_bus.publish_signal(signal)
            self._audit_logger.log_event(signal)

        # If no signals generated for this event but symbol is tracked, send a neutral signal
        # This ensures the barrier gets a response from StatArbAgent
        if not signals:
            # Send a neutral signal if symbol is part of any tracked pair
            for pair_key, pair_state in self._pairs.items():
                if symbol == pair_state.symbol_a or symbol == pair_state.symbol_b:
                    # FIX-27: Use actual symbol (not pair key) for heartbeat
                    data_status = f"A:{len(pair_state.price_a)}/B:{len(pair_state.price_b)}"
                    neutral_signal = SignalEvent(
                        source_agent=self.name,
                        strategy_name="stat_arb_pairs",
                        symbol=symbol,
                        direction=SignalDirection.FLAT,
                        strength=0.0,
                        confidence=0.3,
                        rationale=f"Pair {pair_key} monitoring: data={data_status}, zscore={pair_state.zscore:.2f}",
                        data_sources=("ib_market_data", "stat_arb_indicator"),
                        target_price=None,
                        stop_loss=None,
                    )
                    await self._event_bus.publish_signal(neutral_signal)
                    break  # Only need one signal per event

        # P2: Periodic cointegration check
        self._bar_counter += 1
        if self._bar_counter >= self._cointegration_check_interval:
            self._bar_counter = 0
            await self._check_all_cointegrations()

        # Check for correlation divergence signals (MoonDev-inspired)
        if self._divergence_enabled:
            div_signals = await self._check_divergence_signals(symbol, price)
            for div_signal in div_signals:
                if div_signal.confidence < self._min_confidence:
                    logger.debug(
                        f"Divergence signal filtered: confidence {div_signal.confidence:.2f} < threshold {self._min_confidence}"
                    )
                    continue
                await self._event_bus.publish_signal(div_signal)
                self._audit_logger.log_event(div_signal)

    async def _check_pair_signal(
        self,
        pair_key: str,
        pair_state: PairState,
    ) -> list[SignalEvent] | None:
        """
        Check if pair generates a trading signal.

        TODO: Implement proper stat arb model:
        1. Estimate cointegration (Engle-Granger or Johansen)
        2. Calculate hedge ratio (OLS, TLS, or Kalman)
        3. Compute spread and z-score
        4. Check for entry/exit signals
        """
        # FIX-25: Need sufficient data (120 bars = 2 hours). 10 bars = noise.
        min_data = 120
        if len(pair_state.price_a) < min_data or len(pair_state.price_b) < min_data:
            return None

        # Calculate spread
        prices_a = np.array(list(pair_state.price_a))
        prices_b = np.array(list(pair_state.price_b))

        # Align lengths
        min_len = min(len(prices_a), len(prices_b))
        prices_a = prices_a[-min_len:]
        prices_b = prices_b[-min_len:]

        # TODO: Implement proper hedge ratio estimation
        # Currently using simple ratio - should use OLS/Kalman
        pair_state.hedge_ratio = self._estimate_hedge_ratio(prices_a, prices_b)

        # Calculate spread
        spread = prices_a - pair_state.hedge_ratio * prices_b
        pair_state.spread_history.append(spread[-1])

        # Calculate z-score (reduced threshold for faster startup)
        if len(pair_state.spread_history) < 5:
            return None

        spread_array = np.array(list(pair_state.spread_history))
        spread_mean = np.mean(spread_array)
        spread_std = np.std(spread_array)

        if spread_std < 1e-8:
            return None

        zscore = (spread[-1] - spread_mean) / spread_std
        pair_state.zscore = zscore

        # P2: Update spread volatility for pair quality scoring
        pair_state.spread_volatility = spread_std

        # P2: Check for z-score alerts
        self._check_zscore_alerts(pair_key, zscore)

        # P2: Update pair quality score
        self._update_pair_quality(pair_key, pair_state, spread_array)

        # Generate signal based on z-score
        return self._generate_zscore_signal(pair_key, pair_state, zscore)

    def _estimate_hedge_ratio(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> float:
        """
        Estimate hedge ratio for the pair.

        TODO: Implement proper estimation:
        - OLS regression
        - Total Least Squares
        - Kalman filter for dynamic hedge ratio
        """
        # Simple OLS estimate
        try:
            beta = np.cov(prices_a, prices_b)[0, 1] / np.var(prices_b)
            return max(0.1, min(10.0, beta))  # Clamp to reasonable range
        except Exception as e:
            logger.error(f"Failed to calculate hedge ratio: {e}", exc_info=True)
            return 1.0  # Fallback to 1:1 ratio

    def _generate_zscore_signal(
        self,
        pair_key: str,
        pair_state: PairState,
        zscore: float,
    ) -> list[SignalEvent] | None:
        """
        Generate signals based on z-score.

        FIX-27: Emits two separate signals (one per leg) instead of a single
        signal with pair symbol "MES:MNQ" which IB cannot execute.
        """
        current_signal = pair_state.last_signal

        # P2: Check if pair is tradeable (allow signals with reduced confidence if not verified)
        tradeable, reason = self.is_pair_tradeable(pair_key)
        confidence_penalty = 0.0 if tradeable else 0.3  # Reduce confidence if not verified tradeable

        # Get last prices for SL/TP
        price_a = list(pair_state.price_a)[-1] if pair_state.price_a else 0
        price_b = list(pair_state.price_b)[-1] if pair_state.price_b else 0

        # Entry signals
        if zscore > self._zscore_entry and current_signal != SignalDirection.SHORT:
            # Spread too high - short A, long B
            pair_state.last_signal = SignalDirection.SHORT
            confidence = max(0.1, self._calculate_confidence(zscore, pair_state) - confidence_penalty)
            strength = min(1.0, zscore / 3.0)
            rationale = (
                f"Pair {pair_key} spread z-score={zscore:.2f} > {self._zscore_entry}. "
                f"Quality={pair_state.pair_quality_score:.2f}"
            )

            # FIX-27: Emit two executable signals, one per leg
            signals = []
            if price_a > 0:
                signals.append(SignalEvent(
                    source_agent=self.name,
                    strategy_name="stat_arb_pairs",
                    symbol=pair_state.symbol_a,
                    direction=SignalDirection.SHORT,
                    strength=-strength,
                    confidence=confidence,
                    target_price=round(price_a * 0.96, 2),
                    stop_loss=round(price_a * 1.02, 2),
                    rationale=f"Short leg: {rationale}",
                    data_sources=("ib_market_data", "stat_arb_indicator", "pair_correlation"),
                ))
            if price_b > 0:
                signals.append(SignalEvent(
                    source_agent=self.name,
                    strategy_name="stat_arb_pairs",
                    symbol=pair_state.symbol_b,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    confidence=confidence,
                    target_price=round(price_b * 1.04, 2),
                    stop_loss=round(price_b * 0.98, 2),
                    rationale=f"Long leg: {rationale}",
                    data_sources=("ib_market_data", "stat_arb_indicator", "pair_correlation"),
                ))
            return signals if signals else None

        elif zscore < -self._zscore_entry and current_signal != SignalDirection.LONG:
            # Spread too low - long A, short B
            pair_state.last_signal = SignalDirection.LONG
            confidence = max(0.1, self._calculate_confidence(zscore, pair_state) - confidence_penalty)
            strength = min(1.0, abs(zscore) / 3.0)
            rationale = (
                f"Pair {pair_key} spread z-score={zscore:.2f} < -{self._zscore_entry}. "
                f"Quality={pair_state.pair_quality_score:.2f}"
            )

            # FIX-27: Emit two executable signals, one per leg
            signals = []
            if price_a > 0:
                signals.append(SignalEvent(
                    source_agent=self.name,
                    strategy_name="stat_arb_pairs",
                    symbol=pair_state.symbol_a,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    confidence=confidence,
                    target_price=round(price_a * 1.04, 2),
                    stop_loss=round(price_a * 0.98, 2),
                    rationale=f"Long leg: {rationale}",
                    data_sources=("ib_market_data", "stat_arb_indicator", "pair_correlation"),
                ))
            if price_b > 0:
                signals.append(SignalEvent(
                    source_agent=self.name,
                    strategy_name="stat_arb_pairs",
                    symbol=pair_state.symbol_b,
                    direction=SignalDirection.SHORT,
                    strength=-strength,
                    confidence=confidence,
                    target_price=round(price_b * 0.96, 2),
                    stop_loss=round(price_b * 1.02, 2),
                    rationale=f"Short leg: {rationale}",
                    data_sources=("ib_market_data", "stat_arb_indicator", "pair_correlation"),
                ))
            return signals if signals else None

        # Exit signals
        elif abs(zscore) < self._zscore_exit and current_signal != SignalDirection.FLAT:
            pair_state.last_signal = SignalDirection.FLAT
            # FIX-27: Emit FLAT for both legs
            signals = []
            for sym in [pair_state.symbol_a, pair_state.symbol_b]:
                signals.append(SignalEvent(
                    source_agent=self.name,
                    strategy_name="stat_arb_pairs",
                    symbol=sym,
                    direction=SignalDirection.FLAT,
                    strength=0.0,
                    confidence=0.8,
                    rationale=f"Pair {pair_key} spread reverted, z-score={zscore:.2f}. Exit position.",
                    data_sources=("ib_market_data", "stat_arb_indicator", "pair_correlation"),
                    target_price=None,
                    stop_loss=None,
                ))
            return signals

        return None

    def _calculate_confidence(self, zscore: float, pair_state: PairState | None = None) -> float:
        """
        Calculate signal confidence based on z-score magnitude and pair quality (P2 enhanced).

        Incorporates:
        - Z-score magnitude
        - Cointegration strength (P2)
        - Pair quality score (P2)
        - Half-life of mean reversion
        """
        # Base confidence from z-score
        base_confidence = min(0.9, 0.5 + abs(zscore) * 0.1)

        if pair_state is None:
            return base_confidence

        # P2: Adjust confidence based on cointegration
        if pair_state.is_cointegrated:
            # Stronger cointegration (lower p-value) = higher confidence
            coint_boost = (1 - pair_state.cointegration_pvalue) * 0.1
            base_confidence = min(0.95, base_confidence + coint_boost)
        else:
            # Not cointegrated = reduce confidence
            base_confidence *= 0.7

        # P2: Adjust based on pair quality score
        quality_multiplier = 0.8 + pair_state.pair_quality_score * 0.4  # 0.8 to 1.2
        base_confidence *= quality_multiplier

        return min(0.95, max(0.1, base_confidence))

    # =========================================================================
    # P2: Z-SCORE ALERTS
    # =========================================================================

    def _check_zscore_alerts(self, pair_key: str, zscore: float) -> None:
        """
        Check for z-score threshold breaches and generate alerts (P2).

        Alert levels:
        - Warning: |z| > 2.5
        - Critical: |z| > 3.0
        - Extreme: |z| > 4.0 (potential regime break or data issue)
        """
        from datetime import datetime, timezone

        abs_zscore = abs(zscore)
        direction = "high" if zscore > 0 else "low"

        alert_level = None
        if abs_zscore >= self._zscore_extreme:
            alert_level = "extreme"
        elif abs_zscore >= self._zscore_critical:
            alert_level = "critical"
        elif abs_zscore >= self._zscore_warning:
            alert_level = "warning"

        if alert_level:
            message = (
                f"Pair {pair_key}: Z-score {alert_level.upper()} alert - "
                f"zscore={zscore:.2f} ({direction})"
            )

            alert = ZScoreAlert(
                pair_key=pair_key,
                timestamp=datetime.now(timezone.utc),
                zscore=zscore,
                alert_level=alert_level,
                direction=direction,
                message=message,
            )

            self._zscore_alerts.append(alert)

            # Trim old alerts
            if len(self._zscore_alerts) > self._max_alerts:
                self._zscore_alerts = self._zscore_alerts[-self._max_alerts:]

            # Log based on severity
            if alert_level == "extreme":
                logger.warning(f"StatArb: {message}")
            elif alert_level == "critical":
                logger.info(f"StatArb: {message}")
            else:
                logger.debug(f"StatArb: {message}")

    def get_zscore_alerts(self, hours: int = 24) -> list[dict]:
        """
        Get recent z-score alerts (P2).

        Args:
            hours: Lookback period

        Returns:
            List of alert dictionaries
        """
        from datetime import datetime, timezone, timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        return [
            {
                "pair_key": a.pair_key,
                "timestamp": a.timestamp.isoformat(),
                "zscore": a.zscore,
                "alert_level": a.alert_level,
                "direction": a.direction,
                "message": a.message,
            }
            for a in self._zscore_alerts
            if a.timestamp >= cutoff
        ]

    # =========================================================================
    # P2: COINTEGRATION MONITORING
    # =========================================================================

    async def _check_all_cointegrations(self) -> None:
        """
        Check cointegration status for all pairs (P2).

        Monitors for cointegration breakdown which signals pair may no longer be tradeable.
        """
        from datetime import datetime, timezone

        for pair_key, pair_state in self._pairs.items():
            if len(pair_state.price_a) < 100 or len(pair_state.price_b) < 100:
                continue

            prices_a = np.array(list(pair_state.price_a))
            prices_b = np.array(list(pair_state.price_b))

            # Align lengths
            min_len = min(len(prices_a), len(prices_b))
            prices_a = prices_a[-min_len:]
            prices_b = prices_b[-min_len:]

            # Perform ADF test on spread
            spread = prices_a - pair_state.hedge_ratio * prices_b

            adf_stat, pvalue = self._adf_test(spread)

            pair_state.cointegration_score = adf_stat
            pair_state.cointegration_pvalue = pvalue
            pair_state.cointegration_last_check = datetime.now(timezone.utc)

            was_cointegrated = pair_state.is_cointegrated
            pair_state.is_cointegrated = pvalue < self._cointegration_pvalue_threshold

            # Check for breakdown
            if was_cointegrated and not pair_state.is_cointegrated:
                pair_state.cointegration_breakdown_count += 1
                logger.warning(
                    f"StatArb: Cointegration breakdown detected for {pair_key} "
                    f"(p-value={pvalue:.4f}, breakdown #{pair_state.cointegration_breakdown_count})"
                )

                # If too many breakdowns, consider disabling the pair
                if pair_state.cointegration_breakdown_count >= self._cointegration_breakdown_threshold:
                    logger.error(
                        f"StatArb: Pair {pair_key} has broken down {pair_state.cointegration_breakdown_count} times - "
                        f"consider removing from trading"
                    )

            elif not was_cointegrated and pair_state.is_cointegrated:
                logger.info(
                    f"StatArb: Cointegration restored for {pair_key} (p-value={pvalue:.4f})"
                )

    def _adf_test(self, series: np.ndarray) -> tuple[float, float]:
        """
        Perform Augmented Dickey-Fuller test for stationarity (P2).

        Returns:
            Tuple of (ADF statistic, p-value)
        """
        try:
            # Simple ADF implementation without statsmodels
            # H0: series has a unit root (non-stationary)
            # Reject H0 if ADF stat < critical value

            n = len(series)
            if n < 20:
                return 0.0, 1.0

            # Calculate differences
            diff = np.diff(series)
            lag_series = series[:-1]

            # Simple regression: diff = alpha + beta * lag + epsilon
            # ADF statistic = beta / se(beta)
            x = lag_series - np.mean(lag_series)
            y = diff - np.mean(diff)

            beta = np.sum(x * y) / np.sum(x ** 2) if np.sum(x ** 2) > 0 else 0

            # Estimate standard error
            residuals = y - beta * x
            mse = np.sum(residuals ** 2) / (n - 2) if n > 2 else 1
            se_beta = np.sqrt(mse / np.sum(x ** 2)) if np.sum(x ** 2) > 0 else 1

            adf_stat = beta / se_beta if se_beta > 0 else 0

            # Approximate p-value using critical values
            # Critical values at 1%, 5%, 10% for n=100 are approximately -3.5, -2.9, -2.6
            if adf_stat < -3.5:
                pvalue = 0.01
            elif adf_stat < -2.9:
                pvalue = 0.05
            elif adf_stat < -2.6:
                pvalue = 0.10
            elif adf_stat < -1.9:
                pvalue = 0.25
            else:
                pvalue = 0.50 + min(0.49, abs(adf_stat) * 0.1)

            return adf_stat, pvalue

        except Exception as e:
            logger.error(f"ADF test failed: {e}", exc_info=True)
            return 0.0, 1.0

    def get_cointegration_status(self) -> dict[str, dict]:
        """
        Get cointegration status for all pairs (P2).

        Returns:
            Dictionary mapping pair_key to cointegration metrics
        """
        return {
            pair_key: {
                "is_cointegrated": ps.is_cointegrated,
                "score": ps.cointegration_score,
                "pvalue": ps.cointegration_pvalue,
                "breakdown_count": ps.cointegration_breakdown_count,
                "last_check": ps.cointegration_last_check.isoformat() if ps.cointegration_last_check else None,
            }
            for pair_key, ps in self._pairs.items()
        }

    # =========================================================================
    # P2: PAIR SELECTION OPTIMIZATION
    # =========================================================================

    def _update_pair_quality(
        self,
        pair_key: str,
        pair_state: PairState,
        spread_array: np.ndarray,
    ) -> None:
        """
        Update pair quality score for pair selection optimization (P2).

        Quality factors:
        1. Cointegration strength (lower p-value = better)
        2. Mean reversion speed (lower half-life = better)
        3. Spread volatility (moderate is ideal)
        4. Historical performance
        """
        # Factor 1: Cointegration strength (0 to 0.3)
        if pair_state.cointegration_pvalue < 0.01:
            coint_score = 0.3
        elif pair_state.cointegration_pvalue < 0.05:
            coint_score = 0.2
        elif pair_state.cointegration_pvalue < 0.10:
            coint_score = 0.1
        else:
            coint_score = 0.0

        # Factor 2: Mean reversion speed via half-life (0 to 0.3)
        half_life = self._estimate_half_life(spread_array)
        pair_state.half_life = half_life

        if half_life <= 0:
            reversion_score = 0.0
        elif half_life < 5:  # Very fast
            reversion_score = 0.3
        elif half_life < 10:  # Fast
            reversion_score = 0.25
        elif half_life < 20:  # Moderate
            reversion_score = 0.15
        elif half_life < 50:  # Slow
            reversion_score = 0.05
        else:
            reversion_score = 0.0

        pair_state.mean_reversion_strength = reversion_score / 0.3  # Normalize to 0-1

        # Factor 3: Spread volatility (0 to 0.2)
        # Want moderate volatility - too low means no opportunity, too high means risk
        spread_vol = pair_state.spread_volatility
        spread_mean = np.mean(spread_array) if len(spread_array) > 0 else 1
        vol_ratio = spread_vol / abs(spread_mean) if spread_mean != 0 else 0

        if 0.05 < vol_ratio < 0.20:  # Ideal range
            vol_score = 0.2
        elif 0.02 < vol_ratio < 0.30:  # Acceptable
            vol_score = 0.1
        else:
            vol_score = 0.0

        # Factor 4: Trading history (0 to 0.2)
        # Pairs with more data are more reliable
        data_points = min(len(pair_state.spread_history), self._lookback_size)
        data_score = min(0.2, data_points / self._lookback_size * 0.2)

        # Total quality score
        pair_state.pair_quality_score = coint_score + reversion_score + vol_score + data_score

        logger.debug(
            f"StatArb: Pair {pair_key} quality score = {pair_state.pair_quality_score:.3f} "
            f"(coint={coint_score:.2f}, rev={reversion_score:.2f}, vol={vol_score:.2f}, data={data_score:.2f})"
        )

    def _estimate_half_life(self, spread: np.ndarray) -> float:
        """
        Estimate mean reversion half-life using OLS (P2).

        Half-life = -ln(2) / theta where theta is the mean reversion coefficient.
        """
        try:
            n = len(spread)
            if n < 20:
                return float('inf')

            # Ornstein-Uhlenbeck: dS = theta * (mu - S) * dt + sigma * dW
            # Regression: spread[t] - spread[t-1] = theta * (mu - spread[t-1])
            lag_spread = spread[:-1]
            diff_spread = np.diff(spread)

            # OLS regression
            x = lag_spread - np.mean(lag_spread)
            y = diff_spread

            theta = -np.sum(x * y) / np.sum(x ** 2) if np.sum(x ** 2) > 0 else 0

            if theta <= 0:
                return float('inf')

            half_life = -np.log(2) / np.log(1 - theta) if theta < 1 else theta
            return max(0, half_life)

        except Exception as e:
            logger.error(f"Half-life estimation failed: {e}", exc_info=True)
            return float('inf')

    def get_pair_rankings(self) -> list[dict]:
        """
        Get pairs ranked by quality score (P2).

        Returns:
            List of pairs sorted by quality score (best first)
        """
        rankings = []
        for pair_key, ps in self._pairs.items():
            rankings.append({
                "pair_key": pair_key,
                "quality_score": ps.pair_quality_score,
                "is_cointegrated": ps.is_cointegrated,
                "half_life": ps.half_life,
                "zscore": ps.zscore,
                "breakdown_count": ps.cointegration_breakdown_count,
                "tradeable": (
                    ps.is_cointegrated and
                    ps.pair_quality_score >= self._min_pair_quality and
                    ps.cointegration_breakdown_count < self._cointegration_breakdown_threshold
                ),
            })

        # Sort by quality score descending
        rankings.sort(key=lambda x: x["quality_score"], reverse=True)
        return rankings

    def is_pair_tradeable(self, pair_key: str) -> tuple[bool, str]:
        """
        Check if a pair is tradeable based on quality metrics (P2).

        Returns:
            Tuple of (is_tradeable, reason)
        """
        if pair_key not in self._pairs:
            return False, f"Pair {pair_key} not found"

        ps = self._pairs[pair_key]

        if not ps.is_cointegrated:
            return False, f"Pair {pair_key} is not cointegrated (p={ps.cointegration_pvalue:.4f})"

        if ps.pair_quality_score < self._min_pair_quality:
            return False, f"Pair {pair_key} quality score too low ({ps.pair_quality_score:.3f} < {self._min_pair_quality})"

        if ps.cointegration_breakdown_count >= self._cointegration_breakdown_threshold:
            return False, f"Pair {pair_key} has too many breakdowns ({ps.cointegration_breakdown_count})"

        return True, "OK"

    # =========================================================================
    # CORRELATION DIVERGENCE DETECTION (MoonDev-inspired)
    # =========================================================================

    async def _check_divergence_signals(
        self,
        symbol: str,
        price: float,
    ) -> list[SignalEvent]:
        """
        Check for correlation divergence signals.

        Detects when normally correlated assets diverge significantly,
        signaling a mean-reversion opportunity.
        """
        from datetime import datetime, timezone

        signals: list[SignalEvent] = []

        for div_key, state in self._divergence_states.items():
            # Update prices
            if symbol == state.symbol_a:
                # Calculate return if we have previous price
                if state.returns_a:
                    prev_prices_a = list(state.returns_a)
                    if prev_prices_a:
                        last_price_a = prev_prices_a[-1] if isinstance(prev_prices_a[-1], (int, float)) else price
                        ret = (price - last_price_a) / last_price_a if last_price_a > 0 else 0
                        state.returns_a.append(ret)
                    else:
                        state.returns_a.append(0)
                else:
                    state.returns_a.append(0)

                # Store price for next iteration
                if not hasattr(state, '_last_price_a'):
                    state._last_price_a = price
                state._last_price_a = price

            elif symbol == state.symbol_b:
                if state.returns_b:
                    prev_prices_b = list(state.returns_b)
                    if prev_prices_b:
                        last_price_b = prev_prices_b[-1] if isinstance(prev_prices_b[-1], (int, float)) else price
                        ret = (price - last_price_b) / last_price_b if last_price_b > 0 else 0
                        state.returns_b.append(ret)
                    else:
                        state.returns_b.append(0)
                else:
                    state.returns_b.append(0)

                if not hasattr(state, '_last_price_b'):
                    state._last_price_b = price
                state._last_price_b = price

            else:
                continue

            # Need sufficient data for correlation
            if len(state.returns_a) < self._divergence_lookback or len(state.returns_b) < self._divergence_lookback:
                continue

            # Calculate rolling correlation
            returns_a = np.array(list(state.returns_a)[-self._divergence_lookback:])
            returns_b = np.array(list(state.returns_b)[-self._divergence_lookback:])

            # Align lengths
            min_len = min(len(returns_a), len(returns_b))
            if min_len < 10:
                continue

            returns_a = returns_a[-min_len:]
            returns_b = returns_b[-min_len:]

            # Calculate correlation
            correlation = self._calculate_correlation(returns_a, returns_b)
            state.current_correlation = correlation
            state.correlation_history.append(correlation)

            # Calculate baseline correlation (longer-term average)
            if len(state.correlation_history) >= 20:
                state.baseline_correlation = np.mean(list(state.correlation_history)[-30:])
            else:
                state.baseline_correlation = correlation

            # Detect divergence
            signal = self._detect_divergence_signal(div_key, state, returns_a, returns_b)
            if signal:
                signals.append(signal)

        return signals

    def _calculate_correlation(self, returns_a: np.ndarray, returns_b: np.ndarray) -> float:
        """Calculate Pearson correlation between two return series."""
        try:
            if len(returns_a) < 5 or len(returns_b) < 5:
                return 0.0

            # Remove mean
            a_centered = returns_a - np.mean(returns_a)
            b_centered = returns_b - np.mean(returns_b)

            # Calculate correlation
            numerator = np.sum(a_centered * b_centered)
            denominator = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))

            if denominator < 1e-10:
                return 0.0

            return numerator / denominator

        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return 0.0

    def _detect_divergence_signal(
        self,
        div_key: str,
        state: DivergenceState,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
    ) -> SignalEvent | None:
        """
        Detect divergence between correlated assets and generate signal.

        Strategy:
        - When correlation breaks down (drops significantly), expect reversion
        - When cumulative returns diverge, bet on convergence
        """
        from datetime import datetime, timezone

        # Calculate cumulative returns divergence
        cum_ret_a = np.sum(returns_a)
        cum_ret_b = np.sum(returns_b)
        divergence = cum_ret_a - cum_ret_b

        # Calculate divergence z-score
        divergence_history = []
        for i in range(min(len(returns_a), len(returns_b)) - 5):
            d = np.sum(returns_a[:i+5]) - np.sum(returns_b[:i+5])
            divergence_history.append(d)

        if len(divergence_history) < 5:
            return None

        div_mean = np.mean(divergence_history)
        div_std = np.std(divergence_history)

        if div_std < 1e-8:
            return None

        state.divergence_zscore = (divergence - div_mean) / div_std

        # Check for correlation breakdown
        correlation_dropped = (
            state.baseline_correlation > 0.5 and
            state.current_correlation < state.baseline_correlation - self._correlation_breakdown_threshold
        )

        # Generate signal on significant divergence
        abs_zscore = abs(state.divergence_zscore)

        if abs_zscore < self._divergence_threshold:
            # No significant divergence - check for exit
            if state.last_signal != SignalDirection.FLAT and abs_zscore < 0.5:
                state.last_signal = SignalDirection.FLAT
                state.divergence_start = None
                return SignalEvent(
                    source_agent=self.name,
                    strategy_name="correlation_divergence",
                    symbol=div_key,
                    direction=SignalDirection.FLAT,
                    strength=0.0,
                    confidence=0.7,
                    rationale=(
                        f"Divergence reverted for {state.symbol_a}/{state.symbol_b}. "
                        f"Z-score={state.divergence_zscore:.2f}, corr={state.current_correlation:.2f}"
                    ),
                    data_sources=("ib_market_data", "stat_arb_indicator", "pair_correlation"),
                    target_price=None,
                    stop_loss=None,
                )
            return None

        # Determine direction (bet on convergence)
        if state.divergence_zscore > self._divergence_threshold:
            # A outperformed B significantly - expect A to underperform (short A, long B)
            if state.last_signal != SignalDirection.SHORT:
                state.last_signal = SignalDirection.SHORT
                state.divergence_start = datetime.now(timezone.utc)

                confidence = min(0.85, 0.5 + abs_zscore * 0.1)
                if correlation_dropped:
                    confidence = min(0.9, confidence + 0.1)  # Higher confidence on correlation breakdown

                # Calculate stop_loss and target_price based on divergence z-score
                # For SHORT: divergence should decrease, target is lower z-score, stop is higher
                current_div = state.divergence_zscore
                stop_loss_div = current_div * 1.02  # 2% above (adverse move)
                target_div = current_div * 0.96  # 4% below (favorable move)

                return SignalEvent(
                    source_agent=self.name,
                    strategy_name="correlation_divergence",
                    symbol=div_key,
                    direction=SignalDirection.SHORT,
                    strength=-min(1.0, abs_zscore / 3.0),
                    confidence=confidence,
                    rationale=(
                        f"Divergence detected: {state.symbol_a} outperformed {state.symbol_b}. "
                        f"Z-score={state.divergence_zscore:.2f}, corr={state.current_correlation:.2f} "
                        f"(baseline={state.baseline_correlation:.2f}). "
                        f"Strategy: Short {state.symbol_a}, Long {state.symbol_b}"
                    ),
                    data_sources=("ib_market_data", "stat_arb_indicator", "pair_correlation"),
                    target_price=target_div,
                    stop_loss=stop_loss_div,
                )

        elif state.divergence_zscore < -self._divergence_threshold:
            # B outperformed A significantly - expect B to underperform (long A, short B)
            if state.last_signal != SignalDirection.LONG:
                state.last_signal = SignalDirection.LONG
                state.divergence_start = datetime.now(timezone.utc)

                confidence = min(0.85, 0.5 + abs_zscore * 0.1)
                if correlation_dropped:
                    confidence = min(0.9, confidence + 0.1)

                # Calculate stop_loss and target_price based on divergence z-score
                # For LONG: divergence should increase (less negative), target is higher z-score, stop is lower
                current_div = state.divergence_zscore
                stop_loss_div = current_div * 1.02  # 2% more negative (adverse move)
                target_div = current_div * 0.96  # 4% less negative (favorable move)

                return SignalEvent(
                    source_agent=self.name,
                    strategy_name="correlation_divergence",
                    symbol=div_key,
                    direction=SignalDirection.LONG,
                    strength=min(1.0, abs_zscore / 3.0),
                    confidence=confidence,
                    rationale=(
                        f"Divergence detected: {state.symbol_b} outperformed {state.symbol_a}. "
                        f"Z-score={state.divergence_zscore:.2f}, corr={state.current_correlation:.2f} "
                        f"(baseline={state.baseline_correlation:.2f}). "
                        f"Strategy: Long {state.symbol_a}, Short {state.symbol_b}"
                    ),
                    data_sources=("ib_market_data", "stat_arb_indicator", "pair_correlation"),
                    target_price=target_div,
                    stop_loss=stop_loss_div,
                )

        return None

    def get_divergence_status(self) -> dict[str, dict]:
        """Get divergence status for all tracked pairs."""
        return {
            div_key: {
                "symbol_a": state.symbol_a,
                "symbol_b": state.symbol_b,
                "current_correlation": state.current_correlation,
                "baseline_correlation": state.baseline_correlation,
                "divergence_zscore": state.divergence_zscore,
                "last_signal": state.last_signal.value,
                "data_points_a": len(state.returns_a),
                "data_points_b": len(state.returns_b),
            }
            for div_key, state in self._divergence_states.items()
        }

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

        # TODO: Load historical data to bootstrap cointegration estimates

    async def process_event(self, event: Event) -> None:
        """Process market data and generate stat arb signals."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol
        price = event.mid

        if price <= 0:
            return

        # Update relevant pairs
        signals = []
        for pair_key, pair_state in self._pairs.items():
            if symbol == pair_state.symbol_a:
                pair_state.price_a.append(price)
                signal = await self._check_pair_signal(pair_key, pair_state)
                if signal:
                    signals.append(signal)

            elif symbol == pair_state.symbol_b:
                pair_state.price_b.append(price)
                signal = await self._check_pair_signal(pair_key, pair_state)
                if signal:
                    signals.append(signal)

        # Publish signals
        for signal in signals:
            await self._event_bus.publish_signal(signal)
            self._audit_logger.log_event(signal)

    async def _check_pair_signal(
        self,
        pair_key: str,
        pair_state: PairState,
    ) -> SignalEvent | None:
        """
        Check if pair generates a trading signal.

        TODO: Implement proper stat arb model:
        1. Estimate cointegration (Engle-Granger or Johansen)
        2. Calculate hedge ratio (OLS, TLS, or Kalman)
        3. Compute spread and z-score
        4. Check for entry/exit signals
        """
        # Need sufficient data
        min_data = 100
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

        # Calculate z-score
        if len(pair_state.spread_history) < min_data:
            return None

        spread_array = np.array(list(pair_state.spread_history))
        spread_mean = np.mean(spread_array)
        spread_std = np.std(spread_array)

        if spread_std < 1e-8:
            return None

        zscore = (spread[-1] - spread_mean) / spread_std
        pair_state.zscore = zscore

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
        except Exception:
            return 1.0

    def _generate_zscore_signal(
        self,
        pair_key: str,
        pair_state: PairState,
        zscore: float,
    ) -> SignalEvent | None:
        """Generate signal based on z-score."""
        current_signal = pair_state.last_signal

        # Entry signals
        if zscore > self._zscore_entry and current_signal != SignalDirection.SHORT:
            # Spread too high - short A, long B
            pair_state.last_signal = SignalDirection.SHORT
            return SignalEvent(
                source_agent=self.name,
                strategy_name="stat_arb_pairs",
                symbol=pair_key,  # Pair identifier
                direction=SignalDirection.SHORT,
                strength=-min(1.0, zscore / 3.0),
                confidence=self._calculate_confidence(zscore),
                target_price=None,
                rationale=(
                    f"Pair {pair_key} spread z-score={zscore:.2f} > {self._zscore_entry}. "
                    f"Short {pair_state.symbol_a}, Long {pair_state.symbol_b}"
                ),
                data_sources=(pair_state.symbol_a, pair_state.symbol_b, "IB_market_data"),
            )

        elif zscore < -self._zscore_entry and current_signal != SignalDirection.LONG:
            # Spread too low - long A, short B
            pair_state.last_signal = SignalDirection.LONG
            return SignalEvent(
                source_agent=self.name,
                strategy_name="stat_arb_pairs",
                symbol=pair_key,
                direction=SignalDirection.LONG,
                strength=min(1.0, abs(zscore) / 3.0),
                confidence=self._calculate_confidence(zscore),
                target_price=None,
                rationale=(
                    f"Pair {pair_key} spread z-score={zscore:.2f} < -{self._zscore_entry}. "
                    f"Long {pair_state.symbol_a}, Short {pair_state.symbol_b}"
                ),
                data_sources=(pair_state.symbol_a, pair_state.symbol_b, "IB_market_data"),
            )

        # Exit signals
        elif abs(zscore) < self._zscore_exit and current_signal != SignalDirection.FLAT:
            pair_state.last_signal = SignalDirection.FLAT
            return SignalEvent(
                source_agent=self.name,
                strategy_name="stat_arb_pairs",
                symbol=pair_key,
                direction=SignalDirection.FLAT,
                strength=0.0,
                confidence=0.8,
                rationale=f"Pair {pair_key} spread reverted, z-score={zscore:.2f}. Exit position.",
                data_sources=(pair_state.symbol_a, pair_state.symbol_b, "IB_market_data"),
            )

        return None

    def _calculate_confidence(self, zscore: float) -> float:
        """
        Calculate signal confidence based on z-score magnitude.

        TODO: Incorporate additional factors:
        - Cointegration test p-value
        - Half-life of mean reversion
        - Historical hit rate
        """
        # Higher z-score = higher confidence (up to a point)
        confidence = min(0.9, 0.5 + abs(zscore) * 0.1)
        return confidence

"""
Options Volatility Agent
========================

Generates signals based on options market analysis.
Monitors implied volatility, skew, and term structure.

Responsibility: Options/volatility signal generation ONLY.
Does NOT make trading decisions or send orders.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
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
class VolatilityState:
    """State for volatility tracking per underlying."""
    symbol: str
    underlying_price: float = 0.0
    iv_history: deque = None  # Historical IV
    rv_history: deque = None  # Realized volatility (prices)
    return_history: deque = None  # Log returns for RV calculation
    skew_history: deque = None  # Put-call skew
    term_structure: dict = None  # IV by expiry
    iv_percentile: float = 50.0
    vol_premium: float = 0.0  # IV - RV
    realized_vol: float = 0.20  # Current RV estimate
    iv_is_estimated: bool = True  # Flag to indicate if IV is estimated vs. actual
    last_price: float = 0.0  # Previous price for return calculation

    def __post_init__(self):
        self.iv_history = deque(maxlen=252)  # 1 year daily
        self.rv_history = deque(maxlen=252)
        self.return_history = deque(maxlen=60)  # 60 days of returns for RV
        self.skew_history = deque(maxlen=100)
        self.term_structure = {}


class OptionsVolAgent(SignalAgent):
    """
    Options Volatility Agent.

    Analyzes options markets for volatility-based signals:
    1. IV percentile ranking
    2. Volatility risk premium (IV vs RV)
    3. Skew analysis
    4. Term structure

    Signal output:
    - Volatility regime (high/low/normal)
    - Vol selling/buying opportunities
    - Skew trades
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, event_bus, audit_logger)

        # Configuration
        self._iv_percentile_threshold = config.parameters.get("iv_percentile_threshold", 80)
        self._min_dte = config.parameters.get("min_dte", 7)
        self._max_dte = config.parameters.get("max_dte", 45)
        self._delta_range = config.parameters.get("delta_range", [0.20, 0.40])

        # State per underlying
        self._underlyings: dict[str, VolatilityState] = {}

    async def initialize(self) -> None:
        """Initialize volatility tracking."""
        logger.info(
            f"OptionsVolAgent initializing with IV percentile threshold={self._iv_percentile_threshold}, "
            f"DTE range={self._min_dte}-{self._max_dte}"
        )
        # TODO: Subscribe to VIX and options data feeds

    async def process_event(self, event: Event) -> None:
        """Process market data and generate volatility signals."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol

        # Get or create underlying state
        if symbol not in self._underlyings:
            self._underlyings[symbol] = VolatilityState(symbol=symbol)

        state = self._underlyings[symbol]
        state.underlying_price = event.mid

        # Calculate realized volatility
        self._update_realized_vol(state, event)

        # TODO: Get options data and calculate IV
        # For now, simulate IV updates
        await self._update_implied_vol(state, event)

        # Generate signals if we have enough data
        if len(state.iv_history) >= 20:
            signal = self._generate_vol_signal(state)
            if signal:
                await self._event_bus.publish_signal(signal)
                self._audit_logger.log_event(signal)

    def _update_realized_vol(
        self,
        state: VolatilityState,
        market_data: MarketDataEvent,
    ) -> None:
        """
        Update realized volatility estimate using close-to-close method.

        Implements standard historical volatility calculation:
        RV = std(log_returns) * sqrt(252)
        """
        price = market_data.mid

        # Calculate log return if we have a previous price
        if state.last_price > 0 and price > 0:
            log_return = math.log(price / state.last_price)
            state.return_history.append(log_return)

            # Calculate rolling realized volatility when we have enough data
            if len(state.return_history) >= 20:
                returns_array = np.array(list(state.return_history))
                # Annualized volatility
                state.realized_vol = float(np.std(returns_array) * np.sqrt(252))

        state.last_price = price
        state.rv_history.append(price)

    async def _update_implied_vol(
        self,
        state: VolatilityState,
        market_data: MarketDataEvent,
    ) -> None:
        """
        Update implied volatility from options.

        LIMITATION: Actual IV requires options chain data from IB.
        Until IB options data is integrated, we estimate IV based on:
        1. Realized volatility as baseline
        2. Historical IV premium (typically IV > RV)
        3. VIX correlation if available

        In production, this should:
        1. Query IB for ATM options chain
        2. Calculate IV via Black-Scholes inversion
        3. Build IV surface (strike x expiry)
        4. Calculate skew and term structure
        """
        state.iv_is_estimated = True

        # Estimate IV based on realized volatility with typical IV premium
        # Historically, IV tends to be ~10-20% higher than RV (risk premium)
        if state.realized_vol > 0:
            # Base IV on realized vol with premium
            rv_premium_multiplier = 1.15  # 15% premium over RV
            base_iv = state.realized_vol * rv_premium_multiplier

            # Add mean-reversion component (IV tends to mean-revert)
            long_term_mean_iv = 0.20  # Long-term average IV ~20%
            mean_reversion_rate = 0.05  # 5% pull toward mean per update

            # Blend current estimate with long-term mean
            estimated_iv = base_iv * (1 - mean_reversion_rate) + long_term_mean_iv * mean_reversion_rate

            # Clamp to reasonable bounds
            current_iv = max(0.08, min(0.80, estimated_iv))
        else:
            # Fallback when no RV available
            current_iv = 0.20  # Use long-term average

        state.iv_history.append(current_iv)

        # Calculate IV percentile
        if len(state.iv_history) >= 20:
            iv_array = np.array(list(state.iv_history))
            state.iv_percentile = (np.sum(iv_array < current_iv) / len(iv_array)) * 100

        # Calculate vol premium (IV - RV)
        state.vol_premium = current_iv - state.realized_vol

        # Log warning periodically about estimated IV
        if len(state.iv_history) % 100 == 1:
            logger.warning(
                f"Using ESTIMATED IV for {state.symbol} (IV={current_iv*100:.1f}%, "
                f"RV={state.realized_vol*100:.1f}%). Integrate IB options data for actual IV."
            )

    def _generate_vol_signal(self, state: VolatilityState) -> SignalEvent | None:
        """
        Generate volatility-based trading signal with option selection criteria.

        Strategies based on IV percentile:
        - IV rank > threshold: Sell premium (iron condor, strangle)
        - IV rank < (100 - threshold): Buy premium (straddle, calendar)

        Options filtered by configured delta range and DTE.
        """
        current_iv = state.iv_history[-1] if state.iv_history else 0
        estimated_flag = " [ESTIMATED]" if state.iv_is_estimated else ""

        # High IV - sell premium opportunity
        if state.iv_percentile > self._iv_percentile_threshold:
            direction = SignalDirection.SHORT

            # Get optimal option recommendation
            opt_rec = self.get_optimal_option_for_signal(state, direction)
            opt_info = ""
            if opt_rec:
                opt_info = (
                    f" | Recommended: {opt_rec['dte_days']:.0f} DTE, "
                    f"strike ~{opt_rec['strike']:.2f} (delta ~{opt_rec['target_delta']:.2f})"
                )

            return SignalEvent(
                source_agent=self.name,
                strategy_name="options_vol_sell",
                symbol=state.symbol,
                direction=direction,
                strength=-0.6,
                confidence=self._calculate_confidence(state),
                rationale=(
                    f"High IV percentile ({state.iv_percentile:.1f}%){estimated_flag}. "
                    f"IV={current_iv*100:.1f}%, RV={state.realized_vol*100:.1f}%, "
                    f"Vol premium={state.vol_premium*100:.1f}%. "
                    f"Consider selling premium.{opt_info}"
                ),
                data_sources=(state.symbol, "options_chain", "IB_market_data"),
            )

        # Low IV - buy premium opportunity
        elif state.iv_percentile < (100 - self._iv_percentile_threshold):
            direction = SignalDirection.LONG

            # Get optimal option recommendation
            opt_rec = self.get_optimal_option_for_signal(state, direction)
            opt_info = ""
            if opt_rec:
                opt_info = (
                    f" | Recommended: {opt_rec['dte_days']:.0f} DTE, "
                    f"strike ~{opt_rec['strike']:.2f} (delta ~{opt_rec['target_delta']:.2f})"
                )

            return SignalEvent(
                source_agent=self.name,
                strategy_name="options_vol_buy",
                symbol=state.symbol,
                direction=direction,
                strength=0.4,
                confidence=self._calculate_confidence(state) * 0.8,  # Lower confidence for long vol
                rationale=(
                    f"Low IV percentile ({state.iv_percentile:.1f}%){estimated_flag}. "
                    f"IV={current_iv*100:.1f}%, RV={state.realized_vol*100:.1f}%, "
                    f"Vol premium={state.vol_premium*100:.1f}%. "
                    f"Consider buying premium.{opt_info}"
                ),
                data_sources=(state.symbol, "options_chain", "IB_market_data"),
            )

        return None

    def _calculate_confidence(self, state: VolatilityState) -> float:
        """
        Calculate signal confidence.

        TODO: Factor in:
        - IV percentile extremity
        - Vol premium magnitude
        - Skew normality
        - Term structure shape
        """
        # Base confidence on IV percentile extremity
        extremity = abs(state.iv_percentile - 50) / 50  # 0 to 1

        # Adjust for vol premium
        premium_factor = 1.0 if state.vol_premium > 0 else 0.8

        confidence = 0.5 + (extremity * 0.3) * premium_factor

        return min(0.85, confidence)

    def calculate_black_scholes_iv(
        self,
        option_price: float,
        spot: float,
        strike: float,
        tte: float,  # Time to expiry in years
        rate: float,
        is_call: bool,
        dividend_yield: float = 0.0,
        max_iterations: int = 100,
        precision: float = 1e-6,
    ) -> float | None:
        """
        Calculate implied volatility via Newton-Raphson method.

        Uses Black-Scholes-Merton formula with continuous dividend yield
        and iteratively solves for IV.

        The dividend yield adjustment uses the Merton extension:
        - S * exp(-q*T) replaces S in the formula
        - d1 = (ln(S/K) + (r - q + σ²/2)T) / (σ√T)

        Args:
            option_price: Market price of the option
            spot: Current spot price of underlying
            strike: Strike price
            tte: Time to expiry in years
            rate: Risk-free rate (annualized)
            is_call: True for call, False for put
            dividend_yield: Continuous dividend yield (annualized)
            max_iterations: Max Newton-Raphson iterations
            precision: Convergence threshold

        Returns:
            Implied volatility or None if convergence fails
        """
        from scipy.stats import norm

        if option_price <= 0 or spot <= 0 or strike <= 0 or tte <= 0:
            return None

        # Dividend-adjusted spot for initial guess
        adj_spot = spot * math.exp(-dividend_yield * tte)

        # Initial guess based on at-the-money approximation
        sigma = math.sqrt(2 * math.pi / tte) * option_price / adj_spot
        sigma = max(0.01, min(2.0, sigma))  # Reasonable starting bounds

        for _ in range(max_iterations):
            sqrt_tte = math.sqrt(tte)

            # Calculate d1 and d2 with dividend adjustment
            # d1 = (ln(S/K) + (r - q + σ²/2)T) / (σ√T)
            d1 = (math.log(spot / strike) + (rate - dividend_yield + sigma**2 / 2) * tte) / (sigma * sqrt_tte)
            d2 = d1 - sigma * sqrt_tte

            # Discount factors
            df_div = math.exp(-dividend_yield * tte)  # Dividend discount
            df_rate = math.exp(-rate * tte)  # Rate discount

            # Calculate option price using current sigma (Black-Scholes-Merton)
            if is_call:
                price = spot * df_div * norm.cdf(d1) - strike * df_rate * norm.cdf(d2)
            else:
                price = strike * df_rate * norm.cdf(-d2) - spot * df_div * norm.cdf(-d1)

            # Calculate vega (same formula, with dividend adjustment)
            vega = spot * df_div * sqrt_tte * norm.pdf(d1)

            if vega < 1e-10:
                return None

            # Newton-Raphson step
            diff = option_price - price
            if abs(diff) < precision:
                return sigma

            sigma = sigma + diff / vega

            # Bounds check
            if sigma <= 0.001 or sigma > 5.0:
                return None

        return None

    def calculate_delta(
        self,
        spot: float,
        strike: float,
        tte: float,
        sigma: float,
        rate: float,
        is_call: bool,
        dividend_yield: float = 0.0,
    ) -> float:
        """
        Calculate option delta using Black-Scholes-Merton with dividend yield.

        For dividend-paying stocks:
        - Call delta = exp(-q*T) * N(d1)
        - Put delta = exp(-q*T) * (N(d1) - 1)

        Args:
            spot: Current spot price
            strike: Strike price
            tte: Time to expiry in years
            sigma: Volatility
            rate: Risk-free rate
            is_call: True for call, False for put
            dividend_yield: Continuous dividend yield (annualized)

        Returns:
            Delta value (-1 to 1)
        """
        from scipy.stats import norm

        if tte <= 0 or sigma <= 0:
            return 0.0

        sqrt_tte = math.sqrt(tte)

        # d1 with dividend adjustment
        d1 = (math.log(spot / strike) + (rate - dividend_yield + sigma**2 / 2) * tte) / (sigma * sqrt_tte)

        # Dividend discount factor
        df_div = math.exp(-dividend_yield * tte)

        if is_call:
            return df_div * norm.cdf(d1)
        else:
            return df_div * (norm.cdf(d1) - 1)

    def filter_options_by_criteria(
        self,
        options: list[dict],
        spot: float,
        rate: float = 0.05,
    ) -> list[dict]:
        """
        Filter options based on configured delta range and DTE criteria.

        Args:
            options: List of option dicts with keys: strike, tte, sigma, is_call, price
            spot: Current spot price
            rate: Risk-free rate

        Returns:
            Filtered list of options meeting criteria
        """
        filtered = []
        min_delta, max_delta = self._delta_range

        for opt in options:
            # Check DTE criteria
            dte_days = opt.get("tte", 0) * 365
            if dte_days < self._min_dte or dte_days > self._max_dte:
                continue

            # Calculate delta
            delta = self.calculate_delta(
                spot=spot,
                strike=opt.get("strike", 0),
                tte=opt.get("tte", 0),
                sigma=opt.get("sigma", 0.20),
                rate=rate,
                is_call=opt.get("is_call", True),
            )

            # Check delta criteria (use absolute value)
            abs_delta = abs(delta)
            if min_delta <= abs_delta <= max_delta:
                opt["delta"] = delta
                filtered.append(opt)

        return filtered

    def get_optimal_option_for_signal(
        self,
        state: VolatilityState,
        direction: SignalDirection,
        rate: float = 0.05,
    ) -> dict | None:
        """
        Find optimal option for given signal direction based on config criteria.

        For vol selling (SHORT signal): Look for high IV options with delta in range
        For vol buying (LONG signal): Look for low IV options with delta in range

        Returns:
            Dict with recommended option parameters or None
        """
        # Target DTE in middle of range
        target_dte = (self._min_dte + self._max_dte) / 2
        tte = target_dte / 365.0

        # Target delta in middle of range
        target_delta = (self._delta_range[0] + self._delta_range[1]) / 2

        # Use current IV estimate
        current_iv = state.iv_history[-1] if state.iv_history else 0.20

        # Calculate approximate strike for target delta
        # Using simplified inverse of delta formula
        spot = state.underlying_price
        if spot <= 0:
            return None

        from scipy.stats import norm

        # For calls with target delta, solve for K
        # delta = N(d1), so d1 = N_inv(delta)
        d1 = norm.ppf(target_delta)
        # d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
        # Solve for K: K = S * exp(-(d1 * σ√T - (r + σ²/2)T))
        sqrt_t = math.sqrt(tte)
        approx_strike = spot * math.exp(
            -(d1 * current_iv * sqrt_t - (rate + current_iv**2 / 2) * tte)
        )

        return {
            "symbol": state.symbol,
            "strike": round(approx_strike, 2),
            "dte_days": target_dte,
            "tte": tte,
            "target_delta": target_delta,
            "estimated_iv": current_iv,
            "direction": direction.value,
            "strategy": "sell_premium" if direction == SignalDirection.SHORT else "buy_premium",
        }

"""
Market Making Strategy
======================

Implements market making logic (IB-compatible latency).

MATURITY: ALPHA
---------------
Status: Basic framework - NOT ready for production
- [x] Fair value estimation (size-weighted mid)
- [x] Basic spread calculation (A-S inspired)
- [x] Inventory management (skew adjustment)
- [x] Quote generation
- [x] Avellaneda-Stoikov with order arrival rate
- [ ] Order arrival rate estimation (TODO - currently uses default)
- [ ] Kyle's lambda (adverse selection) (TODO)
- [ ] VPIN (order flow toxicity) (TODO)

Production Readiness:
- Unit tests: Minimal
- Backtesting: Not performed
- Live testing: Not performed

WARNING: DO NOT USE IN PRODUCTION
- This is a simplified placeholder implementation
- Real market making requires:
  * Sub-100ms latency (not achievable with IB)
  * Sophisticated inventory models
  * Adverse selection detection
  * Fee structure optimization

IMPORTANT: Competitive market making is NOT viable with IB latency.
This implementation is designed for passive market making only,
with quote refresh intervals of 5+ seconds. For competitive MM,
you need direct market access with sub-10ms latency.

NOTE: Not HFT - operates at 100ms+ latencies due to IB constraints.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


# =============================================================================
# AVELLANEDA-STOIKOV OPTIMAL MARKET MAKING
# =============================================================================


@dataclass
class AvellanedaStoikovParams:
    """
    Parameters for Avellaneda-Stoikov optimal market making model.

    Reference: Avellaneda & Stoikov (2008) "High-frequency trading in a limit order book"

    The model provides closed-form solutions for:
    - Optimal reservation price (adjusted mid based on inventory)
    - Optimal bid-ask spread (balancing adverse selection vs fill probability)

    Key parameters:
    - gamma: Risk aversion (higher = more conservative, wider spreads)
    - sigma: Asset volatility (daily or appropriate timeframe)
    - k: Order arrival intensity (orders per time unit)
    - A: Order arrival sensitivity to spread (how much spread affects fills)
    - T: Time horizon for optimization (fraction of day remaining)
    """
    gamma: float = 0.1          # Risk aversion parameter (typical: 0.01-1.0)
    sigma: float = 0.02         # Daily volatility (e.g., 2% = 0.02)
    k: float = 1.5              # Order arrival intensity (orders/time)
    A: float = 140.0            # Arrival sensitivity parameter
    T: float = 1.0              # Time horizon (1.0 = full day remaining)
    max_inventory: int = 1000   # Maximum inventory limit
    min_spread_bps: float = 5.0 # Minimum spread in basis points
    tick_size: float = 0.01     # Price tick size

    def validate(self) -> list[str]:
        """Validate parameters and return list of issues."""
        issues = []
        if self.gamma <= 0:
            issues.append("gamma must be positive")
        if self.sigma <= 0:
            issues.append("sigma must be positive")
        if self.k <= 0:
            issues.append("k (order arrival) must be positive")
        if self.A <= 0:
            issues.append("A (arrival sensitivity) must be positive")
        if self.T <= 0 or self.T > 1:
            issues.append("T must be in (0, 1]")
        return issues


@dataclass
class ASQuoteResult:
    """Result of Avellaneda-Stoikov quote calculation."""
    reservation_price: float
    optimal_spread: float
    bid_price: float
    ask_price: float
    bid_delta: float        # Distance from mid to bid
    ask_delta: float        # Distance from mid to ask
    inventory_adjustment: float
    spread_bps: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reservation_price": self.reservation_price,
            "optimal_spread": self.optimal_spread,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "bid_delta": self.bid_delta,
            "ask_delta": self.ask_delta,
            "inventory_adjustment": self.inventory_adjustment,
            "spread_bps": self.spread_bps,
            "timestamp": self.timestamp.isoformat(),
        }


def calculate_reservation_price(
    mid: float,
    inventory: int,
    gamma: float,
    sigma: float,
    T: float,
) -> float:
    """
    Calculate optimal reservation price (indifference price).

    The reservation price is the price at which the market maker is
    indifferent between holding or liquidating inventory.

    Formula: r = s - q * gamma * sigma^2 * T

    Where:
    - s: Current mid price
    - q: Current inventory
    - gamma: Risk aversion
    - sigma: Volatility
    - T: Time remaining

    Args:
        mid: Current mid price
        inventory: Current inventory (positive = long, negative = short)
        gamma: Risk aversion parameter
        sigma: Volatility (daily)
        T: Time horizon (0 to 1)

    Returns:
        Reservation (indifference) price
    """
    # Inventory adjustment term
    # When long (q > 0), reservation price < mid (want to sell)
    # When short (q < 0), reservation price > mid (want to buy)
    adjustment = inventory * gamma * (sigma ** 2) * T

    reservation = mid - adjustment

    return reservation


def calculate_optimal_spread(
    gamma: float,
    sigma: float,
    k: float,
    A: float,
) -> float:
    """
    Calculate optimal bid-ask spread.

    The optimal spread balances:
    - Wider spread = more profit per trade but fewer fills
    - Narrower spread = more fills but less profit per trade

    Formula: delta = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/k)

    For the simplified case (constant arrival rate):
    delta = (2/gamma) * ln(1 + gamma/k)

    The full formula includes time-dependent terms.

    Args:
        gamma: Risk aversion parameter
        sigma: Volatility
        k: Order arrival intensity
        A: Arrival sensitivity (not used in basic formula)

    Returns:
        Optimal half-spread (bid and ask distance from reservation price)
    """
    # Basic spread component from order arrival
    if gamma > 0 and k > 0:
        spread_component = (2 / gamma) * math.log(1 + gamma / k)
    else:
        spread_component = 0.01  # Default minimum

    # Volatility component
    vol_component = gamma * (sigma ** 2)

    optimal_spread = spread_component + vol_component

    return optimal_spread


def calculate_optimal_quotes(
    mid: float,
    inventory: int,
    params: AvellanedaStoikovParams,
) -> ASQuoteResult:
    """
    Calculate optimal bid and ask prices using Avellaneda-Stoikov model.

    This is the main entry point for A-S quote generation.

    Args:
        mid: Current mid price
        inventory: Current inventory position
        params: A-S model parameters

    Returns:
        ASQuoteResult with optimal quotes
    """
    # Validate parameters
    issues = params.validate()
    if issues:
        logger.warning(f"A-S parameter issues: {issues}")

    # Calculate reservation price
    reservation = calculate_reservation_price(
        mid=mid,
        inventory=inventory,
        gamma=params.gamma,
        sigma=params.sigma,
        T=params.T,
    )

    # Calculate optimal spread
    half_spread = calculate_optimal_spread(
        gamma=params.gamma,
        sigma=params.sigma,
        k=params.k,
        A=params.A,
    ) / 2

    # Apply minimum spread
    min_half_spread = (params.min_spread_bps / 10000) * mid / 2
    half_spread = max(half_spread, min_half_spread)

    # Calculate bid and ask
    bid_price = reservation - half_spread
    ask_price = reservation + half_spread

    # Round to tick size
    bid_price = round(bid_price / params.tick_size) * params.tick_size
    ask_price = round(ask_price / params.tick_size) * params.tick_size

    # Ensure positive spread
    if ask_price <= bid_price:
        ask_price = bid_price + params.tick_size

    # Calculate metrics
    inventory_adjustment = mid - reservation
    spread_bps = (ask_price - bid_price) / mid * 10000

    return ASQuoteResult(
        reservation_price=reservation,
        optimal_spread=ask_price - bid_price,
        bid_price=bid_price,
        ask_price=ask_price,
        bid_delta=mid - bid_price,
        ask_delta=ask_price - mid,
        inventory_adjustment=inventory_adjustment,
        spread_bps=spread_bps,
    )


def update_inventory_risk(
    current_inventory: int,
    max_inventory: int,
) -> tuple[float, str]:
    """
    Calculate inventory risk level and recommended action.

    Args:
        current_inventory: Current position
        max_inventory: Maximum allowed inventory

    Returns:
        (risk_level, action) tuple
        - risk_level: 0.0 (safe) to 1.0 (at limit)
        - action: Recommended action string
    """
    if max_inventory <= 0:
        return 1.0, "invalid_max_inventory"

    inventory_ratio = abs(current_inventory) / max_inventory
    risk_level = min(inventory_ratio, 1.0)

    if risk_level < 0.5:
        action = "normal_quoting"
    elif risk_level < 0.8:
        action = "reduce_exposure"
    elif risk_level < 1.0:
        action = "aggressive_reduction"
    else:
        action = "stop_quoting"

    return risk_level, action


class AvellanedaStoikovMM:
    """
    Avellaneda-Stoikov Market Making strategy class.

    Provides a complete implementation of the A-S optimal market making
    framework with practical extensions for live trading.

    Features:
    - Dynamic parameter adjustment based on market conditions
    - Inventory risk management
    - Volatility regime adaptation
    - Quote caching for efficiency
    """

    def __init__(
        self,
        params: AvellanedaStoikovParams | None = None,
        volatility_lookback: int = 20,
        adapt_to_volatility: bool = True,
    ):
        """
        Initialize A-S market maker.

        Args:
            params: A-S parameters (uses defaults if None)
            volatility_lookback: Periods for volatility estimation
            adapt_to_volatility: Whether to adjust params for volatility
        """
        self.params = params or AvellanedaStoikovParams()
        self.volatility_lookback = volatility_lookback
        self.adapt_to_volatility = adapt_to_volatility

        # State tracking
        self._returns_history: list[float] = []
        self._current_volatility: float = self.params.sigma
        self._last_quote: ASQuoteResult | None = None

    def update_volatility(self, price: float) -> None:
        """
        Update volatility estimate with new price.

        Args:
            price: Latest price observation
        """
        if len(self._returns_history) > 0:
            # Simple return calculation
            # In production, use log returns for accuracy
            last_price = self._returns_history[-1]
            if last_price > 0:
                ret = (price - last_price) / last_price
                self._returns_history.append(price)

                # Keep only recent history
                if len(self._returns_history) > self.volatility_lookback + 1:
                    self._returns_history = self._returns_history[-self.volatility_lookback - 1:]

                # Calculate realized volatility
                if len(self._returns_history) >= 3:
                    returns = []
                    for i in range(1, len(self._returns_history)):
                        r = (self._returns_history[i] - self._returns_history[i-1]) / self._returns_history[i-1]
                        returns.append(r)
                    if returns:
                        self._current_volatility = float(np.std(returns)) * np.sqrt(252)
        else:
            self._returns_history.append(price)

    def update_time_horizon(self, time_remaining_fraction: float) -> None:
        """
        Update time horizon parameter.

        Typically called periodically during the trading day.

        Args:
            time_remaining_fraction: Fraction of trading day remaining (0-1)
        """
        self.params.T = max(0.01, min(1.0, time_remaining_fraction))

    def generate_quotes(
        self,
        mid: float,
        inventory: int,
        override_params: AvellanedaStoikovParams | None = None,
    ) -> ASQuoteResult:
        """
        Generate optimal quotes using A-S model.

        Args:
            mid: Current mid price
            inventory: Current inventory position
            override_params: Optional parameter override

        Returns:
            ASQuoteResult with optimal bid/ask
        """
        params = override_params or self.params

        # Optionally adapt sigma to current volatility
        if self.adapt_to_volatility and self._current_volatility > 0:
            params = AvellanedaStoikovParams(
                gamma=params.gamma,
                sigma=self._current_volatility,
                k=params.k,
                A=params.A,
                T=params.T,
                max_inventory=params.max_inventory,
                min_spread_bps=params.min_spread_bps,
                tick_size=params.tick_size,
            )

        result = calculate_optimal_quotes(mid, inventory, params)
        self._last_quote = result

        return result

    def get_inventory_risk(self, inventory: int) -> tuple[float, str]:
        """
        Get current inventory risk level.

        Args:
            inventory: Current position

        Returns:
            (risk_level, action) tuple
        """
        return update_inventory_risk(inventory, self.params.max_inventory)

    def should_quote(
        self,
        inventory: int,
        market_spread_bps: float,
    ) -> tuple[bool, str]:
        """
        Determine if quoting is advisable.

        Args:
            inventory: Current inventory
            market_spread_bps: Current market spread in bps

        Returns:
            (should_quote, reason) tuple
        """
        risk_level, action = self.get_inventory_risk(inventory)

        if action == "stop_quoting":
            return False, "inventory_at_limit"

        if market_spread_bps < self.params.min_spread_bps * 0.8:
            return False, "market_spread_too_tight"

        return True, "ok"

    def calculate_expected_pnl(
        self,
        inventory: int,
        fill_probability: float = 0.5,
    ) -> float:
        """
        Estimate expected P&L from current quotes.

        This is a simplified calculation assuming symmetric fills.

        Args:
            inventory: Current inventory
            fill_probability: Probability of fill (0-1)

        Returns:
            Expected P&L in price units
        """
        if self._last_quote is None:
            return 0.0

        # Spread capture
        spread_capture = self._last_quote.optimal_spread * fill_probability

        # Inventory cost (variance of P&L from inventory)
        inventory_cost = (
            abs(inventory) *
            self.params.gamma *
            (self.params.sigma ** 2) *
            self.params.T
        )

        return spread_capture - inventory_cost


@dataclass
class Quote:
    """A bid/ask quote."""
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    mid_price: float
    spread: float


@dataclass
class MarketMakingSignal:
    """Market making signal output."""
    symbol: str
    action: str  # "quote", "cancel", "adjust", "exit"
    quote: Quote | None
    inventory_action: str | None  # "reduce_long", "reduce_short", None
    urgency: float  # 0 to 1
    # P2: Enhanced inventory skew
    inventory_skew_bps: float = 0.0  # Skew in basis points
    # P2: Adverse selection metrics
    adverse_selection_risk: float = 0.0  # 0-1 risk score
    toxic_flow_detected: bool = False
    # P2: Volatility-adjusted quotes
    volatility_adjustment_bps: float = 0.0  # Spread adjustment for volatility
    # RISK-001: Stop-loss levels
    inventory_stop_loss: int | None = None     # Max inventory before forced exit
    pnl_stop_loss: float | None = None         # P&L threshold for stop
    # RISK-002: Maximum holding period
    max_position_duration_seconds: int = 300   # Max time to hold inventory position
    # RISK-003: Strategy-level risk limit
    strategy_max_loss_pct: float = 2.0         # Max daily loss for MM strategy
    daily_loss_limit: float | None = None      # Absolute daily loss limit
    # RISK-004: Regime detection
    market_regime: str = "normal"              # "normal", "trending", "volatile", "illiquid"
    regime_suitable: bool = True               # False if regime unfavorable for MM
    regime_warning: str | None = None
    # RISK-005: Exit signal info
    is_exit_signal: bool = False
    exit_reason: str | None = None
    # RISK-006: Spread floor
    min_profitable_spread_bps: float = 5.0     # Minimum spread to be profitable


class MarketMakingStrategy:
    """
    Market Making Strategy Implementation.

    Implements:
    1. Fair value estimation
    2. Optimal spread calculation
    3. Inventory management
    4. Adverse selection avoidance

    NOT HFT - All quotes refreshed at IB-compatible intervals (100ms+).

    TODO: Implement proper models:
    - Avellaneda-Stoikov optimal market making
    - Kyle's lambda for adverse selection
    - Order flow toxicity (VPIN)
    """

    def __init__(self, config: dict[str, Any]):
        self._min_spread_bps = config.get("spread_bps", 10)
        self._max_inventory = config.get("max_inventory", 1000)
        self._risk_aversion = config.get("risk_aversion", 0.1)
        # Default 5000ms for passive market making (competitive MM not viable with IB)
        self._quote_refresh_ms = config.get("quote_refresh_ms", 5000)
        # Order arrival rate for Avellaneda-Stoikov model (orders per time unit)
        self._order_arrival_rate = config.get("order_arrival_rate", 1.0)

        # Ensure not HFT
        if self._quote_refresh_ms < 100:
            logger.warning("Quote refresh < 100ms, adjusting to 100ms (IB compatible)")
            self._quote_refresh_ms = 100

        # P2: Enhanced inventory skew settings
        self._inventory_skew_factor = config.get("inventory_skew_factor", 0.5)  # bps per % inventory
        self._max_skew_bps = config.get("max_skew_bps", 10)  # Maximum skew in bps

        # P2: Adverse selection detection settings
        self._adverse_selection_lookback = config.get("adverse_selection_lookback", 20)
        self._toxic_flow_threshold = config.get("toxic_flow_threshold", 0.6)  # VPIN-like threshold
        self._price_impact_threshold = config.get("price_impact_threshold", 0.001)  # 10bps

        # P2: Volatility adjustment settings
        self._vol_spread_multiplier = config.get("vol_spread_multiplier", 1.5)
        self._base_volatility = config.get("base_volatility", 0.01)  # 1% daily vol baseline
        self._max_vol_adjustment_bps = config.get("max_vol_adjustment_bps", 20)

        # RISK-001: Stop-loss settings
        self._inventory_stop_loss_pct = config.get("inventory_stop_loss_pct", 120)  # 120% of max
        self._pnl_stop_loss = config.get("pnl_stop_loss", -1000.0)  # Absolute P&L stop

        # RISK-002: Maximum holding period
        self._max_position_duration_seconds = config.get("max_position_duration_seconds", 300)

        # RISK-003: Strategy-level risk limit
        self._strategy_max_loss_pct = config.get("strategy_max_loss_pct", 2.0)  # 2% daily max
        self._daily_loss_limit = config.get("daily_loss_limit", -5000.0)  # Absolute daily limit

        # RISK-004: Regime detection thresholds
        self._trending_threshold = config.get("trending_threshold", 0.02)  # 2% move = trending
        self._volatile_threshold = config.get("volatile_threshold", 2.0)  # 2x normal vol
        self._illiquidity_threshold = config.get("illiquidity_threshold", 0.5)  # 50% of normal volume

        # RISK-006: Spread floor
        self._min_profitable_spread_bps = config.get("min_profitable_spread_bps", 5.0)

    def estimate_fair_value(
        self,
        bid: float,
        ask: float,
        bid_size: int,
        ask_size: int,
    ) -> float:
        """
        Estimate fair value from order book.

        Uses size-weighted mid price.

        TODO: Implement more sophisticated fair value:
        - Micro-price
        - Order flow imbalance
        - Volume-weighted price
        """
        if bid_size + ask_size == 0:
            return (bid + ask) / 2

        # Size-weighted mid (micro-price approximation)
        # Weight each price by its own size: larger size = more liquidity at that price
        fair_value = (bid * bid_size + ask * ask_size) / (bid_size + ask_size)

        return fair_value

    def calculate_optimal_spread(
        self,
        volatility: float,
        inventory: int,
        time_horizon: float = 1.0,
        order_arrival_rate: float | None = None,
    ) -> float:
        """
        Calculate optimal spread using Avellaneda-Stoikov framework.

        Full A-S formula:
        s* = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/k)

        Where:
        - gamma: risk aversion parameter
        - sigma: volatility
        - T: time horizon
        - k: order arrival rate (lambda in original paper)

        The first term compensates for inventory risk over time.
        The second term accounts for adverse selection from order flow.
        """
        gamma = self._risk_aversion
        sigma = volatility
        k = order_arrival_rate if order_arrival_rate is not None else self._order_arrival_rate

        # Volatility component: compensates for inventory risk
        vol_component = gamma * (sigma ** 2) * time_horizon

        # Order arrival component: (2/gamma) * ln(1 + gamma/k)
        # This term captures the trade-off between spread and fill probability
        # Wider spread = fewer fills but more profit per fill
        if gamma > 0 and k > 0:
            arrival_component = (2 / gamma) * np.log(1 + gamma / k)
        else:
            arrival_component = 0.0

        # Inventory adjustment - widen spread as inventory grows
        inventory_pct = abs(inventory) / self._max_inventory
        inventory_component = inventory_pct * self._min_spread_bps / 10000

        optimal_spread = vol_component + arrival_component + inventory_component

        # Apply minimum spread
        min_spread = self._min_spread_bps / 10000
        optimal_spread = max(optimal_spread, min_spread)

        return optimal_spread

    def calculate_quote_skew(self, inventory: int) -> float:
        """
        Calculate quote skew based on inventory.

        Positive skew = higher ask (want to sell)
        Negative skew = lower bid (want to buy)
        """
        inventory_pct = inventory / self._max_inventory
        max_skew = 0.0005  # 5 bps max skew

        skew = inventory_pct * max_skew

        return skew

    def calculate_inventory_skew_enhanced(
        self,
        inventory: int,
        fair_value: float
    ) -> tuple[float, float]:
        """
        P2: Enhanced inventory skew calculation.

        Uses non-linear skew that increases more aggressively
        as inventory approaches limits.

        Args:
            inventory: Current inventory position
            fair_value: Fair value price

        Returns:
            (skew_price, skew_bps)
            - skew_price: Absolute price adjustment
            - skew_bps: Skew in basis points
        """
        inventory_pct = inventory / self._max_inventory

        # Non-linear skew: more aggressive as inventory grows
        # Using sigmoid-like function for smooth transition
        if abs(inventory_pct) < 0.5:
            # Linear region for moderate inventory
            skew_factor = inventory_pct * self._inventory_skew_factor
        else:
            # Exponential region for high inventory
            sign = 1 if inventory_pct > 0 else -1
            abs_pct = abs(inventory_pct)
            # Accelerate skew beyond 50% inventory
            skew_factor = sign * (
                0.5 * self._inventory_skew_factor +
                (abs_pct - 0.5) * self._inventory_skew_factor * 2
            )

        # Cap at max skew
        skew_bps = max(-self._max_skew_bps, min(self._max_skew_bps, skew_factor))

        # Convert to price
        skew_price = fair_value * (skew_bps / 10000)

        return skew_price, skew_bps

    def detect_adverse_selection(
        self,
        trade_prices: np.ndarray,
        trade_sides: np.ndarray,
        mid_prices: np.ndarray
    ) -> tuple[float, bool]:
        """
        P2: Detect adverse selection / toxic order flow.

        Analyzes recent trades to detect informed flow patterns.
        Uses simplified VPIN-like approach.

        Args:
            trade_prices: Recent trade prices
            trade_sides: Trade sides (1 = buy, -1 = sell)
            mid_prices: Mid prices at time of trade

        Returns:
            (adverse_selection_risk, is_toxic)
            - adverse_selection_risk: 0-1 risk score
            - is_toxic: True if toxic flow detected
        """
        if len(trade_prices) < 5:
            return 0.0, False

        lookback = min(len(trade_prices), self._adverse_selection_lookback)

        recent_trades = trade_prices[-lookback:]
        recent_sides = trade_sides[-lookback:]
        recent_mids = mid_prices[-lookback:]

        # Metric 1: Order flow imbalance
        buy_volume = np.sum(recent_sides == 1)
        sell_volume = np.sum(recent_sides == -1)
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return 0.0, False

        flow_imbalance = abs(buy_volume - sell_volume) / total_volume

        # Metric 2: Price impact correlation
        # Informed traders tend to trade in direction of subsequent price moves
        if len(recent_trades) >= 3:
            price_changes = np.diff(recent_mids)
            trade_directions = recent_sides[:-1]  # Align with price changes

            if len(price_changes) > 0 and len(trade_directions) > 0:
                min_len = min(len(price_changes), len(trade_directions))
                correlation = np.corrcoef(
                    trade_directions[:min_len],
                    price_changes[:min_len]
                )[0, 1]

                if np.isfinite(correlation):
                    impact_score = abs(correlation)
                else:
                    impact_score = 0.0
            else:
                impact_score = 0.0
        else:
            impact_score = 0.0

        # Metric 3: Trade clustering (many trades on same side)
        # Count consecutive same-side trades
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(recent_sides)):
            if recent_sides[i] == recent_sides[i - 1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        cluster_score = min(1.0, max_consecutive / 5.0)  # Normalize: 5+ is max

        # Combined adverse selection risk
        adverse_selection_risk = (
            flow_imbalance * 0.4 +
            impact_score * 0.4 +
            cluster_score * 0.2
        )

        is_toxic = adverse_selection_risk > self._toxic_flow_threshold

        return adverse_selection_risk, is_toxic

    def calculate_volatility_adjustment(
        self,
        volatility: float,
        fair_value: float
    ) -> tuple[float, float]:
        """
        P2: Calculate quote spread adjustment for volatility.

        Higher volatility = wider spreads to compensate for
        increased inventory risk.

        Args:
            volatility: Current volatility estimate (annualized)
            fair_value: Fair value price

        Returns:
            (adjustment_price, adjustment_bps)
            - adjustment_price: Absolute spread adjustment (one side)
            - adjustment_bps: Adjustment in basis points
        """
        if volatility <= 0:
            return 0.0, 0.0

        # Calculate volatility ratio vs baseline
        vol_ratio = volatility / self._base_volatility

        if vol_ratio <= 1.0:
            # Normal or low volatility - no adjustment
            adjustment_bps = 0.0
        else:
            # High volatility - widen spread
            # Adjustment increases with vol^1.5 (concave function)
            excess_vol = vol_ratio - 1.0
            adjustment_bps = min(
                self._max_vol_adjustment_bps,
                excess_vol ** 1.5 * self._vol_spread_multiplier * 10  # Convert to bps
            )

        adjustment_price = fair_value * (adjustment_bps / 10000)

        return adjustment_price, adjustment_bps

    def generate_quotes(
        self,
        symbol: str,
        fair_value: float,
        volatility: float,
        inventory: int,
        bid_size: int = 100,
        ask_size: int = 100,
        trade_prices: np.ndarray | None = None,
        trade_sides: np.ndarray | None = None,
        mid_prices: np.ndarray | None = None,
    ) -> MarketMakingSignal:
        """
        Generate bid/ask quotes with P2 enhancements.

        Args:
            symbol: Trading symbol
            fair_value: Estimated fair value
            volatility: Current volatility estimate
            inventory: Current inventory position
            bid_size: Quote bid size
            ask_size: Quote ask size
            trade_prices: Recent trade prices (for adverse selection)
            trade_sides: Recent trade sides (for adverse selection)
            mid_prices: Recent mid prices (for adverse selection)
        """
        # Calculate optimal spread
        spread = self.calculate_optimal_spread(volatility, inventory)

        # P2: Calculate enhanced inventory skew
        skew_price, skew_bps = self.calculate_inventory_skew_enhanced(inventory, fair_value)

        # P2: Calculate volatility adjustment
        vol_adj_price, vol_adj_bps = self.calculate_volatility_adjustment(volatility, fair_value)

        # P2: Detect adverse selection if trade data available
        adverse_selection_risk = 0.0
        toxic_flow_detected = False
        if trade_prices is not None and trade_sides is not None and mid_prices is not None:
            adverse_selection_risk, toxic_flow_detected = self.detect_adverse_selection(
                trade_prices, trade_sides, mid_prices
            )

            # If toxic flow detected, widen spreads defensively
            if toxic_flow_detected:
                spread *= 1.5  # 50% wider spread
                logger.warning(f"{symbol}: Toxic flow detected, widening spreads")

        # Calculate bid and ask with all adjustments
        half_spread = spread / 2
        # Add volatility adjustment to spread
        half_spread += vol_adj_price

        bid_price = fair_value - half_spread - skew_price
        ask_price = fair_value + half_spread - skew_price

        # Round to tick size (assume 0.01)
        bid_price = round(bid_price, 2)
        ask_price = round(ask_price, 2)

        # Ensure positive spread
        if ask_price <= bid_price:
            ask_price = bid_price + 0.01

        quote = Quote(
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            mid_price=fair_value,
            spread=ask_price - bid_price,
        )

        # Determine if inventory management needed
        inventory_action = None
        urgency = 0.0

        if inventory > self._max_inventory * 0.8:
            inventory_action = "reduce_long"
            urgency = 0.8
        elif inventory < -self._max_inventory * 0.8:
            inventory_action = "reduce_short"
            urgency = 0.8

        # P2: Increase urgency if adverse selection risk is high
        if adverse_selection_risk > 0.5:
            urgency = max(urgency, adverse_selection_risk)

        # RISK-004: Detect market regime (simplified - would need price history)
        # For now, use volatility and adverse selection as proxy
        vol_ratio = volatility / self._base_volatility if self._base_volatility > 0 else 1.0

        if toxic_flow_detected:
            market_regime = "trending"
            regime_suitable = False
            regime_warning = "Toxic flow detected - possible trending market"
        elif vol_ratio > self._volatile_threshold:
            market_regime = "volatile"
            regime_suitable = False
            regime_warning = f"High volatility ({vol_ratio:.1f}x normal)"
        else:
            market_regime = "normal"
            regime_suitable = True
            regime_warning = None

        return MarketMakingSignal(
            symbol=symbol,
            action="quote",
            quote=quote,
            inventory_action=inventory_action,
            urgency=urgency,
            inventory_skew_bps=skew_bps,
            adverse_selection_risk=adverse_selection_risk,
            toxic_flow_detected=toxic_flow_detected,
            volatility_adjustment_bps=vol_adj_bps,
            inventory_stop_loss=int(self._max_inventory * (self._inventory_stop_loss_pct / 100)),
            pnl_stop_loss=self._pnl_stop_loss,
            max_position_duration_seconds=self._max_position_duration_seconds,
            strategy_max_loss_pct=self._strategy_max_loss_pct,
            daily_loss_limit=self._daily_loss_limit,
            market_regime=market_regime,
            regime_suitable=regime_suitable,
            regime_warning=regime_warning,
            is_exit_signal=False,
            exit_reason=None,
            min_profitable_spread_bps=self._min_profitable_spread_bps,
        )

    def detect_mm_regime(
        self,
        price_change_pct: float,
        volatility: float,
        volume_ratio: float
    ) -> tuple[str, bool, str | None]:
        """
        RISK-004: Detect market regime suitability for market making.

        Market making works best in:
        - Ranging/mean-reverting markets (NOT trending)
        - Normal volatility (NOT too high or low)
        - Adequate liquidity

        Args:
            price_change_pct: Recent price change as percentage
            volatility: Current volatility relative to baseline
            volume_ratio: Current volume / average volume

        Returns:
            (regime, is_suitable, warning)
        """
        vol_ratio = volatility / self._base_volatility if self._base_volatility > 0 else 1.0

        # Detect trending market
        if abs(price_change_pct) > self._trending_threshold:
            regime = "trending"
            suitable = False
            warning = f"Market trending ({price_change_pct:.1%}) - MM at risk of adverse selection"

        # Detect high volatility
        elif vol_ratio > self._volatile_threshold:
            regime = "volatile"
            suitable = False
            warning = f"High volatility ({vol_ratio:.1f}x normal) - widen spreads or pause"

        # Detect illiquidity
        elif volume_ratio < self._illiquidity_threshold:
            regime = "illiquid"
            suitable = False
            warning = f"Low liquidity ({volume_ratio:.1%} of normal) - wide spreads expected"

        else:
            regime = "normal"
            suitable = True
            warning = None

        return regime, suitable, warning

    def check_exit_conditions(
        self,
        inventory: int,
        current_pnl: float,
        daily_pnl: float,
        position_duration_seconds: int,
        adverse_selection_risk: float
    ) -> tuple[bool, str | None]:
        """
        RISK-005: Check if exit conditions are met for MM position.

        Args:
            inventory: Current inventory position
            current_pnl: Current position P&L
            daily_pnl: Daily strategy P&L
            position_duration_seconds: How long current inventory held
            adverse_selection_risk: Current adverse selection risk score

        Returns:
            (should_exit, reason)
        """
        # Check inventory stop-loss
        inventory_stop = int(self._max_inventory * (self._inventory_stop_loss_pct / 100))
        if abs(inventory) >= inventory_stop:
            return True, "inventory_stop_loss_exceeded"

        # Check P&L stop-loss
        if current_pnl <= self._pnl_stop_loss:
            return True, "pnl_stop_loss_triggered"

        # Check daily loss limit
        if daily_pnl <= self._daily_loss_limit:
            return True, "daily_loss_limit_reached"

        # Check max position duration
        if position_duration_seconds >= self._max_position_duration_seconds:
            return True, "max_position_duration_exceeded"

        # Check adverse selection - exit if too high
        if adverse_selection_risk > self._toxic_flow_threshold * 1.5:
            return True, "extreme_adverse_selection"

        return False, None

    def generate_exit_signal(
        self,
        symbol: str,
        exit_reason: str,
        inventory: int,
        fair_value: float
    ) -> MarketMakingSignal:
        """
        RISK-005: Generate exit signal for MM position.

        Args:
            symbol: Trading symbol
            exit_reason: Reason for exit
            inventory: Current inventory
            fair_value: Current fair value

        Returns:
            MarketMakingSignal with exit action
        """
        # Determine inventory action based on current position
        if inventory > 0:
            inventory_action = "reduce_long"
        elif inventory < 0:
            inventory_action = "reduce_short"
        else:
            inventory_action = None

        return MarketMakingSignal(
            symbol=symbol,
            action="exit",
            quote=None,
            inventory_action=inventory_action,
            urgency=1.0,  # High urgency for exits
            inventory_skew_bps=0.0,
            adverse_selection_risk=0.0,
            toxic_flow_detected=False,
            volatility_adjustment_bps=0.0,
            inventory_stop_loss=int(self._max_inventory * (self._inventory_stop_loss_pct / 100)),
            pnl_stop_loss=self._pnl_stop_loss,
            max_position_duration_seconds=self._max_position_duration_seconds,
            strategy_max_loss_pct=self._strategy_max_loss_pct,
            daily_loss_limit=self._daily_loss_limit,
            market_regime="unknown",
            regime_suitable=False,
            regime_warning=f"Exit: {exit_reason}",
            is_exit_signal=True,
            exit_reason=exit_reason,
            min_profitable_spread_bps=self._min_profitable_spread_bps,
        )

    def should_quote(
        self,
        current_spread: float,
        our_spread: float,
        inventory: int,
        adverse_selection_risk: float = 0.0,
        market_regime: str = "normal"
    ) -> bool:
        """
        Determine if we should be quoting.

        Don't quote if:
        - Spread is too tight (can't make money)
        - Inventory is at limit
        - Market is too fast (adverse selection)
        - Adverse selection risk is too high (P2)
        - Market regime is unfavorable (RISK-004)
        """
        # RISK-004: Check market regime
        if market_regime in ["trending", "volatile"]:
            logger.info(f"Not quoting due to {market_regime} market regime")
            return False

        # Check if spread is profitable
        min_spread = self._min_profitable_spread_bps / 10000
        if current_spread < min_spread:
            logger.debug(f"Spread {current_spread:.4f} below minimum {min_spread:.4f}")
            return False

        if current_spread < our_spread * 0.8:
            return False  # Can't compete

        # Check inventory limits
        if abs(inventory) >= self._max_inventory:
            return False

        # P2: Check adverse selection risk
        if adverse_selection_risk > self._toxic_flow_threshold:
            logger.info(
                f"Not quoting due to high adverse selection risk: {adverse_selection_risk:.2f}"
            )
            return False

        return True

    def analyze_trade_flow(
        self,
        trade_prices: np.ndarray,
        trade_sides: np.ndarray,
        trade_sizes: np.ndarray,
    ) -> dict[str, float]:
        """
        P2: Analyze trade flow for market making decisions.

        Returns metrics about recent trading activity that can
        inform quote placement and risk management.

        Args:
            trade_prices: Recent trade prices
            trade_sides: Trade sides (1 = buy, -1 = sell)
            trade_sizes: Trade sizes

        Returns:
            Dictionary of flow metrics
        """
        if len(trade_prices) < 3:
            return {
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "net_flow": 0.0,
                "vwap": 0.0,
                "flow_imbalance": 0.0,
            }

        buy_mask = trade_sides == 1
        sell_mask = trade_sides == -1

        buy_volume = np.sum(trade_sizes[buy_mask])
        sell_volume = np.sum(trade_sizes[sell_mask])
        total_volume = buy_volume + sell_volume

        net_flow = buy_volume - sell_volume

        if total_volume > 0:
            vwap = np.sum(trade_prices * trade_sizes) / total_volume
            flow_imbalance = net_flow / total_volume
        else:
            vwap = trade_prices[-1] if len(trade_prices) > 0 else 0.0
            flow_imbalance = 0.0

        return {
            "buy_volume": float(buy_volume),
            "sell_volume": float(sell_volume),
            "net_flow": float(net_flow),
            "vwap": float(vwap),
            "flow_imbalance": float(flow_imbalance),
        }

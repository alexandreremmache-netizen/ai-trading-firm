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
from dataclasses import dataclass
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


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
    action: str  # "quote", "cancel", "adjust"
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
        )

    def should_quote(
        self,
        current_spread: float,
        our_spread: float,
        inventory: int,
        adverse_selection_risk: float = 0.0,
    ) -> bool:
        """
        Determine if we should be quoting.

        Don't quote if:
        - Spread is too tight (can't make money)
        - Inventory is at limit
        - Market is too fast (adverse selection)
        - Adverse selection risk is too high (P2)
        """
        # Check if spread is profitable
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

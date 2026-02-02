"""
Portfolio Construction Module
=============================

Target portfolio construction (Issue #P14).
Trade list generation (Issue #P15).
Portfolio comparison tools (Issue #P18).

Features:
- Target weight calculation
- Trade list with rebalancing
- Portfolio comparison metrics
- Transaction cost optimization
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RebalanceMethod(str, Enum):
    """Rebalancing methodology."""
    FULL = "full"  # Rebalance all positions
    THRESHOLD = "threshold"  # Only if outside threshold
    TAX_AWARE = "tax_aware"  # Minimize tax impact
    COST_AWARE = "cost_aware"  # Minimize transaction costs


class TradeReason(str, Enum):
    """Reason for generating a trade."""
    REBALANCE = "rebalance"
    NEW_TARGET = "new_target"
    EXIT_POSITION = "exit_position"
    RISK_REDUCTION = "risk_reduction"
    CASH_RAISE = "cash_raise"
    DIVIDEND_REINVEST = "dividend_reinvest"


@dataclass
class TargetPosition:
    """Target position in a portfolio."""
    symbol: str
    target_weight: float  # Target as % of portfolio (0-1)
    target_shares: int = 0
    target_value: float = 0.0

    # Constraints
    min_weight: float = 0.0
    max_weight: float = 1.0

    # Current state
    current_weight: float = 0.0
    current_shares: int = 0
    current_value: float = 0.0

    # Metadata
    sector: str = ""
    asset_class: str = ""

    @property
    def weight_difference(self) -> float:
        """Difference from target."""
        return self.target_weight - self.current_weight

    @property
    def is_overweight(self) -> bool:
        return self.current_weight > self.target_weight

    @property
    def is_underweight(self) -> bool:
        return self.current_weight < self.target_weight

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'target_weight': self.target_weight,
            'target_shares': self.target_shares,
            'target_value': self.target_value,
            'current_weight': self.current_weight,
            'current_shares': self.current_shares,
            'current_value': self.current_value,
            'weight_difference': self.weight_difference,
            'sector': self.sector,
            'asset_class': self.asset_class,
        }


@dataclass
class Trade:
    """Individual trade in a trade list."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    estimated_price: float

    # Trade details
    reason: TradeReason
    urgency: str = "normal"  # 'immediate', 'normal', 'patient'

    # Costs
    estimated_commission: float = 0.0
    estimated_impact: float = 0.0
    total_cost: float = 0.0

    # Tax
    estimated_gain_loss: float = 0.0
    is_short_term: bool = False
    estimated_tax: float = 0.0

    # Order details
    order_type: str = "LIMIT"
    limit_price: float | None = None

    # Metadata
    priority: int = 0  # Lower is higher priority

    @property
    def notional(self) -> float:
        return self.quantity * self.estimated_price

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'estimated_price': self.estimated_price,
            'notional': self.notional,
            'reason': self.reason.value,
            'urgency': self.urgency,
            'estimated_commission': self.estimated_commission,
            'estimated_impact': self.estimated_impact,
            'total_cost': self.total_cost,
            'order_type': self.order_type,
            'limit_price': self.limit_price,
            'priority': self.priority,
        }


@dataclass
class TradeList:
    """Complete trade list for rebalancing."""
    trades: list[Trade]
    generated_at: datetime

    # Summary
    total_buys: int = 0
    total_sells: int = 0
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    net_cash_flow: float = 0.0

    # Costs
    total_commission: float = 0.0
    total_impact: float = 0.0
    total_estimated_tax: float = 0.0

    # Method used
    rebalance_method: RebalanceMethod = RebalanceMethod.FULL

    def to_dict(self) -> dict:
        return {
            'generated_at': self.generated_at.isoformat(),
            'num_trades': len(self.trades),
            'total_buys': self.total_buys,
            'total_sells': self.total_sells,
            'total_buy_value': self.total_buy_value,
            'total_sell_value': self.total_sell_value,
            'net_cash_flow': self.net_cash_flow,
            'total_commission': self.total_commission,
            'total_impact': self.total_impact,
            'total_estimated_tax': self.total_estimated_tax,
            'rebalance_method': self.rebalance_method.value,
            'trades': [t.to_dict() for t in self.trades],
        }


@dataclass
class PortfolioComparison:
    """Comparison between two portfolios (#P18)."""
    portfolio1_name: str
    portfolio2_name: str
    comparison_date: datetime

    # Position differences
    positions_only_in_1: list[str]
    positions_only_in_2: list[str]
    common_positions: list[str]

    # Weight differences
    weight_differences: dict[str, float]  # symbol -> (p2_weight - p1_weight)
    max_weight_diff: float
    avg_weight_diff: float

    # Risk comparison
    var_diff: float | None = None
    volatility_diff: float | None = None
    beta_diff: float | None = None

    # Sector allocation differences
    sector_diffs: dict[str, float] = field(default_factory=dict)

    # Similarity score
    similarity_score: float = 0.0  # 0-1, higher is more similar

    def to_dict(self) -> dict:
        return {
            'portfolio1_name': self.portfolio1_name,
            'portfolio2_name': self.portfolio2_name,
            'comparison_date': self.comparison_date.isoformat(),
            'positions_only_in_1': self.positions_only_in_1,
            'positions_only_in_2': self.positions_only_in_2,
            'common_positions': self.common_positions,
            'num_differences': len(self.positions_only_in_1) + len(self.positions_only_in_2),
            'max_weight_diff': self.max_weight_diff,
            'avg_weight_diff': self.avg_weight_diff,
            'sector_diffs': self.sector_diffs,
            'similarity_score': self.similarity_score,
        }


class TargetPortfolioBuilder:
    """
    Builds target portfolios (#P14).

    Supports multiple construction methodologies.
    """

    def __init__(
        self,
        min_position_weight: float = 0.01,  # 1% minimum
        max_position_weight: float = 0.20,  # 20% maximum
        round_lots: bool = True,
        lot_size: int = 100,
    ):
        self.min_weight = min_position_weight
        self.max_weight = max_position_weight
        self.round_lots = round_lots
        self.lot_size = lot_size

        # Current prices
        self._prices: dict[str, float] = {}

        # Current holdings
        self._holdings: dict[str, int] = {}

        # Sector assignments
        self._sectors: dict[str, str] = {}

    def set_price(self, symbol: str, price: float) -> None:
        """Set price for a symbol."""
        self._prices[symbol] = price

    def set_holding(self, symbol: str, quantity: int) -> None:
        """Set current holding."""
        self._holdings[symbol] = quantity

    def set_sector(self, symbol: str, sector: str) -> None:
        """Set sector for a symbol."""
        self._sectors[symbol] = sector

    def build_equal_weight(
        self,
        symbols: list[str],
        portfolio_value: float,
    ) -> list[TargetPosition]:
        """Build equal-weight portfolio."""
        if not symbols:
            return []

        weight = 1.0 / len(symbols)
        weight = max(self.min_weight, min(self.max_weight, weight))

        targets = []
        for symbol in symbols:
            price = self._prices.get(symbol, 0.0)
            if price <= 0:
                continue

            target_value = portfolio_value * weight
            target_shares = self._calculate_shares(target_value, price)

            current_shares = self._holdings.get(symbol, 0)
            current_value = current_shares * price
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0

            targets.append(TargetPosition(
                symbol=symbol,
                target_weight=weight,
                target_shares=target_shares,
                target_value=target_shares * price,
                current_weight=current_weight,
                current_shares=current_shares,
                current_value=current_value,
                sector=self._sectors.get(symbol, ""),
            ))

        return targets

    def build_market_cap_weight(
        self,
        symbols_with_mcap: dict[str, float],
        portfolio_value: float,
    ) -> list[TargetPosition]:
        """Build market-cap weighted portfolio."""
        if not symbols_with_mcap:
            return []

        total_mcap = sum(symbols_with_mcap.values())
        if total_mcap <= 0:
            return []

        targets = []
        for symbol, mcap in symbols_with_mcap.items():
            price = self._prices.get(symbol, 0.0)
            if price <= 0:
                continue

            raw_weight = mcap / total_mcap
            weight = max(self.min_weight, min(self.max_weight, raw_weight))

            target_value = portfolio_value * weight
            target_shares = self._calculate_shares(target_value, price)

            current_shares = self._holdings.get(symbol, 0)
            current_value = current_shares * price
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0

            targets.append(TargetPosition(
                symbol=symbol,
                target_weight=weight,
                target_shares=target_shares,
                target_value=target_shares * price,
                current_weight=current_weight,
                current_shares=current_shares,
                current_value=current_value,
                sector=self._sectors.get(symbol, ""),
            ))

        return targets

    def build_custom_weight(
        self,
        target_weights: dict[str, float],
        portfolio_value: float,
    ) -> list[TargetPosition]:
        """Build portfolio with custom weights."""
        # Normalize weights
        total_weight = sum(target_weights.values())
        if total_weight <= 0:
            return []

        # First pass: calculate weights and clamp
        clamped_weights = {}
        for symbol, raw_weight in target_weights.items():
            price = self._prices.get(symbol, 0.0)
            if price <= 0:
                continue

            weight = raw_weight / total_weight
            weight = max(self.min_weight, min(self.max_weight, weight))
            clamped_weights[symbol] = weight

        # Re-normalize after clamping to ensure weights sum to 1.0
        clamped_total = sum(clamped_weights.values())
        if clamped_total > 0:
            clamped_weights = {s: w / clamped_total for s, w in clamped_weights.items()}

        # Second pass: build target positions with normalized weights
        targets = []
        for symbol, weight in clamped_weights.items():
            price = self._prices.get(symbol, 0.0)
            # price > 0 already checked in first pass

            target_value = portfolio_value * weight
            target_shares = self._calculate_shares(target_value, price)

            current_shares = self._holdings.get(symbol, 0)
            current_value = current_shares * price
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0

            targets.append(TargetPosition(
                symbol=symbol,
                target_weight=weight,
                target_shares=target_shares,
                target_value=target_shares * price,
                current_weight=current_weight,
                current_shares=current_shares,
                current_value=current_value,
                sector=self._sectors.get(symbol, ""),
            ))

        return targets

    def build_risk_parity(
        self,
        symbols_with_vol: dict[str, float],
        portfolio_value: float,
    ) -> list[TargetPosition]:
        """Build risk parity portfolio (equal risk contribution)."""
        if not symbols_with_vol:
            return []

        # Inverse volatility weighting
        total_inv_vol = sum(1.0 / v for v in symbols_with_vol.values() if v > 0)
        if total_inv_vol <= 0:
            return []

        targets = []
        for symbol, vol in symbols_with_vol.items():
            if vol <= 0:
                continue

            price = self._prices.get(symbol, 0.0)
            if price <= 0:
                continue

            raw_weight = (1.0 / vol) / total_inv_vol
            weight = max(self.min_weight, min(self.max_weight, raw_weight))

            target_value = portfolio_value * weight
            target_shares = self._calculate_shares(target_value, price)

            current_shares = self._holdings.get(symbol, 0)
            current_value = current_shares * price
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0

            targets.append(TargetPosition(
                symbol=symbol,
                target_weight=weight,
                target_shares=target_shares,
                target_value=target_shares * price,
                current_weight=current_weight,
                current_shares=current_shares,
                current_value=current_value,
                sector=self._sectors.get(symbol, ""),
            ))

        return targets

    def _calculate_shares(self, target_value: float, price: float) -> int:
        """Calculate target shares with optional rounding."""
        if price <= 0:
            return 0

        raw_shares = target_value / price

        if self.round_lots:
            return int(raw_shares // self.lot_size) * self.lot_size
        else:
            return int(raw_shares)


class TradeListGenerator:
    """
    Generates trade lists from target portfolios (#P15).

    Supports multiple rebalancing methods.
    """

    def __init__(
        self,
        commission_per_share: float = 0.005,
        commission_min: float = 1.0,
        impact_rate: float = 0.0005,  # 5 bps market impact
        short_term_tax_rate: float = 0.35,
        long_term_tax_rate: float = 0.15,
    ):
        self.commission_per_share = commission_per_share
        self.commission_min = commission_min
        self.impact_rate = impact_rate
        self.short_term_rate = short_term_tax_rate
        self.long_term_rate = long_term_tax_rate

        # Cost basis for tax calculations
        self._cost_basis: dict[str, float] = {}  # symbol -> average cost
        self._holding_periods: dict[str, bool] = {}  # symbol -> is_short_term

    def set_cost_basis(self, symbol: str, cost_per_share: float, is_short_term: bool = True) -> None:
        """Set cost basis for tax calculations."""
        self._cost_basis[symbol] = cost_per_share
        self._holding_periods[symbol] = is_short_term

    def generate_full_rebalance(
        self,
        targets: list[TargetPosition],
        available_cash: float = 0.0,
    ) -> TradeList:
        """Generate trade list for full rebalance."""
        trades = []

        for target in targets:
            shares_diff = target.target_shares - target.current_shares

            if shares_diff == 0:
                continue

            side = "BUY" if shares_diff > 0 else "SELL"
            quantity = abs(shares_diff)
            price = target.target_value / target.target_shares if target.target_shares > 0 else 0

            trade = self._create_trade(
                symbol=target.symbol,
                side=side,
                quantity=quantity,
                price=price,
                reason=TradeReason.REBALANCE,
            )
            trades.append(trade)

        return self._finalize_trade_list(trades, RebalanceMethod.FULL)

    def generate_threshold_rebalance(
        self,
        targets: list[TargetPosition],
        threshold: float = 0.05,  # 5% threshold
    ) -> TradeList:
        """Generate trades only for positions outside threshold."""
        trades = []

        for target in targets:
            weight_diff = abs(target.weight_difference)

            if weight_diff < threshold:
                continue

            shares_diff = target.target_shares - target.current_shares

            if shares_diff == 0:
                continue

            side = "BUY" if shares_diff > 0 else "SELL"
            quantity = abs(shares_diff)
            price = target.target_value / target.target_shares if target.target_shares > 0 else 0

            # Urgency based on how far outside threshold
            urgency = "immediate" if weight_diff > threshold * 2 else "normal"

            trade = self._create_trade(
                symbol=target.symbol,
                side=side,
                quantity=quantity,
                price=price,
                reason=TradeReason.REBALANCE,
                urgency=urgency,
            )
            trades.append(trade)

        return self._finalize_trade_list(trades, RebalanceMethod.THRESHOLD)

    def generate_tax_aware_rebalance(
        self,
        targets: list[TargetPosition],
        max_tax: float = 10000.0,
    ) -> TradeList:
        """Generate trades minimizing tax impact."""
        trades = []
        cumulative_tax = 0.0

        # Sort: buys first, then sells by gain (losses first)
        # weight_difference = target_weight - current_weight
        # Positive weight_difference = underweight = need to BUY
        # Negative weight_difference = overweight = need to SELL
        sells = [t for t in targets if t.weight_difference < 0]  # overweight, sell to reduce
        buys = [t for t in targets if t.weight_difference > 0]   # underweight, buy to increase

        # Sort sells by gain (realize losses first)
        sells_with_gain = []
        for target in sells:
            # Defensive check for zero shares
            if target.current_shares > 0:
                cost_basis = self._cost_basis.get(target.symbol, target.current_value / target.current_shares)
                current_price = target.current_value / target.current_shares
            else:
                cost_basis = self._cost_basis.get(target.symbol, 0)
                current_price = 0
            gain_per_share = current_price - cost_basis
            sells_with_gain.append((target, gain_per_share))

        sells_with_gain.sort(key=lambda x: x[1])  # Losses first

        # Generate sells
        for target, gain_per_share in sells_with_gain:
            shares_diff = target.current_shares - target.target_shares
            if shares_diff <= 0:
                continue

            # Calculate tax
            is_short_term = self._holding_periods.get(target.symbol, True)
            tax_rate = self.short_term_rate if is_short_term else self.long_term_rate
            estimated_tax = max(0, gain_per_share * shares_diff * tax_rate)

            # Check if we can afford this tax
            if cumulative_tax + estimated_tax > max_tax and gain_per_share > 0:
                # Skip this trade to stay under tax limit
                continue

            cumulative_tax += estimated_tax

            price = target.current_value / target.current_shares if target.current_shares > 0 else 0

            trade = self._create_trade(
                symbol=target.symbol,
                side="SELL",
                quantity=shares_diff,
                price=price,
                reason=TradeReason.REBALANCE,
            )
            trade.estimated_gain_loss = gain_per_share * shares_diff
            trade.is_short_term = is_short_term
            trade.estimated_tax = estimated_tax

            trades.append(trade)

        # Generate buys
        for target in buys:
            shares_diff = target.target_shares - target.current_shares
            if shares_diff <= 0:
                continue

            price = target.target_value / target.target_shares if target.target_shares > 0 else 0

            trade = self._create_trade(
                symbol=target.symbol,
                side="BUY",
                quantity=shares_diff,
                price=price,
                reason=TradeReason.REBALANCE,
            )
            trades.append(trade)

        return self._finalize_trade_list(trades, RebalanceMethod.TAX_AWARE)

    def _create_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        reason: TradeReason,
        urgency: str = "normal",
    ) -> Trade:
        """Create a trade with cost estimates."""
        notional = quantity * price

        # Commission
        commission = max(self.commission_min, quantity * self.commission_per_share)

        # Market impact
        impact = notional * self.impact_rate

        return Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            estimated_price=price,
            reason=reason,
            urgency=urgency,
            estimated_commission=commission,
            estimated_impact=impact,
            total_cost=commission + impact,
            order_type="LIMIT",
            limit_price=price,
        )

    def _finalize_trade_list(
        self,
        trades: list[Trade],
        method: RebalanceMethod,
    ) -> TradeList:
        """Finalize trade list with summary calculations."""
        buys = [t for t in trades if t.side == "BUY"]
        sells = [t for t in trades if t.side == "SELL"]

        total_buy_value = sum(t.notional for t in buys)
        total_sell_value = sum(t.notional for t in sells)

        return TradeList(
            trades=trades,
            generated_at=datetime.now(timezone.utc),
            total_buys=len(buys),
            total_sells=len(sells),
            total_buy_value=total_buy_value,
            total_sell_value=total_sell_value,
            net_cash_flow=total_sell_value - total_buy_value,
            total_commission=sum(t.estimated_commission for t in trades),
            total_impact=sum(t.estimated_impact for t in trades),
            total_estimated_tax=sum(t.estimated_tax for t in trades),
            rebalance_method=method,
        )


class PortfolioComparator:
    """
    Compares portfolios (#P18).

    Provides detailed comparison metrics.
    """

    def compare(
        self,
        portfolio1: dict[str, float],  # symbol -> weight
        portfolio2: dict[str, float],
        name1: str = "Portfolio 1",
        name2: str = "Portfolio 2",
        sectors: dict[str, str] | None = None,
    ) -> PortfolioComparison:
        """
        Compare two portfolios by weight.

        Args:
            portfolio1: First portfolio weights
            portfolio2: Second portfolio weights
            name1: Name for first portfolio
            name2: Name for second portfolio
            sectors: Optional sector assignments

        Returns:
            PortfolioComparison with detailed metrics
        """
        symbols1 = set(portfolio1.keys())
        symbols2 = set(portfolio2.keys())

        only_in_1 = list(symbols1 - symbols2)
        only_in_2 = list(symbols2 - symbols1)
        common = list(symbols1 & symbols2)

        # Weight differences for common positions
        weight_diffs = {
            s: portfolio2.get(s, 0) - portfolio1.get(s, 0)
            for s in symbols1 | symbols2
        }

        abs_diffs = [abs(d) for d in weight_diffs.values()]
        max_diff = max(abs_diffs) if abs_diffs else 0
        avg_diff = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0

        # Sector differences
        sector_diffs = {}
        if sectors:
            sectors_in_1: dict[str, float] = {}
            sectors_in_2: dict[str, float] = {}

            for s, w in portfolio1.items():
                sector = sectors.get(s, "Unknown")
                sectors_in_1[sector] = sectors_in_1.get(sector, 0) + w

            for s, w in portfolio2.items():
                sector = sectors.get(s, "Unknown")
                sectors_in_2[sector] = sectors_in_2.get(sector, 0) + w

            all_sectors = set(sectors_in_1.keys()) | set(sectors_in_2.keys())
            sector_diffs = {
                s: sectors_in_2.get(s, 0) - sectors_in_1.get(s, 0)
                for s in all_sectors
            }

        # Similarity score (1 - normalized sum of absolute differences)
        total_diff = sum(abs(d) for d in weight_diffs.values())
        # Max possible difference is 2 (if completely different)
        similarity = max(0, 1 - total_diff / 2)

        return PortfolioComparison(
            portfolio1_name=name1,
            portfolio2_name=name2,
            comparison_date=datetime.now(timezone.utc),
            positions_only_in_1=only_in_1,
            positions_only_in_2=only_in_2,
            common_positions=common,
            weight_differences=weight_diffs,
            max_weight_diff=max_diff,
            avg_weight_diff=avg_diff,
            sector_diffs=sector_diffs,
            similarity_score=similarity,
        )

    def track_drift(
        self,
        original: dict[str, float],
        current: dict[str, float],
        name: str = "Portfolio",
    ) -> dict:
        """
        Track portfolio drift from original weights.

        Returns drift metrics and rebalance recommendations.
        """
        comparison = self.compare(original, current, f"{name} Original", f"{name} Current")

        # Calculate drift metrics
        symbols_drifted = [
            s for s, d in comparison.weight_differences.items()
            if abs(d) > 0.02  # >2% drift
        ]

        significant_drift = [
            s for s, d in comparison.weight_differences.items()
            if abs(d) > 0.05  # >5% drift
        ]

        # Turnover needed
        turnover = sum(abs(d) for d in comparison.weight_differences.values()) / 2

        return {
            'comparison': comparison.to_dict(),
            'num_positions_drifted': len(symbols_drifted),
            'num_significant_drift': len(significant_drift),
            'symbols_with_significant_drift': significant_drift,
            'implied_turnover': turnover,
            'needs_rebalance': len(significant_drift) > 0 or turnover > 0.1,
            'rebalance_urgency': 'high' if turnover > 0.2 else 'normal' if turnover > 0.1 else 'low',
        }

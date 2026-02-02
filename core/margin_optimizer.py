"""
Margin Optimization Module
==========================

Cross-margin benefit calculation and optimization (Issue #R14).
Risk contribution attribution by strategy (Issue #R16).

Features:
- Portfolio margin vs Reg-T comparison
- Cross-margin benefits for hedged positions
- Risk contribution by strategy
- Margin efficiency optimization
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MarginType(str, Enum):
    """Margin calculation methodology."""
    REG_T = "reg_t"  # Standard 50% initial
    PORTFOLIO_MARGIN = "portfolio_margin"  # TIMS/SPAN based
    SPAN = "span"  # Futures/options
    CROSS_MARGIN = "cross_margin"  # Cross-product netting


@dataclass
class PositionMargin:
    """Margin requirement for a single position."""
    symbol: str
    quantity: int
    market_value: float

    # Margin components
    reg_t_margin: float
    portfolio_margin: float
    span_margin: float | None = None

    # Offsets
    hedging_offset: float = 0.0
    correlation_offset: float = 0.0

    # Final
    effective_margin: float = 0.0
    margin_type_used: MarginType = MarginType.REG_T

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'market_value': self.market_value,
            'reg_t_margin': self.reg_t_margin,
            'portfolio_margin': self.portfolio_margin,
            'span_margin': self.span_margin,
            'hedging_offset': self.hedging_offset,
            'correlation_offset': self.correlation_offset,
            'effective_margin': self.effective_margin,
            'margin_type_used': self.margin_type_used.value,
        }


@dataclass
class CrossMarginBenefit:
    """Cross-margin benefit analysis."""
    position1_symbol: str
    position2_symbol: str

    # Individual margins
    position1_standalone_margin: float
    position2_standalone_margin: float
    combined_standalone: float

    # Cross-margined
    combined_cross_margin: float

    # Benefit
    margin_savings: float
    savings_pct: float

    # Reason
    correlation: float
    hedge_ratio: float
    benefit_type: str  # 'correlation', 'hedge', 'spread'

    def to_dict(self) -> dict:
        return {
            'position1_symbol': self.position1_symbol,
            'position2_symbol': self.position2_symbol,
            'position1_standalone_margin': self.position1_standalone_margin,
            'position2_standalone_margin': self.position2_standalone_margin,
            'combined_standalone': self.combined_standalone,
            'combined_cross_margin': self.combined_cross_margin,
            'margin_savings': self.margin_savings,
            'savings_pct': self.savings_pct,
            'correlation': self.correlation,
            'hedge_ratio': self.hedge_ratio,
            'benefit_type': self.benefit_type,
        }


@dataclass
class StrategyRiskContribution:
    """Risk contribution by strategy (#R16)."""
    strategy_name: str

    # Position data
    positions: list[str]
    gross_exposure: float
    net_exposure: float

    # Risk metrics
    var_contribution: float  # Dollar VaR contribution
    var_contribution_pct: float  # % of total portfolio VaR
    marginal_var: float  # Additional VaR from this strategy

    # Margin
    margin_required: float
    margin_pct_of_portfolio: float

    # Performance context
    sharpe_ratio: float | None = None
    return_on_margin: float | None = None  # Return / margin required

    # Risk efficiency
    risk_adjusted_return: float | None = None

    def to_dict(self) -> dict:
        return {
            'strategy_name': self.strategy_name,
            'positions': self.positions,
            'gross_exposure': self.gross_exposure,
            'net_exposure': self.net_exposure,
            'var_contribution': self.var_contribution,
            'var_contribution_pct': self.var_contribution_pct,
            'marginal_var': self.marginal_var,
            'margin_required': self.margin_required,
            'margin_pct_of_portfolio': self.margin_pct_of_portfolio,
            'sharpe_ratio': self.sharpe_ratio,
            'return_on_margin': self.return_on_margin,
            'risk_adjusted_return': self.risk_adjusted_return,
        }


class CrossMarginCalculator:
    """
    Calculates cross-margin benefits (#R14).

    Identifies margin savings from hedged positions and correlated assets.
    """

    # Reg-T margin rates by security type
    REG_T_RATES = {
        'equity': 0.50,  # 50% initial margin
        'etf': 0.50,
        'equity_option': 0.20,  # Varies by strategy
        'index_future': 0.05,  # Depends on exchange
        'commodity_future': 0.08,
        'fx': 0.02,  # Retail forex
    }

    # Portfolio margin risk percentages (simplified TIMS)
    PM_RISK_RATES = {
        'equity': 0.15,  # 15% for broad market
        'etf': 0.12,
        'equity_option': 0.15,  # Net position
        'index_future': 0.04,
        'commodity_future': 0.06,
        'fx': 0.02,
    }

    def __init__(
        self,
        correlation_threshold: float = -0.5,  # Minimum negative correlation for hedge
        hedge_ratio_tolerance: float = 0.3,  # Max deviation from 1:1
    ):
        self.correlation_threshold = correlation_threshold
        self.hedge_ratio_tolerance = hedge_ratio_tolerance

        # Correlation matrix (symbol pairs)
        self._correlations: dict[tuple[str, str], float] = {}

        # Asset types
        self._asset_types: dict[str, str] = {}

        # Position data
        self._positions: dict[str, dict] = {}

        # Beta values for hedge ratio
        self._betas: dict[str, float] = {}

    def set_correlation(self, symbol1: str, symbol2: str, correlation: float) -> None:
        """Set correlation between two symbols."""
        key = tuple(sorted([symbol1, symbol2]))
        self._correlations[key] = correlation

    def set_asset_type(self, symbol: str, asset_type: str) -> None:
        """Set asset type for a symbol."""
        self._asset_types[symbol] = asset_type

    def set_beta(self, symbol: str, beta: float) -> None:
        """Set beta for hedge ratio calculation."""
        self._betas[symbol] = beta

    def update_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        asset_type: str | None = None,
    ) -> None:
        """Update position data."""
        if asset_type:
            self._asset_types[symbol] = asset_type

        self._positions[symbol] = {
            'quantity': quantity,
            'price': price,
            'market_value': quantity * price,
            'asset_type': self._asset_types.get(symbol, 'equity'),
        }

    def calculate_position_margins(self) -> dict[str, PositionMargin]:
        """Calculate margin for all positions."""
        results = {}

        for symbol, pos in self._positions.items():
            asset_type = pos['asset_type']
            market_value = abs(pos['market_value'])

            # Reg-T margin
            reg_t_rate = self.REG_T_RATES.get(asset_type, 0.50)
            reg_t_margin = market_value * reg_t_rate

            # Portfolio margin
            pm_rate = self.PM_RISK_RATES.get(asset_type, 0.15)
            portfolio_margin = market_value * pm_rate

            results[symbol] = PositionMargin(
                symbol=symbol,
                quantity=pos['quantity'],
                market_value=pos['market_value'],
                reg_t_margin=reg_t_margin,
                portfolio_margin=portfolio_margin,
                effective_margin=min(reg_t_margin, portfolio_margin),
                margin_type_used=MarginType.PORTFOLIO_MARGIN if portfolio_margin < reg_t_margin else MarginType.REG_T,
            )

        return results

    def find_cross_margin_benefits(self) -> list[CrossMarginBenefit]:
        """
        Find all cross-margin benefits in the portfolio.

        Analyzes pairs of positions for hedging/correlation benefits.
        """
        benefits = []
        symbols = list(self._positions.keys())

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                benefit = self._analyze_pair_benefit(sym1, sym2)
                if benefit and benefit.margin_savings > 0:
                    benefits.append(benefit)

        # Sort by savings
        benefits.sort(key=lambda x: x.margin_savings, reverse=True)
        return benefits

    def _analyze_pair_benefit(
        self,
        symbol1: str,
        symbol2: str,
    ) -> CrossMarginBenefit | None:
        """Analyze cross-margin benefit for a pair."""
        pos1 = self._positions.get(symbol1)
        pos2 = self._positions.get(symbol2)

        if not pos1 or not pos2:
            return None

        # Get correlation
        key = tuple(sorted([symbol1, symbol2]))
        correlation = self._correlations.get(key, 0.0)

        # Get standalone margins
        type1 = pos1['asset_type']
        type2 = pos2['asset_type']

        mv1 = abs(pos1['market_value'])
        mv2 = abs(pos2['market_value'])

        pm_rate1 = self.PM_RISK_RATES.get(type1, 0.15)
        pm_rate2 = self.PM_RISK_RATES.get(type2, 0.15)

        standalone1 = mv1 * pm_rate1
        standalone2 = mv2 * pm_rate2
        combined_standalone = standalone1 + standalone2

        # Calculate combined margin with correlation benefit
        # Variance reduction: Var(A+B) = Var(A) + Var(B) + 2*Cov(A,B)
        # For opposing positions with negative correlation, this reduces

        # Check if positions are offsetting
        same_direction = (pos1['quantity'] > 0) == (pos2['quantity'] > 0)

        if not same_direction and correlation > 0:
            # Offsetting positions with positive correlation = hedge
            benefit_type = "hedge"
            # Hedge benefit proportional to smaller position
            hedge_ratio = min(mv1, mv2) / max(mv1, mv2) if max(mv1, mv2) > 0 else 0

            # Margin reduction from hedging
            hedged_portion = min(mv1, mv2)
            unhedged_portion = abs(mv1 - mv2)

            # Hedged portion gets reduced margin based on correlation
            hedge_margin_rate = max(0.02, pm_rate1 * (1 - correlation * 0.8))
            combined_cross = (
                hedged_portion * hedge_margin_rate +
                unhedged_portion * max(pm_rate1, pm_rate2)
            )

        elif same_direction and correlation < self.correlation_threshold:
            # Same direction but negatively correlated = diversification
            benefit_type = "diversification"
            hedge_ratio = 0

            # Diversification benefit - guard against division by zero
            if (mv1 + mv2) > 0:
                vol_reduction = math.sqrt(1 + 2 * correlation * (mv1 * mv2) / (mv1 + mv2)**2)
                combined_cross = combined_standalone * vol_reduction
            else:
                combined_cross = combined_standalone

        elif not same_direction and correlation < 0:
            # Offsetting with negative correlation = enhanced hedge
            benefit_type = "enhanced_hedge"
            hedge_ratio = min(mv1, mv2) / max(mv1, mv2) if max(mv1, mv2) > 0 else 0

            hedged_portion = min(mv1, mv2)
            unhedged_portion = abs(mv1 - mv2)

            # Enhanced reduction for negative correlation
            hedge_margin_rate = max(0.01, pm_rate1 * (1 - abs(correlation)))
            combined_cross = (
                hedged_portion * hedge_margin_rate +
                unhedged_portion * max(pm_rate1, pm_rate2)
            )

        else:
            # No significant benefit
            return None

        savings = combined_standalone - combined_cross
        savings_pct = savings / combined_standalone * 100 if combined_standalone > 0 else 0

        if savings <= 0:
            return None

        return CrossMarginBenefit(
            position1_symbol=symbol1,
            position2_symbol=symbol2,
            position1_standalone_margin=standalone1,
            position2_standalone_margin=standalone2,
            combined_standalone=combined_standalone,
            combined_cross_margin=combined_cross,
            margin_savings=savings,
            savings_pct=savings_pct,
            correlation=correlation,
            hedge_ratio=hedge_ratio,
            benefit_type=benefit_type,
        )

    def calculate_portfolio_margin_summary(self) -> dict:
        """Calculate total portfolio margin with cross-benefits."""
        position_margins = self.calculate_position_margins()
        benefits = self.find_cross_margin_benefits()

        total_standalone = sum(pm.effective_margin for pm in position_margins.values())

        # Apply benefits
        total_benefit = sum(b.margin_savings for b in benefits)

        # Cap benefit at reasonable level (can't go negative)
        total_benefit = min(total_benefit, total_standalone * 0.5)

        total_with_cross = total_standalone - total_benefit

        return {
            'total_positions': len(position_margins),
            'total_market_value': sum(abs(pm.market_value) for pm in position_margins.values()),
            'standalone_margin': total_standalone,
            'cross_margin_benefit': total_benefit,
            'effective_margin': total_with_cross,
            'margin_reduction_pct': total_benefit / total_standalone * 100 if total_standalone > 0 else 0,
            'cross_margin_pairs': len(benefits),
            'top_benefits': [b.to_dict() for b in benefits[:5]],
            'positions': {sym: pm.to_dict() for sym, pm in position_margins.items()},
        }


class RiskContributionAnalyzer:
    """
    Analyzes risk contribution by strategy (#R16).

    Decomposes portfolio risk into strategy components.
    """

    def __init__(self):
        # Strategy positions mapping
        self._strategy_positions: dict[str, dict[str, dict]] = {}  # strategy -> symbol -> position

        # Correlation matrix
        self._correlations: dict[tuple[str, str], float] = {}

        # Volatilities
        self._volatilities: dict[str, float] = {}

        # Performance data
        self._strategy_returns: dict[str, list[float]] = {}
        self._strategy_sharpe: dict[str, float] = {}

    def register_strategy_position(
        self,
        strategy: str,
        symbol: str,
        quantity: int,
        market_value: float,
        volatility: float | None = None,
    ) -> None:
        """Register a position under a strategy."""
        if strategy not in self._strategy_positions:
            self._strategy_positions[strategy] = {}

        self._strategy_positions[strategy][symbol] = {
            'quantity': quantity,
            'market_value': market_value,
        }

        if volatility is not None:
            self._volatilities[symbol] = volatility

    def set_correlation(self, symbol1: str, symbol2: str, correlation: float) -> None:
        """Set correlation between symbols."""
        key = tuple(sorted([symbol1, symbol2]))
        self._correlations[key] = correlation

    def set_strategy_performance(
        self,
        strategy: str,
        returns: list[float],
        sharpe_ratio: float,
    ) -> None:
        """Set historical performance for a strategy."""
        self._strategy_returns[strategy] = returns
        self._strategy_sharpe[strategy] = sharpe_ratio

    def calculate_strategy_var(
        self,
        strategy: str,
        confidence: float = 0.95,
    ) -> float:
        """Calculate VaR for a single strategy."""
        positions = self._strategy_positions.get(strategy, {})

        if not positions:
            return 0.0

        # Simple parametric VaR
        total_var = 0.0
        symbols = list(positions.keys())

        for i, sym1 in enumerate(symbols):
            pos1 = positions[sym1]
            vol1 = self._volatilities.get(sym1, 0.02)
            mv1 = pos1['market_value']

            for j, sym2 in enumerate(symbols):
                pos2 = positions[sym2]
                vol2 = self._volatilities.get(sym2, 0.02)
                mv2 = pos2['market_value']

                if i == j:
                    corr = 1.0
                else:
                    key = tuple(sorted([sym1, sym2]))
                    corr = self._correlations.get(key, 0.3)

                total_var += mv1 * mv2 * vol1 * vol2 * corr

        portfolio_vol = math.sqrt(max(0, total_var))

        # VaR using normal distribution
        z_score = 1.645 if confidence == 0.95 else 2.326  # 99%
        var = portfolio_vol * z_score

        return var

    def calculate_marginal_var(
        self,
        strategy: str,
        portfolio_var: float,
        confidence: float = 0.95,
    ) -> float:
        """Calculate marginal VaR contribution of a strategy."""
        # VaR without this strategy
        all_strategies = list(self._strategy_positions.keys())
        other_strategies = [s for s in all_strategies if s != strategy]

        # Temporarily remove strategy and recalculate
        var_without = self._calculate_combined_var(other_strategies, confidence)

        # Marginal contribution
        return portfolio_var - var_without

    def _calculate_combined_var(
        self,
        strategies: list[str],
        confidence: float = 0.95,
    ) -> float:
        """Calculate combined VaR for multiple strategies."""
        if not strategies:
            return 0.0

        # Combine all positions
        combined_positions: dict[str, dict] = {}
        for strategy in strategies:
            for symbol, pos in self._strategy_positions.get(strategy, {}).items():
                if symbol in combined_positions:
                    combined_positions[symbol]['market_value'] += pos['market_value']
                    combined_positions[symbol]['quantity'] += pos['quantity']
                else:
                    combined_positions[symbol] = pos.copy()

        # Calculate VaR on combined
        total_var = 0.0
        symbols = list(combined_positions.keys())

        for i, sym1 in enumerate(symbols):
            vol1 = self._volatilities.get(sym1, 0.02)
            mv1 = combined_positions[sym1]['market_value']

            for j, sym2 in enumerate(symbols):
                vol2 = self._volatilities.get(sym2, 0.02)
                mv2 = combined_positions[sym2]['market_value']

                if i == j:
                    corr = 1.0
                else:
                    key = tuple(sorted([sym1, sym2]))
                    corr = self._correlations.get(key, 0.3)

                total_var += mv1 * mv2 * vol1 * vol2 * corr

        portfolio_vol = math.sqrt(max(0, total_var))
        z_score = 1.645 if confidence == 0.95 else 2.326
        return portfolio_vol * z_score

    def analyze_all_strategies(
        self,
        total_margin: float,
        confidence: float = 0.95,
    ) -> list[StrategyRiskContribution]:
        """
        Analyze risk contribution for all strategies.

        Returns list of StrategyRiskContribution sorted by VaR contribution.
        """
        strategies = list(self._strategy_positions.keys())

        # Calculate portfolio VaR
        portfolio_var = self._calculate_combined_var(strategies, confidence)

        contributions = []

        for strategy in strategies:
            positions = self._strategy_positions.get(strategy, {})

            # Exposures
            gross_exposure = sum(abs(p['market_value']) for p in positions.values())
            net_exposure = sum(p['market_value'] for p in positions.values())

            # Strategy VaR
            strategy_var = self.calculate_strategy_var(strategy, confidence)

            # Marginal VaR
            marginal_var = self.calculate_marginal_var(strategy, portfolio_var, confidence)

            # VaR contribution percentage
            var_contrib_pct = (strategy_var / portfolio_var * 100) if portfolio_var > 0 else 0

            # Margin allocation (proportional to VaR)
            margin_required = total_margin * (strategy_var / portfolio_var) if portfolio_var > 0 else 0
            margin_pct = margin_required / total_margin * 100 if total_margin > 0 else 0

            # Performance metrics
            sharpe = self._strategy_sharpe.get(strategy)

            # Return on margin
            returns = self._strategy_returns.get(strategy, [])
            if returns and margin_required > 0:
                avg_return = sum(returns) / len(returns) * 252  # Annualized
                rom = avg_return / margin_required
            else:
                rom = None

            contributions.append(StrategyRiskContribution(
                strategy_name=strategy,
                positions=list(positions.keys()),
                gross_exposure=gross_exposure,
                net_exposure=net_exposure,
                var_contribution=strategy_var,
                var_contribution_pct=var_contrib_pct,
                marginal_var=marginal_var,
                margin_required=margin_required,
                margin_pct_of_portfolio=margin_pct,
                sharpe_ratio=sharpe,
                return_on_margin=rom,
                risk_adjusted_return=sharpe * rom if sharpe and rom else None,
            ))

        # Sort by VaR contribution
        contributions.sort(key=lambda x: x.var_contribution, reverse=True)
        return contributions

    def get_risk_attribution_summary(
        self,
        total_margin: float,
        confidence: float = 0.95,
    ) -> dict:
        """Generate full risk attribution summary."""
        contributions = self.analyze_all_strategies(total_margin, confidence)
        strategies = list(self._strategy_positions.keys())
        portfolio_var = self._calculate_combined_var(strategies, confidence)

        # Diversification benefit
        sum_individual_var = sum(c.var_contribution for c in contributions)
        diversification_benefit = sum_individual_var - portfolio_var

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'portfolio_var': portfolio_var,
            'sum_strategy_var': sum_individual_var,
            'diversification_benefit': diversification_benefit,
            'diversification_ratio': portfolio_var / sum_individual_var if sum_individual_var > 0 else 1.0,
            'total_margin': total_margin,
            'num_strategies': len(contributions),
            'strategy_contributions': [c.to_dict() for c in contributions],
            'top_risk_contributor': contributions[0].strategy_name if contributions else None,
            'most_efficient': min(contributions, key=lambda x: x.var_contribution_pct / max(0.01, x.margin_pct_of_portfolio)).strategy_name if contributions else None,
        }

    def clear_all(self) -> None:
        """Clear all data for reset."""
        self._strategy_positions.clear()
        self._correlations.clear()
        self._volatilities.clear()
        self._strategy_returns.clear()
        self._strategy_sharpe.clear()

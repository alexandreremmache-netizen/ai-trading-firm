"""
Risk Factor Decomposition Module
================================

Risk factor analysis and decomposition (Issue #R12).

Features:
- Multi-factor risk model (market, size, value, momentum, quality)
- Factor exposure calculation
- Risk contribution attribution
- Factor covariance estimation
- Systematic vs idiosyncratic risk separation
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class RiskFactor(str, Enum):
    """Standard risk factors."""
    # Market factors
    MARKET = "market"  # Beta to market
    SMB = "smb"  # Small minus big (size)
    HML = "hml"  # High minus low (value)
    MOM = "mom"  # Momentum
    QMJ = "qmj"  # Quality minus junk

    # Fixed income factors
    DURATION = "duration"  # Interest rate sensitivity
    CREDIT = "credit"  # Credit spread sensitivity
    INFLATION = "inflation"  # Inflation beta

    # Volatility factors
    VEGA = "vega"  # Volatility sensitivity
    CONVEXITY = "convexity"  # Gamma-like exposure

    # Currency factors
    USD = "usd"  # Dollar exposure
    EUR = "eur"  # Euro exposure
    JPY = "jpy"  # Yen exposure

    # Commodity factors
    COMMODITY = "commodity"  # Broad commodity
    ENERGY = "energy"  # Energy subsector
    METALS = "metals"  # Metals subsector
    AGRICULTURE = "agriculture"  # Ag subsector

    # Sector factors
    TECHNOLOGY = "technology"
    FINANCIALS = "financials"
    HEALTHCARE = "healthcare"
    INDUSTRIALS = "industrials"
    CONSUMER = "consumer"
    UTILITIES = "utilities"


@dataclass
class FactorExposure:
    """Single factor exposure measurement."""
    factor: RiskFactor
    exposure: float  # Beta/loading
    t_stat: float  # Statistical significance
    contribution_pct: float  # % of variance explained


@dataclass
class PositionFactorExposures:
    """Factor exposures for a single position."""
    symbol: str
    exposures: dict[RiskFactor, FactorExposure] = field(default_factory=dict)
    r_squared: float = 0.0  # Total explained variance
    idiosyncratic_vol: float = 0.0  # Unexplained volatility

    def get_exposure(self, factor: RiskFactor) -> float:
        """Get exposure to a specific factor."""
        if factor in self.exposures:
            return self.exposures[factor].exposure
        return 0.0


@dataclass
class PortfolioFactorDecomposition:
    """Full portfolio factor decomposition."""
    timestamp: datetime
    total_variance: float
    systematic_variance: float
    idiosyncratic_variance: float
    factor_exposures: dict[RiskFactor, float] = field(default_factory=dict)
    factor_contributions: dict[RiskFactor, float] = field(default_factory=dict)  # % of variance
    position_exposures: dict[str, PositionFactorExposures] = field(default_factory=dict)
    correlation_with_market: float = 0.0
    tracking_error: float = 0.0  # vs benchmark

    def get_systematic_pct(self) -> float:
        """Get percentage of risk that is systematic."""
        if self.total_variance == 0:
            return 0.0
        return self.systematic_variance / self.total_variance

    def get_top_factors(self, n: int = 5) -> list[tuple[RiskFactor, float]]:
        """Get top N factors by contribution."""
        sorted_factors = sorted(
            self.factor_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_factors[:n]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_variance': self.total_variance,
            'systematic_variance': self.systematic_variance,
            'idiosyncratic_variance': self.idiosyncratic_variance,
            'systematic_pct': self.get_systematic_pct() * 100,
            'factor_exposures': {k.value: v for k, v in self.factor_exposures.items()},
            'factor_contributions_pct': {k.value: v * 100 for k, v in self.factor_contributions.items()},
            'correlation_with_market': self.correlation_with_market,
            'tracking_error_pct': self.tracking_error * 100,
        }


class FactorModel:
    """
    Multi-factor risk model.

    Implements factor-based risk decomposition.
    """

    def __init__(
        self,
        factors: list[RiskFactor] | None = None,
        lookback_days: int = 252,
        min_observations: int = 60,
    ):
        self.factors = factors or [
            RiskFactor.MARKET,
            RiskFactor.SMB,
            RiskFactor.HML,
            RiskFactor.MOM,
        ]
        self.lookback_days = lookback_days
        self.min_observations = min_observations

        # Factor return history
        self._factor_returns: dict[RiskFactor, list[float]] = defaultdict(list)

        # Position return history
        self._position_returns: dict[str, list[float]] = defaultdict(list)

        # Factor covariance matrix
        self._factor_covariance: dict[tuple[RiskFactor, RiskFactor], float] = {}

        # Cached exposures
        self._cached_exposures: dict[str, PositionFactorExposures] = {}
        self._cache_timestamp: datetime | None = None

    def update_factor_returns(self, factor_returns: dict[RiskFactor, float]) -> None:
        """Update factor return history."""
        for factor, ret in factor_returns.items():
            self._factor_returns[factor].append(ret)

            # Trim to lookback
            if len(self._factor_returns[factor]) > self.lookback_days:
                self._factor_returns[factor] = self._factor_returns[factor][-self.lookback_days:]

        # Invalidate cache
        self._cache_timestamp = None

    def update_position_returns(self, position_returns: dict[str, float]) -> None:
        """Update position return history."""
        for symbol, ret in position_returns.items():
            self._position_returns[symbol].append(ret)

            # Trim to lookback
            if len(self._position_returns[symbol]) > self.lookback_days:
                self._position_returns[symbol] = self._position_returns[symbol][-self.lookback_days:]

        # Invalidate cache
        self._cache_timestamp = None

    def estimate_factor_covariance(self) -> dict[tuple[RiskFactor, RiskFactor], float]:
        """Estimate factor covariance matrix."""
        self._factor_covariance = {}

        for f1 in self.factors:
            for f2 in self.factors:
                ret1 = self._factor_returns.get(f1, [])
                ret2 = self._factor_returns.get(f2, [])

                if len(ret1) < self.min_observations or len(ret2) < self.min_observations:
                    continue

                # Align lengths
                min_len = min(len(ret1), len(ret2))
                ret1 = ret1[-min_len:]
                ret2 = ret2[-min_len:]

                # Calculate covariance
                cov = self._covariance(ret1, ret2)
                self._factor_covariance[(f1, f2)] = cov

        return self._factor_covariance

    def calculate_position_exposures(self, symbol: str) -> PositionFactorExposures:
        """
        Calculate factor exposures for a single position.

        Uses OLS regression of position returns on factor returns.
        """
        position_returns = self._position_returns.get(symbol, [])
        if len(position_returns) < self.min_observations:
            return PositionFactorExposures(symbol=symbol)

        exposures = PositionFactorExposures(symbol=symbol)

        # Prepare factor return matrix
        factor_data = {}
        min_len = len(position_returns)

        for factor in self.factors:
            factor_rets = self._factor_returns.get(factor, [])
            if len(factor_rets) >= self.min_observations:
                factor_data[factor] = factor_rets[-min_len:]
                min_len = min(min_len, len(factor_data[factor]))

        if not factor_data or min_len < self.min_observations:
            return exposures

        # Align all data
        y = position_returns[-min_len:]
        X = {f: rets[-min_len:] for f, rets in factor_data.items()}

        # Run univariate regressions for each factor (simplified)
        total_explained_var = 0.0
        y_var = self._variance(y)

        for factor, factor_rets in X.items():
            # Simple OLS: beta = cov(y, x) / var(x)
            cov_yx = self._covariance(y, factor_rets)
            var_x = self._variance(factor_rets)

            if var_x == 0:
                continue

            beta = cov_yx / var_x

            # Calculate residuals and t-stat
            residuals = [yi - beta * xi for yi, xi in zip(y, factor_rets)]
            residual_var = self._variance(residuals)

            se_beta = math.sqrt(residual_var / (var_x * len(y))) if var_x > 0 else 0
            t_stat = beta / se_beta if se_beta > 0 else 0

            # Contribution to variance
            explained_var = beta ** 2 * var_x
            contribution_pct = explained_var / y_var if y_var > 0 else 0
            total_explained_var += explained_var

            exposures.exposures[factor] = FactorExposure(
                factor=factor,
                exposure=beta,
                t_stat=t_stat,
                contribution_pct=contribution_pct,
            )

        # R-squared (approximate, ignoring multicollinearity)
        exposures.r_squared = min(1.0, total_explained_var / y_var) if y_var > 0 else 0
        exposures.idiosyncratic_vol = math.sqrt(max(0, y_var - total_explained_var)) * math.sqrt(252)

        return exposures

    def decompose_portfolio(
        self,
        positions: dict[str, float],  # symbol -> weight
    ) -> PortfolioFactorDecomposition:
        """
        Decompose portfolio risk by factor.

        Args:
            positions: Dictionary of symbol to portfolio weight

        Returns:
            PortfolioFactorDecomposition with full analysis
        """
        result = PortfolioFactorDecomposition(
            timestamp=datetime.now(timezone.utc),
            total_variance=0.0,
            systematic_variance=0.0,
            idiosyncratic_variance=0.0,
        )

        # Calculate position-level exposures
        portfolio_exposures: dict[RiskFactor, float] = defaultdict(float)

        for symbol, weight in positions.items():
            pos_exposures = self.calculate_position_exposures(symbol)
            result.position_exposures[symbol] = pos_exposures

            # Aggregate to portfolio level
            for factor, exposure in pos_exposures.exposures.items():
                portfolio_exposures[factor] += weight * exposure.exposure

        result.factor_exposures = dict(portfolio_exposures)

        # Calculate portfolio variance using factor model
        # Var(Rp) = β'Σβ + Σw²σ²_ε

        # Systematic component (factor-based)
        systematic_var = 0.0
        for f1, exp1 in portfolio_exposures.items():
            for f2, exp2 in portfolio_exposures.items():
                cov = self._factor_covariance.get((f1, f2), 0)
                systematic_var += exp1 * exp2 * cov

        # Idiosyncratic component (weighted sum of position idio var)
        idio_var = 0.0
        for symbol, weight in positions.items():
            if symbol in result.position_exposures:
                idio_vol = result.position_exposures[symbol].idiosyncratic_vol
                idio_var += (weight ** 2) * (idio_vol ** 2 / 252)  # Convert to daily

        # Annualize
        result.systematic_variance = systematic_var * 252
        result.idiosyncratic_variance = idio_var * 252
        result.total_variance = result.systematic_variance + result.idiosyncratic_variance

        # Factor contributions
        if result.total_variance > 0:
            for factor, exposure in portfolio_exposures.items():
                factor_var = self._factor_covariance.get((factor, factor), 0)
                marginal_contribution = exposure ** 2 * factor_var * 252
                result.factor_contributions[factor] = marginal_contribution / result.total_variance

        # Market correlation
        if RiskFactor.MARKET in self._factor_returns:
            market_rets = self._factor_returns[RiskFactor.MARKET]
            portfolio_rets = self._calculate_portfolio_returns(positions)

            if len(market_rets) >= self.min_observations and len(portfolio_rets) >= self.min_observations:
                min_len = min(len(market_rets), len(portfolio_rets))
                result.correlation_with_market = self._correlation(
                    market_rets[-min_len:],
                    portfolio_rets[-min_len:]
                )

        return result

    def _calculate_portfolio_returns(self, positions: dict[str, float]) -> list[float]:
        """Calculate historical portfolio returns."""
        # Find common length
        min_len = float('inf')
        for symbol in positions:
            if symbol in self._position_returns:
                min_len = min(min_len, len(self._position_returns[symbol]))

        if min_len == float('inf') or min_len == 0:
            return []

        min_len = int(min_len)
        portfolio_rets = [0.0] * min_len

        for symbol, weight in positions.items():
            if symbol in self._position_returns:
                pos_rets = self._position_returns[symbol][-min_len:]
                for i, ret in enumerate(pos_rets):
                    portfolio_rets[i] += weight * ret

        return portfolio_rets

    def _covariance(self, x: list[float], y: list[float]) -> float:
        """Calculate sample covariance."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        return sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (n - 1)

    def _variance(self, x: list[float]) -> float:
        """Calculate sample variance."""
        if len(x) < 2:
            return 0.0
        return statistics.variance(x)

    def _correlation(self, x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        cov = self._covariance(x, y)
        std_x = math.sqrt(self._variance(x))
        std_y = math.sqrt(self._variance(y))

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)


@dataclass
class RiskContribution:
    """Risk contribution by source."""
    source: str  # Position, factor, or strategy
    absolute_var: float  # Absolute variance contribution
    marginal_var: float  # Marginal variance contribution
    percent_of_total: float  # Percentage of total variance


class RiskContributionAnalyzer:
    """
    Analyzes risk contributions from different sources.

    Supports position-level, factor-level, and strategy-level attribution.
    """

    def __init__(self, factor_model: FactorModel):
        self.factor_model = factor_model

    def analyze_position_contributions(
        self,
        positions: dict[str, float],  # symbol -> weight
        position_vols: dict[str, float],  # symbol -> volatility
        correlation_matrix: dict[tuple[str, str], float],
    ) -> list[RiskContribution]:
        """
        Analyze risk contribution by position.

        Uses Euler decomposition for additive risk attribution.
        """
        contributions = []

        # Calculate total portfolio variance
        total_var = 0.0
        for s1, w1 in positions.items():
            for s2, w2 in positions.items():
                vol1 = position_vols.get(s1, 0)
                vol2 = position_vols.get(s2, 0)

                if s1 == s2:
                    corr = 1.0
                else:
                    corr = correlation_matrix.get((s1, s2), 0)
                    if corr == 0:
                        corr = correlation_matrix.get((s2, s1), 0)

                total_var += w1 * w2 * vol1 * vol2 * corr

        if total_var == 0:
            return contributions

        # Calculate marginal and absolute contributions
        for symbol, weight in positions.items():
            vol = position_vols.get(symbol, 0)

            # Marginal contribution (derivative of portfolio vol w.r.t. weight)
            marginal_cov = 0.0
            for s2, w2 in positions.items():
                vol2 = position_vols.get(s2, 0)
                if symbol == s2:
                    corr = 1.0
                else:
                    corr = correlation_matrix.get((symbol, s2), 0)
                    if corr == 0:
                        corr = correlation_matrix.get((s2, symbol), 0)

                marginal_cov += w2 * vol * vol2 * corr

            marginal_var = marginal_cov

            # Absolute contribution (weight * marginal)
            absolute_var = weight * marginal_var

            contributions.append(RiskContribution(
                source=symbol,
                absolute_var=absolute_var,
                marginal_var=marginal_var,
                percent_of_total=(absolute_var / total_var) * 100 if total_var > 0 else 0,
            ))

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.absolute_var), reverse=True)

        return contributions

    def analyze_factor_contributions(
        self,
        decomposition: PortfolioFactorDecomposition,
    ) -> list[RiskContribution]:
        """
        Analyze risk contribution by factor.

        Based on factor decomposition results.
        """
        contributions = []

        for factor, contribution_pct in decomposition.factor_contributions.items():
            factor_var = contribution_pct * decomposition.total_variance

            contributions.append(RiskContribution(
                source=factor.value,
                absolute_var=factor_var,
                marginal_var=factor_var,  # Simplified
                percent_of_total=contribution_pct * 100,
            ))

        # Add idiosyncratic
        if decomposition.idiosyncratic_variance > 0:
            idio_pct = decomposition.idiosyncratic_variance / decomposition.total_variance
            contributions.append(RiskContribution(
                source="idiosyncratic",
                absolute_var=decomposition.idiosyncratic_variance,
                marginal_var=decomposition.idiosyncratic_variance,
                percent_of_total=idio_pct * 100,
            ))

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.absolute_var), reverse=True)

        return contributions


@dataclass
class FactorTilt:
    """Intentional factor tilt relative to benchmark."""
    factor: RiskFactor
    portfolio_exposure: float
    benchmark_exposure: float
    active_exposure: float
    contribution_to_te: float  # Contribution to tracking error


class ActiveRiskDecomposer:
    """
    Decomposes active risk (tracking error) by factor.

    Useful for understanding sources of benchmark-relative risk.
    """

    def __init__(self, factor_model: FactorModel):
        self.factor_model = factor_model

    def decompose_active_risk(
        self,
        portfolio_positions: dict[str, float],
        benchmark_positions: dict[str, float],
    ) -> tuple[float, list[FactorTilt]]:
        """
        Decompose tracking error by factor tilt.

        Returns:
            - Tracking error (annualized volatility)
            - List of factor tilts with contributions
        """
        # Get decompositions
        portfolio_decomp = self.factor_model.decompose_portfolio(portfolio_positions)
        benchmark_decomp = self.factor_model.decompose_portfolio(benchmark_positions)

        tilts = []
        total_active_var = 0.0

        # Compare factor exposures
        all_factors = set(portfolio_decomp.factor_exposures.keys()) | set(benchmark_decomp.factor_exposures.keys())

        for factor in all_factors:
            port_exp = portfolio_decomp.factor_exposures.get(factor, 0)
            bench_exp = benchmark_decomp.factor_exposures.get(factor, 0)
            active_exp = port_exp - bench_exp

            # Contribution to tracking error (simplified)
            factor_var = self.factor_model._factor_covariance.get((factor, factor), 0)
            contrib_to_te = active_exp ** 2 * factor_var * 252

            total_active_var += contrib_to_te

            tilts.append(FactorTilt(
                factor=factor,
                portfolio_exposure=port_exp,
                benchmark_exposure=bench_exp,
                active_exposure=active_exp,
                contribution_to_te=contrib_to_te,
            ))

        # Add idiosyncratic component
        idio_diff = portfolio_decomp.idiosyncratic_variance - benchmark_decomp.idiosyncratic_variance
        total_active_var += abs(idio_diff)

        tracking_error = math.sqrt(max(0, total_active_var))

        # Sort by contribution
        tilts.sort(key=lambda x: abs(x.contribution_to_te), reverse=True)

        return tracking_error, tilts

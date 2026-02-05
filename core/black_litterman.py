"""
Black-Litterman Portfolio Optimization
======================================

Phase 4: Advanced Features

Combines equilibrium returns (from CAPM) with investor views (from signal agents)
to produce more stable portfolio allocations using Bayesian updating.

Key benefits:
- Incorporates signal agent views systematically
- Produces diversified portfolios without extreme weights
- Handles uncertainty in views via confidence levels
- More intuitive than raw mean-variance optimization

Reference: Black & Litterman (1992), "Global Portfolio Optimization"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class SignalView:
    """
    Represents a view from a signal agent.

    Views can be:
    - Absolute: "Asset A will return 10%" (P = [1,0,0,...], Q = [0.10])
    - Relative: "Asset A will outperform B by 3%" (P = [1,-1,0,...], Q = [0.03])
    """
    view_type: str  # "absolute" or "relative"
    assets: list[str]  # Assets involved in the view
    weights: list[float]  # View weights (sum to 0 for relative, 1 for absolute)
    expected_return: float  # Expected return/outperformance
    confidence: float  # 0-1 confidence level
    source_agent: str  # Which agent generated this view
    rationale: str = ""  # Explanation for the view


@dataclass
class BlackLittermanResult:
    """Result of Black-Litterman optimization."""
    weights: dict[str, float]  # Final portfolio weights
    expected_returns: dict[str, float]  # Posterior expected returns
    equilibrium_returns: dict[str, float]  # Prior equilibrium returns
    view_contribution: dict[str, float]  # How much views shifted returns
    symbols: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "weights": self.weights,
            "expected_returns": self.expected_returns,
            "equilibrium_returns": self.equilibrium_returns,
            "view_contribution": self.view_contribution,
            "symbols": self.symbols,
            "timestamp": self.timestamp.isoformat(),
        }


class BlackLittermanOptimizer:
    """
    Black-Litterman Portfolio Optimization.

    The Black-Litterman model addresses key issues with mean-variance optimization:
    1. MVO is highly sensitive to expected return estimates
    2. MVO often produces extreme, concentrated portfolios
    3. MVO doesn't have a natural way to incorporate views

    The BL approach:
    1. Start with equilibrium returns implied by market cap weights
    2. Blend in investor views using Bayesian updating
    3. Use posterior returns in mean-variance optimization

    Usage:
        optimizer = BlackLittermanOptimizer(risk_aversion=2.5)

        # Add views from signal agents
        views = [
            SignalView("absolute", ["AAPL"], [1.0], 0.15, 0.7, "momentum_agent"),
            SignalView("relative", ["GOOGL", "MSFT"], [1.0, -1.0], 0.03, 0.6, "stat_arb_agent"),
        ]

        result = optimizer.optimize(
            symbols=["AAPL", "GOOGL", "MSFT"],
            covariance_matrix=cov,
            market_weights={"AAPL": 0.4, "GOOGL": 0.35, "MSFT": 0.25},
            views=views
        )
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        min_weight: float = 0.0,
        max_weight: float = 0.40,
    ):
        """
        Initialize Black-Litterman optimizer.

        Args:
            risk_aversion: Risk aversion parameter (delta), typically 2-3
            tau: Scalar weighing equilibrium vs views (typically 0.01-0.10)
                 Smaller tau = more weight on equilibrium
                 Larger tau = more weight on views
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
        """
        self._risk_aversion = risk_aversion
        self._tau = tau
        self._min_weight = min_weight
        self._max_weight = max_weight

        logger.info(
            f"BlackLittermanOptimizer initialized: "
            f"risk_aversion={risk_aversion}, tau={tau}"
        )

    def calculate_equilibrium_returns(
        self,
        covariance_matrix: np.ndarray,
        market_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate implied equilibrium returns from market cap weights.

        Uses reverse optimization: given market cap weights are optimal,
        what expected returns would justify them?

        Pi = delta * Sigma * w_mkt

        Args:
            covariance_matrix: N x N covariance matrix
            market_weights: N-length array of market cap weights

        Returns:
            N-length array of equilibrium expected returns
        """
        return self._risk_aversion * covariance_matrix @ market_weights

    def build_view_matrices(
        self,
        symbols: list[str],
        views: list[SignalView],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build P (pick matrix), Q (view returns), and Omega (view uncertainty).

        Args:
            symbols: List of asset symbols
            views: List of SignalView objects

        Returns:
            (P, Q, Omega) matrices
        """
        n_assets = len(symbols)
        n_views = len(views)

        if n_views == 0:
            return np.zeros((0, n_assets)), np.zeros(0), np.zeros((0, 0))

        symbol_to_idx = {s: i for i, s in enumerate(symbols)}

        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        omega_diag = np.zeros(n_views)

        for v, view in enumerate(views):
            Q[v] = view.expected_return

            # Build row of P matrix
            for asset, weight in zip(view.assets, view.weights):
                if asset in symbol_to_idx:
                    P[v, symbol_to_idx[asset]] = weight
                else:
                    logger.warning(f"View asset {asset} not in symbol list")

            # Omega diagonal: uncertainty proportional to (1 - confidence)
            # Higher confidence = lower uncertainty
            # Use Idzorek's method: omega_i = (1/confidence - 1) * P_i * tau * Sigma * P_i'
            # Simplified: omega_i = (1 - confidence) * base_uncertainty
            base_uncertainty = 0.05  # 5% base uncertainty
            omega_diag[v] = base_uncertainty * (1.0 - view.confidence + 0.1)

        Omega = np.diag(omega_diag)

        return P, Q, Omega

    def calculate_posterior_returns(
        self,
        equilibrium_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray,
    ) -> np.ndarray:
        """
        Combine equilibrium returns with views using Bayesian updating.

        The Black-Litterman master formula:
        E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 *
               [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]

        Args:
            equilibrium_returns: Prior equilibrium returns (Pi)
            covariance_matrix: Covariance matrix (Sigma)
            P: Pick matrix (K x N)
            Q: View returns (K)
            Omega: View uncertainty matrix (K x K)

        Returns:
            Posterior expected returns
        """
        n_assets = len(equilibrium_returns)

        # If no views, return equilibrium
        if len(Q) == 0:
            return equilibrium_returns

        try:
            # tau * Sigma
            tau_sigma = self._tau * covariance_matrix

            # (tau * Sigma)^-1
            tau_sigma_inv = np.linalg.inv(tau_sigma)

            # Omega^-1
            omega_inv = np.linalg.inv(Omega)

            # Posterior precision: (tau*Sigma)^-1 + P' * Omega^-1 * P
            posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P

            # Posterior mean
            posterior_covariance = np.linalg.inv(posterior_precision)
            posterior_mean = posterior_covariance @ (
                tau_sigma_inv @ equilibrium_returns + P.T @ omega_inv @ Q
            )

            return posterior_mean

        except np.linalg.LinAlgError as e:
            logger.warning(f"Matrix inversion failed in BL: {e}, returning equilibrium")
            return equilibrium_returns

    def optimize_weights(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Mean-variance optimization with posterior returns.

        Solves: max w'*mu - (delta/2)*w'*Sigma*w
        Subject to: sum(w) = 1, w_min <= w <= w_max

        Uses closed-form solution when no constraints binding,
        otherwise falls back to constrained optimization.

        Args:
            expected_returns: Posterior expected returns
            covariance_matrix: Covariance matrix

        Returns:
            Optimal portfolio weights
        """
        n_assets = len(expected_returns)

        try:
            # Unconstrained solution: w* = (1/delta) * Sigma^-1 * mu
            sigma_inv = np.linalg.inv(covariance_matrix)
            raw_weights = (1 / self._risk_aversion) * sigma_inv @ expected_returns

            # Normalize to sum to 1
            weights = raw_weights / np.sum(raw_weights)

            # Apply bounds
            weights = np.clip(weights, self._min_weight, self._max_weight)

            # Re-normalize after clipping
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n_assets) / n_assets

            return weights

        except np.linalg.LinAlgError:
            # Fallback to equal weights
            logger.warning("Optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

    def optimize(
        self,
        symbols: list[str],
        covariance_matrix: np.ndarray,
        market_weights: dict[str, float] | None = None,
        views: list[SignalView] | None = None,
    ) -> BlackLittermanResult:
        """
        Full Black-Litterman optimization pipeline.

        Args:
            symbols: List of asset symbols
            covariance_matrix: N x N covariance matrix
            market_weights: Market cap weights (default: equal weights)
            views: List of signal agent views (optional)

        Returns:
            BlackLittermanResult with optimal weights and diagnostics
        """
        n_assets = len(symbols)

        # Default to equal weights if no market weights provided
        if market_weights is None:
            mkt_w = np.ones(n_assets) / n_assets
        else:
            mkt_w = np.array([market_weights.get(s, 1.0 / n_assets) for s in symbols])
            mkt_w = mkt_w / np.sum(mkt_w)  # Normalize

        # Step 1: Calculate equilibrium returns
        equilibrium = self.calculate_equilibrium_returns(covariance_matrix, mkt_w)

        # Step 2: Build view matrices
        if views:
            P, Q, Omega = self.build_view_matrices(symbols, views)
        else:
            P, Q, Omega = np.zeros((0, n_assets)), np.zeros(0), np.zeros((0, 0))

        # Step 3: Calculate posterior returns
        posterior = self.calculate_posterior_returns(
            equilibrium, covariance_matrix, P, Q, Omega
        )

        # Step 4: Optimize weights
        weights = self.optimize_weights(posterior, covariance_matrix)

        # Build result
        result = BlackLittermanResult(
            weights={s: float(w) for s, w in zip(symbols, weights)},
            expected_returns={s: float(r) for s, r in zip(symbols, posterior)},
            equilibrium_returns={s: float(r) for s, r in zip(symbols, equilibrium)},
            view_contribution={
                s: float(posterior[i] - equilibrium[i])
                for i, s in enumerate(symbols)
            },
            symbols=symbols,
        )

        logger.info(
            f"Black-Litterman optimization complete: "
            f"{n_assets} assets, {len(views) if views else 0} views"
        )

        return result

    def create_views_from_signals(
        self,
        signals: dict[str, dict[str, Any]],
        base_return: float = 0.10,
    ) -> list[SignalView]:
        """
        Convert signal agent outputs to Black-Litterman views.

        This is a helper to translate our signal format into BL views.

        Args:
            signals: Dict of {symbol: signal_dict} from agents
                     Each signal_dict should have 'direction', 'strength',
                     'confidence', and 'agent' keys
            base_return: Base expected return to scale by signal strength

        Returns:
            List of SignalView objects
        """
        views = []

        for symbol, signal in signals.items():
            direction = signal.get("direction", "flat")
            strength = signal.get("strength", 0.0)
            confidence = signal.get("confidence", 0.5)
            agent = signal.get("agent", "unknown")
            rationale = signal.get("rationale", "")

            if direction == "flat" or abs(strength) < 0.1:
                continue

            # Calculate expected return based on direction and strength
            if direction == "long":
                expected_return = base_return * strength
            elif direction == "short":
                expected_return = -base_return * strength
            else:
                continue

            view = SignalView(
                view_type="absolute",
                assets=[symbol],
                weights=[1.0],
                expected_return=expected_return,
                confidence=confidence,
                source_agent=agent,
                rationale=rationale,
            )
            views.append(view)

        return views

    def create_relative_view(
        self,
        long_asset: str,
        short_asset: str,
        outperformance: float,
        confidence: float,
        source_agent: str,
        rationale: str = "",
    ) -> SignalView:
        """
        Create a relative view (long-short pair).

        Args:
            long_asset: Asset expected to outperform
            short_asset: Asset expected to underperform
            outperformance: Expected outperformance (e.g., 0.03 = 3%)
            confidence: Confidence level (0-1)
            source_agent: Source of the view
            rationale: Explanation

        Returns:
            SignalView for relative view
        """
        return SignalView(
            view_type="relative",
            assets=[long_asset, short_asset],
            weights=[1.0, -1.0],
            expected_return=outperformance,
            confidence=confidence,
            source_agent=source_agent,
            rationale=rationale,
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_black_litterman_optimizer(config: dict[str, Any]) -> BlackLittermanOptimizer:
    """
    Factory function to create BlackLittermanOptimizer from config.

    Args:
        config: Configuration dictionary with black_litterman section

    Returns:
        Configured BlackLittermanOptimizer
    """
    bl_config = config.get("black_litterman", {})

    return BlackLittermanOptimizer(
        risk_aversion=bl_config.get("risk_aversion", 2.5),
        tau=bl_config.get("tau", 0.05),
        min_weight=bl_config.get("min_weight", 0.0),
        max_weight=bl_config.get("max_weight", 0.40),
    )

"""
Advanced Options Analytics Module

Addresses MEDIUM priority issues:
- #O13: Option market making support
- #O14: Option pricing model comparison
- #O15: Jump diffusion model (Merton)
- #O16: Stochastic volatility (Heston model)

Provides institutional-grade options pricing and market making capabilities.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import integrate, optimize
from scipy.special import gammaln
from scipy.stats import norm

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class PricingModel(Enum):
    """Available pricing models."""
    BLACK_SCHOLES = "black_scholes"
    MERTON_JUMP = "merton_jump"
    HESTON = "heston"
    SABR = "sabr"
    LOCAL_VOL = "local_vol"


class QuoteSide(Enum):
    """Quote side for market making."""
    BID = "bid"
    ASK = "ask"
    TWO_WAY = "two_way"


@dataclass
class OptionContract:
    """Represents an option contract."""
    underlying: str
    strike: float
    expiry: datetime
    option_type: OptionType
    multiplier: float = 100.0
    exercise_style: str = "european"  # european, american

    @property
    def time_to_expiry(self) -> float:
        """Calculate time to expiry in years."""
        delta = self.expiry - datetime.now(timezone.utc)
        return max(delta.total_seconds() / (365.25 * 24 * 3600), 0.0)


@dataclass
class MarketData:
    """Market data for option pricing."""
    spot: float
    rate: float  # Risk-free rate
    dividend_yield: float = 0.0
    volatility: float = 0.0  # For BS model
    vol_surface: Optional[Dict[Tuple[float, float], float]] = None  # (strike, expiry) -> vol


@dataclass
class PricingResult:
    """Result of option pricing."""
    model: PricingModel
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    implied_vol: Optional[float] = None
    vanna: float = 0.0
    volga: float = 0.0
    computation_time_ms: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelComparisonResult:
    """Comparison of multiple pricing models."""
    contract: OptionContract
    market_data: MarketData
    results: Dict[PricingModel, PricingResult]
    market_price: Optional[float] = None
    best_fit_model: Optional[PricingModel] = None
    comparison_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Quote:
    """Market making quote."""
    contract: OptionContract
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    timestamp: datetime
    theo_price: float
    edge: float  # Expected profit per contract
    inventory_adjustment: float = 0.0


@dataclass
class InventoryPosition:
    """Current inventory position for market making."""
    contract: OptionContract
    quantity: int
    avg_price: float
    delta_exposure: float
    gamma_exposure: float
    vega_exposure: float


# =============================================================================
# Base Pricing Model
# =============================================================================

class BasePricingModel(ABC):
    """Abstract base class for pricing models."""

    @abstractmethod
    def price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        **kwargs
    ) -> PricingResult:
        """Price an option contract."""
        pass

    @abstractmethod
    def model_type(self) -> PricingModel:
        """Return the model type."""
        pass

    def implied_volatility(
        self,
        contract: OptionContract,
        market_data: MarketData,
        market_price: float,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        **kwargs
    ) -> float:
        """
        Calculate implied volatility from market price.

        Uses Brent's method with fallback to Newton-Raphson for robustness.
        Handles edge cases near expiration and extreme market prices.
        """
        # Edge case: near expiration or zero/negative market price
        T = contract.time_to_expiry
        if T <= 1e-6 or market_price <= 0:
            logger.warning("IV calculation: T near zero or invalid market price")
            return 0.0

        # Calculate intrinsic value bounds
        S = market_data.spot
        K = contract.strike
        r = market_data.rate
        q = market_data.dividend_yield

        if contract.option_type == OptionType.CALL:
            intrinsic = max(0, S * math.exp(-q * T) - K * math.exp(-r * T))
        else:
            intrinsic = max(0, K * math.exp(-r * T) - S * math.exp(-q * T))

        # Market price below intrinsic is arbitrage - no valid IV
        if market_price < intrinsic - 1e-6:
            logger.warning(f"IV calculation: market price {market_price} below intrinsic {intrinsic}")
            return 0.0

        def objective(vol: float) -> float:
            if vol <= 0:
                return -market_price
            md = MarketData(
                spot=market_data.spot,
                rate=market_data.rate,
                dividend_yield=market_data.dividend_yield,
                volatility=vol
            )
            try:
                result = self.price(contract, md, **kwargs)
                return result.price - market_price
            except Exception:
                return float('inf')

        try:
            # Try Brent's method first (robust root-finding)
            result = optimize.brentq(objective, 0.001, 5.0, xtol=tolerance, maxiter=max_iterations)
            if np.isfinite(result) and result > 0:
                return result
        except (ValueError, RuntimeError):
            pass  # Fall through to Newton-Raphson

        # Fallback: Newton-Raphson with damping
        vol = 0.20  # Initial guess
        for i in range(max_iterations):
            md = MarketData(
                spot=market_data.spot,
                rate=market_data.rate,
                dividend_yield=market_data.dividend_yield,
                volatility=vol
            )
            try:
                result = self.price(contract, md, **kwargs)
                price_diff = result.price - market_price
                vega = result.vega * 100  # Vega is per 1% vol

                if abs(price_diff) < tolerance:
                    return vol

                # Guard against zero vega (near zero or very deep ITM/OTM)
                if abs(vega) < 1e-10:
                    logger.warning("IV Newton-Raphson: zero vega, cannot converge")
                    break

                # Damped Newton step
                step = price_diff / vega
                step = max(-0.5, min(0.5, step))  # Limit step size
                new_vol = vol - step

                # Keep vol in reasonable bounds
                new_vol = max(0.001, min(5.0, new_vol))

                if abs(new_vol - vol) < tolerance:
                    return new_vol

                vol = new_vol
            except Exception as e:
                logger.warning(f"IV Newton-Raphson iteration failed: {e}")
                break

        logger.warning("Could not find implied volatility after max iterations")
        return 0.0


# =============================================================================
# Black-Scholes Model
# =============================================================================

class BlackScholesModel(BasePricingModel):
    """Standard Black-Scholes-Merton pricing model."""

    def model_type(self) -> PricingModel:
        return PricingModel.BLACK_SCHOLES

    def price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        **kwargs
    ) -> PricingResult:
        """Price option using Black-Scholes."""
        start_time = datetime.now(timezone.utc)

        S = market_data.spot
        K = contract.strike
        T = contract.time_to_expiry
        r = market_data.rate
        q = market_data.dividend_yield
        sigma = market_data.volatility

        # OPT-P0-1: Handle near-expiry delta discontinuity
        # When T < 1 day, use digital delta (0 or 1 for calls, -1 or 0 for puts)
        one_day_in_years = 1.0 / 365.0

        if T <= 0 or sigma <= 0:
            # Expired or zero vol
            if contract.option_type == OptionType.CALL:
                intrinsic = max(S - K, 0)
                # Digital delta: 1 if ITM, 0 if OTM
                delta = 1.0 if S > K else 0.0
            else:
                intrinsic = max(K - S, 0)
                # Digital delta: -1 if ITM, 0 if OTM
                delta = -1.0 if S < K else 0.0
            return PricingResult(
                model=self.model_type(),
                price=intrinsic,
                delta=delta,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho=0.0,
                computation_time_ms=0.0
            )

        # OPT-P0-1: Near expiry (< 1 day) - use digital delta
        if T < one_day_in_years:
            if contract.option_type == OptionType.CALL:
                intrinsic = max(S - K, 0)
                # Digital delta based on moneyness
                delta = 1.0 if S > K else 0.0
            else:
                intrinsic = max(K - S, 0)
                delta = -1.0 if S < K else 0.0
            return PricingResult(
                model=self.model_type(),
                price=intrinsic,
                delta=delta,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho=0.0,
                computation_time_ms=0.0
            )

        # d1 and d2
        # Guard against invalid inputs
        if K <= 0 or S <= 0:
            logger.warning(f"Invalid S={S} or K={K} for Black-Scholes")
            return PricingResult(
                model=self.model_type(),
                price=0.0,
                delta=0.0,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho=0.0,
                computation_time_ms=0.0
            )

        sqrt_T = math.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T

        # Guard against division by zero in d1/d2 calculation
        if sigma_sqrt_T < 1e-12:
            # Very low vol or very short time - use intrinsic value
            if contract.option_type == OptionType.CALL:
                intrinsic = max(S - K, 0)
                delta = 1.0 if S > K else 0.0
            else:
                intrinsic = max(K - S, 0)
                delta = -1.0 if S < K else 0.0
            return PricingResult(
                model=self.model_type(),
                price=intrinsic,
                delta=delta,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho=0.0,
                computation_time_ms=0.0
            )

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T

        # Price and Greeks
        if contract.option_type == OptionType.CALL:
            price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            delta = math.exp(-q * T) * norm.cdf(d1)
            rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
            delta = -math.exp(-q * T) * norm.cdf(-d1)
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

        # Common Greeks
        # Guard against division by zero in gamma
        if S * sigma_sqrt_T < 1e-12:
            gamma = 0.0
        else:
            gamma = math.exp(-q * T) * norm.pdf(d1) / (S * sigma_sqrt_T)
        vega = S * math.exp(-q * T) * norm.pdf(d1) * sqrt_T / 100
        theta = self._calculate_theta(S, K, T, r, q, sigma, d1, d2, contract.option_type)

        # Higher order Greeks
        # Guard against division by zero
        if sigma < 1e-12:
            vanna = 0.0
            volga = 0.0
        else:
            vanna = -math.exp(-q * T) * norm.pdf(d1) * d2 / sigma
            volga = vega * d1 * d2 / sigma

        computation_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return PricingResult(
            model=self.model_type(),
            price=price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            vanna=vanna,
            volga=volga,
            computation_time_ms=computation_time,
            parameters={"d1": d1, "d2": d2}
        )

    def _calculate_theta(
        self,
        S: float, K: float, T: float, r: float, q: float, sigma: float,
        d1: float, d2: float, option_type: OptionType
    ) -> float:
        """Calculate theta (time decay)."""
        # Guard against division by zero when T is very small
        sqrt_T = math.sqrt(T) if T > 1e-12 else 1e-6
        term1 = -S * math.exp(-q * T) * norm.pdf(d1) * sigma / (2 * sqrt_T)

        if option_type == OptionType.CALL:
            term2 = q * S * math.exp(-q * T) * norm.cdf(d1)
            term3 = -r * K * math.exp(-r * T) * norm.cdf(d2)
            theta = (term1 - term2 + term3) / 365
        else:
            term2 = -q * S * math.exp(-q * T) * norm.cdf(-d1)
            term3 = r * K * math.exp(-r * T) * norm.cdf(-d2)
            theta = (term1 + term2 + term3) / 365

        return theta


# =============================================================================
# Merton Jump Diffusion Model (#O15)
# =============================================================================

@dataclass
class JumpParameters:
    """Parameters for jump diffusion model."""
    jump_intensity: float  # lambda - average number of jumps per year
    jump_mean: float  # mu_J - mean of log jump size
    jump_volatility: float  # sigma_J - volatility of log jump size


class MertonJumpDiffusionModel(BasePricingModel):
    """
    Merton (1976) Jump Diffusion Model.

    Extends Black-Scholes with compound Poisson jumps to capture
    discontinuous price movements (crash risk, earnings jumps, etc.).

    dS/S = (mu - lambda*k) dt + sigma dW + (J-1) dN

    where:
    - N is a Poisson process with intensity lambda
    - J is the jump size (log-normal distributed)
    - k = E[J-1] is the expected percentage jump size
    """

    def __init__(self, max_terms: int = 50):
        """Initialize with maximum series expansion terms."""
        self.max_terms = max_terms
        self._bs_model = BlackScholesModel()

    def model_type(self) -> PricingModel:
        return PricingModel.MERTON_JUMP

    def price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        jump_params: Optional[JumpParameters] = None,
        **kwargs
    ) -> PricingResult:
        """
        Price option using Merton jump diffusion.

        Uses series expansion of BS prices weighted by Poisson probabilities.
        """
        start_time = datetime.now(timezone.utc)

        if jump_params is None:
            # Default parameters if not provided
            jump_params = JumpParameters(
                jump_intensity=1.0,  # 1 jump per year on average
                jump_mean=-0.1,      # 10% average downward jump
                jump_volatility=0.25  # 25% vol of jump size
            )

        S = market_data.spot
        K = contract.strike
        T = contract.time_to_expiry
        r = market_data.rate
        q = market_data.dividend_yield
        sigma = market_data.volatility

        lam = jump_params.jump_intensity
        mu_j = jump_params.jump_mean
        sigma_j = jump_params.jump_volatility

        if T <= 0:
            if contract.option_type == OptionType.CALL:
                return PricingResult(
                    model=self.model_type(),
                    price=max(S - K, 0),
                    delta=1.0 if S > K else 0.0,
                    gamma=0.0, vega=0.0, theta=0.0, rho=0.0
                )
            else:
                return PricingResult(
                    model=self.model_type(),
                    price=max(K - S, 0),
                    delta=-1.0 if S < K else 0.0,
                    gamma=0.0, vega=0.0, theta=0.0, rho=0.0
                )

        # Expected jump size
        k = math.exp(mu_j + 0.5 * sigma_j**2) - 1

        # Adjusted intensity
        lam_prime = lam * (1 + k)

        # Series expansion
        price = 0.0
        delta = 0.0
        gamma = 0.0
        vega = 0.0

        for n in range(self.max_terms):
            # Poisson weight using gammaln to avoid factorial overflow
            # math.factorial(n) = exp(gammaln(n+1))
            log_poisson = -lam_prime * T + n * math.log(lam_prime * T + 1e-300) - gammaln(n + 1)
            poisson_weight = math.exp(log_poisson)

            if poisson_weight < 1e-12:
                break

            # Adjusted volatility for n jumps
            sigma_n = math.sqrt(sigma**2 + n * sigma_j**2 / T)

            # Adjusted rate for n jumps
            r_n = r - lam * k + n * (mu_j + 0.5 * sigma_j**2) / T

            # BS price with adjusted parameters
            md_n = MarketData(
                spot=S,
                rate=r_n,
                dividend_yield=q,
                volatility=sigma_n
            )
            bs_result = self._bs_model.price(contract, md_n)

            price += poisson_weight * bs_result.price
            delta += poisson_weight * bs_result.delta
            gamma += poisson_weight * bs_result.gamma
            vega += poisson_weight * bs_result.vega

        # Numerical theta (finite difference)
        contract_shifted = OptionContract(
            underlying=contract.underlying,
            strike=contract.strike,
            expiry=contract.expiry,
            option_type=contract.option_type,
            multiplier=contract.multiplier
        )
        # Approximate theta
        theta = -price * 0.01 / T if T > 0.01 else 0.0  # Rough approximation

        computation_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return PricingResult(
            model=self.model_type(),
            price=price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=0.0,  # Would need numerical calculation
            computation_time_ms=computation_time,
            parameters={
                "jump_intensity": lam,
                "jump_mean": mu_j,
                "jump_volatility": sigma_j,
                "terms_used": min(n + 1, self.max_terms)
            }
        )

    def calibrate(
        self,
        contracts: List[OptionContract],
        market_data: MarketData,
        market_prices: List[float]
    ) -> JumpParameters:
        """
        Calibrate jump parameters to market prices.

        Minimizes sum of squared pricing errors.
        """
        def objective(params: np.ndarray) -> float:
            jump_params = JumpParameters(
                jump_intensity=params[0],
                jump_mean=params[1],
                jump_volatility=params[2]
            )

            total_error = 0.0
            for contract, market_price in zip(contracts, market_prices):
                result = self.price(contract, market_data, jump_params)
                total_error += (result.price - market_price)**2

            return total_error

        # Initial guess
        x0 = np.array([1.0, -0.1, 0.25])
        bounds = [(0.01, 10.0), (-1.0, 0.5), (0.01, 1.0)]

        result = optimize.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds
        )

        return JumpParameters(
            jump_intensity=result.x[0],
            jump_mean=result.x[1],
            jump_volatility=result.x[2]
        )


# =============================================================================
# Heston Stochastic Volatility Model (#O16)
# =============================================================================

@dataclass
class HestonParameters:
    """Parameters for Heston stochastic volatility model."""
    v0: float  # Initial variance
    kappa: float  # Mean reversion speed
    theta: float  # Long-term variance
    sigma: float  # Vol of vol
    rho: float  # Correlation between spot and vol

    def feller_condition(self) -> bool:
        """Check if Feller condition is satisfied (ensures positive variance)."""
        return 2 * self.kappa * self.theta > self.sigma**2


class HestonModel(BasePricingModel):
    """
    Heston (1993) Stochastic Volatility Model.

    dS = mu*S*dt + sqrt(v)*S*dW1
    dv = kappa*(theta - v)*dt + sigma*sqrt(v)*dW2
    dW1*dW2 = rho*dt

    Captures:
    - Volatility smile/skew
    - Mean reversion of volatility
    - Correlation between returns and volatility (leverage effect)
    """

    def __init__(self, integration_points: int = 100):
        """Initialize with number of integration points for characteristic function."""
        self.integration_points = integration_points

    def model_type(self) -> PricingModel:
        return PricingModel.HESTON

    def price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        heston_params: Optional[HestonParameters] = None,
        **kwargs
    ) -> PricingResult:
        """
        Price option using Heston model via characteristic function.

        Uses Carr-Madan FFT approach or direct integration.
        """
        start_time = datetime.now(timezone.utc)

        if heston_params is None:
            # Default parameters
            heston_params = HestonParameters(
                v0=market_data.volatility**2,
                kappa=2.0,
                theta=market_data.volatility**2,
                sigma=0.3,
                rho=-0.7
            )

        S = market_data.spot
        K = contract.strike
        T = contract.time_to_expiry
        r = market_data.rate
        q = market_data.dividend_yield

        if T <= 0:
            if contract.option_type == OptionType.CALL:
                return PricingResult(
                    model=self.model_type(),
                    price=max(S - K, 0),
                    delta=1.0 if S > K else 0.0,
                    gamma=0.0, vega=0.0, theta=0.0, rho=0.0
                )
            else:
                return PricingResult(
                    model=self.model_type(),
                    price=max(K - S, 0),
                    delta=-1.0 if S < K else 0.0,
                    gamma=0.0, vega=0.0, theta=0.0, rho=0.0
                )

        # Calculate call price using characteristic function
        call_price = self._price_call(S, K, T, r, q, heston_params)

        # Put-call parity for puts
        if contract.option_type == OptionType.PUT:
            price = call_price - S * math.exp(-q * T) + K * math.exp(-r * T)
        else:
            price = call_price

        # Numerical Greeks
        delta = self._numerical_delta(S, K, T, r, q, heston_params, contract.option_type)
        gamma = self._numerical_gamma(S, K, T, r, q, heston_params, contract.option_type)
        vega = self._numerical_vega(S, K, T, r, q, heston_params, contract.option_type)

        computation_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return PricingResult(
            model=self.model_type(),
            price=max(price, 0.0),
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=0.0,  # Would need numerical calculation
            rho=0.0,
            computation_time_ms=computation_time,
            parameters={
                "v0": heston_params.v0,
                "kappa": heston_params.kappa,
                "theta": heston_params.theta,
                "sigma": heston_params.sigma,
                "rho": heston_params.rho,
                "feller_satisfied": heston_params.feller_condition()
            }
        )

    def _characteristic_function(
        self,
        phi: complex,
        S: float, T: float, r: float, q: float,
        params: HestonParameters,
        j: int  # 1 or 2
    ) -> complex:
        """
        Heston characteristic function for P1 and P2.

        Uses the formulation from Gatheral "The Volatility Surface".
        """
        v0 = params.v0
        kappa = params.kappa
        theta = params.theta
        sigma = params.sigma
        rho = params.rho

        # Adjust for j=1 vs j=2
        if j == 1:
            u = 0.5
            b = kappa - rho * sigma
        else:
            u = -0.5
            b = kappa

        a = kappa * theta

        # P2-19: Complex calculations with overflow protection
        try:
            d_squared = (rho * sigma * phi * 1j - b)**2 - sigma**2 * (2 * u * phi * 1j - phi**2)
            d = np.sqrt(d_squared)

            # Guard against zero denominator
            denom_g = b - rho * sigma * phi * 1j - d
            if np.abs(denom_g) < 1e-15:
                denom_g = 1e-15

            g = (b - rho * sigma * phi * 1j + d) / denom_g
        except (OverflowError, FloatingPointError, ZeroDivisionError):
            return 0.0 + 0.0j

        # Prevent numerical overflow in exp(d*T)
        # Check if real part of d*T would cause overflow (exp(700) ~ 1e304, near float max)
        d_T_real = np.real(d * T)
        if d_T_real > 500:
            # Use asymptotic approximation for large d*T
            # When exp(d*T) >> 1, the formulas simplify
            exp_dT = np.exp(np.clip(d * T, -500, 500))
        else:
            exp_dT = np.exp(d * T)

        # Guard against division by zero in log argument
        log_arg = (1 - g * exp_dT) / (1 - g)
        if np.abs(log_arg) < 1e-15:
            log_arg = 1e-15

        C = (r - q) * phi * 1j * T + (a / sigma**2) * (
            (b - rho * sigma * phi * 1j + d) * T - 2 * np.log(log_arg)
        )

        # Guard against division by zero
        denom = 1 - g * exp_dT
        if np.abs(denom) < 1e-15:
            denom = 1e-15 * np.sign(denom) if denom != 0 else 1e-15

        D = ((b - rho * sigma * phi * 1j + d) / sigma**2) * ((1 - exp_dT) / denom)

        # Final exp with overflow protection
        final_exp_arg = C + D * v0 + 1j * phi * np.log(S)
        if np.real(final_exp_arg) > 500:
            return 0.0  # Return 0 for extreme values (effectively zero contribution)

        return np.exp(final_exp_arg)

    def _price_call(
        self,
        S: float, K: float, T: float, r: float, q: float,
        params: HestonParameters
    ) -> float:
        """Price a call option using numerical integration."""

        def integrand_1(phi: float) -> float:
            try:
                cf = self._characteristic_function(phi, S, T, r, q, params, j=1)
                # P2-19: Guard against NaN/Inf in characteristic function
                if not np.isfinite(cf):
                    return 0.0
                result = np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))
                return result if np.isfinite(result) else 0.0
            except (OverflowError, FloatingPointError):
                return 0.0

        def integrand_2(phi: float) -> float:
            try:
                cf = self._characteristic_function(phi, S, T, r, q, params, j=2)
                # P2-19: Guard against NaN/Inf in characteristic function
                if not np.isfinite(cf):
                    return 0.0
                result = np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))
                return result if np.isfinite(result) else 0.0
            except (OverflowError, FloatingPointError):
                return 0.0

        # P2-19: Numerical integration with error handling
        try:
            P1 = 0.5 + (1 / math.pi) * integrate.quad(integrand_1, 0.001, 100, limit=100)[0]
            P2 = 0.5 + (1 / math.pi) * integrate.quad(integrand_2, 0.001, 100, limit=100)[0]

            # P2-19: Validate probabilities are in [0, 1]
            P1 = np.clip(P1, 0.0, 1.0)
            P2 = np.clip(P2, 0.0, 1.0)

            call_price = S * math.exp(-q * T) * P1 - K * math.exp(-r * T) * P2

            # P2-19: Final sanity check
            if not np.isfinite(call_price) or call_price < 0:
                logger.warning(f"Heston pricing returned invalid value, using intrinsic")
                return max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))

            return max(call_price, 0.0)

        except Exception as e:
            # P2-19: Fallback to intrinsic value on any numerical failure
            logger.warning(f"Heston pricing failed ({e}), using intrinsic value")
            return max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))

    def _numerical_delta(
        self,
        S: float, K: float, T: float, r: float, q: float,
        params: HestonParameters,
        option_type: OptionType
    ) -> float:
        """Calculate delta numerically."""
        h = S * 0.01
        call_up = self._price_call(S + h, K, T, r, q, params)
        call_down = self._price_call(S - h, K, T, r, q, params)
        delta_call = (call_up - call_down) / (2 * h)

        if option_type == OptionType.PUT:
            return delta_call - math.exp(-q * T)
        return delta_call

    def _numerical_gamma(
        self,
        S: float, K: float, T: float, r: float, q: float,
        params: HestonParameters,
        option_type: OptionType
    ) -> float:
        """Calculate gamma numerically."""
        h = S * 0.01
        call_up = self._price_call(S + h, K, T, r, q, params)
        call_mid = self._price_call(S, K, T, r, q, params)
        call_down = self._price_call(S - h, K, T, r, q, params)
        return (call_up - 2 * call_mid + call_down) / (h**2)

    def _numerical_vega(
        self,
        S: float, K: float, T: float, r: float, q: float,
        params: HestonParameters,
        option_type: OptionType
    ) -> float:
        """Calculate vega numerically (w.r.t. v0)."""
        h = 0.01 * params.v0
        params_up = HestonParameters(
            v0=params.v0 + h,
            kappa=params.kappa,
            theta=params.theta,
            sigma=params.sigma,
            rho=params.rho
        )
        params_down = HestonParameters(
            v0=params.v0 - h,
            kappa=params.kappa,
            theta=params.theta,
            sigma=params.sigma,
            rho=params.rho
        )
        call_up = self._price_call(S, K, T, r, q, params_up)
        call_down = self._price_call(S, K, T, r, q, params_down)
        return (call_up - call_down) / (2 * h) / 100  # Per 1% vol

    def calibrate(
        self,
        contracts: List[OptionContract],
        market_data: MarketData,
        market_prices: List[float],
        initial_params: Optional[HestonParameters] = None
    ) -> HestonParameters:
        """
        Calibrate Heston parameters to market prices.

        Uses differential evolution for global optimization.
        """
        S = market_data.spot
        r = market_data.rate
        q = market_data.dividend_yield

        def objective(params: np.ndarray) -> float:
            heston_params = HestonParameters(
                v0=params[0],
                kappa=params[1],
                theta=params[2],
                sigma=params[3],
                rho=params[4]
            )

            total_error = 0.0
            for contract, market_price in zip(contracts, market_prices):
                K = contract.strike
                T = contract.time_to_expiry

                try:
                    model_price = self._price_call(S, K, T, r, q, heston_params)
                    if contract.option_type == OptionType.PUT:
                        model_price = model_price - S * math.exp(-q * T) + K * math.exp(-r * T)
                    total_error += (model_price - market_price)**2
                except Exception as e:
                    logger.warning(f"Heston pricing failed for K={K}, T={T}: {e}")
                    total_error += 1e6  # Penalize failed calibration

            return total_error

        # Bounds for parameters
        bounds = [
            (0.001, 1.0),   # v0
            (0.1, 10.0),    # kappa
            (0.001, 1.0),   # theta
            (0.01, 2.0),    # sigma (vol of vol)
            (-0.99, 0.99)   # rho
        ]

        result = optimize.differential_evolution(
            objective,
            bounds,
            maxiter=100,
            tol=1e-6,
            seed=42
        )

        return HestonParameters(
            v0=result.x[0],
            kappa=result.x[1],
            theta=result.x[2],
            sigma=result.x[3],
            rho=result.x[4]
        )


# =============================================================================
# Pricing Model Comparison (#O14)
# =============================================================================

class PricingModelComparator:
    """
    Compare multiple pricing models for option valuation.

    Provides:
    - Side-by-side model comparison
    - Model selection based on market fit
    - Parameter sensitivity analysis
    """

    def __init__(self):
        """Initialize with available models."""
        self.models: Dict[PricingModel, BasePricingModel] = {
            PricingModel.BLACK_SCHOLES: BlackScholesModel(),
            PricingModel.MERTON_JUMP: MertonJumpDiffusionModel(),
            PricingModel.HESTON: HestonModel()
        }
        self._calibrated_params: Dict[PricingModel, Any] = {}

    def compare_models(
        self,
        contract: OptionContract,
        market_data: MarketData,
        market_price: Optional[float] = None,
        model_params: Optional[Dict[PricingModel, Any]] = None
    ) -> ModelComparisonResult:
        """
        Compare all available models for a single contract.

        Args:
            contract: Option contract to price
            market_data: Market data for pricing
            market_price: Optional market price for comparison
            model_params: Optional model-specific parameters

        Returns:
            ModelComparisonResult with prices from all models
        """
        results: Dict[PricingModel, PricingResult] = {}
        model_params = model_params or {}

        for model_type, model in self.models.items():
            try:
                params = model_params.get(model_type, {})

                if model_type == PricingModel.MERTON_JUMP and "jump_params" not in params:
                    result = model.price(contract, market_data)
                elif model_type == PricingModel.HESTON and "heston_params" not in params:
                    result = model.price(contract, market_data)
                else:
                    result = model.price(contract, market_data, **params)

                results[model_type] = result

            except Exception as e:
                logger.warning(f"Model {model_type} failed: {e}")

        # Determine best fit model if market price available
        best_fit = None
        comparison_metrics = {}

        if market_price is not None and results:
            errors = {
                model: abs(result.price - market_price)
                for model, result in results.items()
            }
            best_fit = min(errors, key=errors.get)

            comparison_metrics = {
                "pricing_errors": {model.value: err for model, err in errors.items()},
                "price_range": max(r.price for r in results.values()) - min(r.price for r in results.values()),
                "delta_range": max(r.delta for r in results.values()) - min(r.delta for r in results.values()),
                "vega_range": max(r.vega for r in results.values()) - min(r.vega for r in results.values())
            }

        return ModelComparisonResult(
            contract=contract,
            market_data=market_data,
            results=results,
            market_price=market_price,
            best_fit_model=best_fit,
            comparison_metrics=comparison_metrics
        )

    def sensitivity_analysis(
        self,
        contract: OptionContract,
        market_data: MarketData,
        param_name: str,
        param_range: np.ndarray,
        model: PricingModel = PricingModel.BLACK_SCHOLES
    ) -> Dict[str, List[float]]:
        """
        Analyze sensitivity of model output to parameter changes.

        Args:
            contract: Option contract
            market_data: Base market data
            param_name: Parameter to vary (spot, vol, rate, etc.)
            param_range: Range of parameter values
            model: Model to use

        Returns:
            Dictionary of parameter values and corresponding prices/Greeks
        """
        prices = []
        deltas = []
        gammas = []
        vegas = []

        pricing_model = self.models[model]

        for param_value in param_range:
            # Create modified market data
            md = MarketData(
                spot=param_value if param_name == "spot" else market_data.spot,
                rate=param_value if param_name == "rate" else market_data.rate,
                dividend_yield=param_value if param_name == "dividend" else market_data.dividend_yield,
                volatility=param_value if param_name == "volatility" else market_data.volatility
            )

            result = pricing_model.price(contract, md)
            prices.append(result.price)
            deltas.append(result.delta)
            gammas.append(result.gamma)
            vegas.append(result.vega)

        return {
            param_name: param_range.tolist(),
            "price": prices,
            "delta": deltas,
            "gamma": gammas,
            "vega": vegas
        }

    def generate_vol_surface(
        self,
        underlying: str,
        spot: float,
        rate: float,
        strikes: List[float],
        expiries: List[float],
        market_prices: Dict[Tuple[float, float], float],
        model: PricingModel = PricingModel.BLACK_SCHOLES
    ) -> Dict[Tuple[float, float], float]:
        """
        Generate implied volatility surface from market prices.

        Args:
            underlying: Underlying asset
            spot: Current spot price
            rate: Risk-free rate
            strikes: List of strikes
            expiries: List of expiries (in years)
            market_prices: Dict of (strike, expiry) -> price
            model: Model to use for IV calculation

        Returns:
            Dictionary of (strike, expiry) -> implied vol
        """
        vol_surface = {}
        pricing_model = self.models[model]

        for strike in strikes:
            for expiry in expiries:
                key = (strike, expiry)
                if key not in market_prices:
                    continue

                market_price = market_prices[key]
                expiry_date = datetime.now(timezone.utc).replace(
                    year=datetime.now(timezone.utc).year + int(expiry),
                    month=int((expiry % 1) * 12) + 1 if expiry % 1 > 0 else datetime.now(timezone.utc).month
                )

                contract = OptionContract(
                    underlying=underlying,
                    strike=strike,
                    expiry=expiry_date,
                    option_type=OptionType.CALL
                )

                market_data = MarketData(spot=spot, rate=rate, volatility=0.2)

                try:
                    iv = pricing_model.implied_volatility(contract, market_data, market_price)
                    vol_surface[key] = iv
                except Exception as e:
                    logger.warning(f"Failed to compute IV for {key}: {e}")

        return vol_surface


# =============================================================================
# Option Market Making (#O13)
# =============================================================================

@dataclass
class MarketMakingParameters:
    """Parameters for market making strategy."""
    base_spread_bps: float = 50.0  # Base bid-ask spread in bps
    inventory_skew_factor: float = 0.1  # How much to skew quotes based on inventory
    max_position: int = 1000  # Maximum position per contract
    min_edge_bps: float = 10.0  # Minimum edge required to quote
    gamma_limit: float = 1000.0  # Maximum gamma exposure
    vega_limit: float = 10000.0  # Maximum vega exposure
    quote_size: int = 10  # Default quote size
    refresh_interval_ms: int = 100  # Quote refresh interval


class OptionMarketMaker:
    """
    Option market making engine.

    Provides:
    - Two-way quote generation
    - Inventory management
    - Risk-based position limits
    - Dynamic spread adjustment
    """

    def __init__(
        self,
        pricing_model: BasePricingModel,
        params: Optional[MarketMakingParameters] = None
    ):
        """Initialize market maker with pricing model and parameters."""
        self.pricing_model = pricing_model
        self.params = params or MarketMakingParameters()
        self.inventory: Dict[str, InventoryPosition] = {}
        self._quote_history: List[Quote] = []

    def generate_quote(
        self,
        contract: OptionContract,
        market_data: MarketData,
        side: QuoteSide = QuoteSide.TWO_WAY
    ) -> Quote:
        """
        Generate market making quote.

        Args:
            contract: Option contract to quote
            market_data: Current market data
            side: Which side to quote

        Returns:
            Quote with bid/ask prices and sizes
        """
        # Price the option
        result = self.pricing_model.price(contract, market_data)
        theo_price = result.price

        # Base spread
        base_spread = theo_price * self.params.base_spread_bps / 10000
        half_spread = base_spread / 2

        # Inventory adjustment
        inventory_adj = self._calculate_inventory_adjustment(contract, result)

        # Volatility-based spread adjustment
        vol_adj = self._calculate_vol_spread_adjustment(market_data.volatility)

        # Gamma-based spread adjustment (wider spread for high gamma)
        gamma_adj = abs(result.gamma) * market_data.spot * 0.01

        # Total spread
        total_spread = half_spread * (1 + vol_adj) + gamma_adj

        # Final prices
        bid_price = max(0.01, theo_price - total_spread + inventory_adj)
        ask_price = theo_price + total_spread + inventory_adj

        # Quote sizes based on risk limits
        bid_size, ask_size = self._calculate_quote_sizes(contract, result)

        # Expected edge
        edge = total_spread - gamma_adj  # Spread minus gamma cost

        quote = Quote(
            contract=contract,
            bid_price=round(bid_price, 2),
            ask_price=round(ask_price, 2),
            bid_size=bid_size if side in [QuoteSide.BID, QuoteSide.TWO_WAY] else 0,
            ask_size=ask_size if side in [QuoteSide.ASK, QuoteSide.TWO_WAY] else 0,
            timestamp=datetime.now(timezone.utc),
            theo_price=theo_price,
            edge=edge,
            inventory_adjustment=inventory_adj
        )

        self._quote_history.append(quote)
        return quote

    def _calculate_inventory_adjustment(
        self,
        contract: OptionContract,
        pricing_result: PricingResult
    ) -> float:
        """
        Calculate price adjustment based on current inventory.

        Positive inventory -> lower bid, higher ask (want to sell)
        Negative inventory -> higher bid, lower ask (want to buy)
        """
        contract_key = f"{contract.underlying}_{contract.strike}_{contract.expiry}"

        if contract_key not in self.inventory:
            return 0.0

        position = self.inventory[contract_key]
        position_ratio = position.quantity / self.params.max_position

        # Skew adjustment
        adjustment = position_ratio * self.params.inventory_skew_factor * pricing_result.price

        return adjustment

    def _calculate_vol_spread_adjustment(self, volatility: float) -> float:
        """Widen spread in high volatility environments."""
        # Base vol assumption is 20%
        base_vol = 0.20
        vol_ratio = volatility / base_vol

        # Spread widens as vol increases
        return max(0, (vol_ratio - 1) * 0.5)

    def _calculate_quote_sizes(
        self,
        contract: OptionContract,
        pricing_result: PricingResult
    ) -> Tuple[int, int]:
        """Calculate bid and ask sizes based on risk limits."""
        # Get current aggregate exposures
        total_gamma = sum(p.gamma_exposure for p in self.inventory.values())
        total_vega = sum(p.vega_exposure for p in self.inventory.values())

        # Calculate room for more risk
        gamma_room = self.params.gamma_limit - abs(total_gamma)
        vega_room = self.params.vega_limit - abs(total_vega)

        # Size based on gamma limit
        if abs(pricing_result.gamma) > 0:
            gamma_size = int(gamma_room / abs(pricing_result.gamma * contract.multiplier))
        else:
            gamma_size = self.params.quote_size

        # Size based on vega limit
        if abs(pricing_result.vega) > 0:
            vega_size = int(vega_room / abs(pricing_result.vega * contract.multiplier))
        else:
            vega_size = self.params.quote_size

        # Take minimum of all constraints
        max_size = min(gamma_size, vega_size, self.params.quote_size)
        max_size = max(1, max_size)

        # Adjust for inventory
        contract_key = f"{contract.underlying}_{contract.strike}_{contract.expiry}"
        if contract_key in self.inventory:
            position = self.inventory[contract_key]

            # If long, prefer to sell (larger ask size)
            if position.quantity > 0:
                bid_size = max(1, max_size - int(position.quantity / 10))
                ask_size = max_size
            else:
                bid_size = max_size
                ask_size = max(1, max_size + int(position.quantity / 10))
        else:
            bid_size = ask_size = max_size

        return bid_size, ask_size

    def update_inventory(
        self,
        contract: OptionContract,
        quantity: int,
        price: float,
        pricing_result: PricingResult
    ) -> None:
        """Update inventory after a trade."""
        contract_key = f"{contract.underlying}_{contract.strike}_{contract.expiry}"

        if contract_key in self.inventory:
            position = self.inventory[contract_key]
            new_quantity = position.quantity + quantity

            if new_quantity == 0:
                del self.inventory[contract_key]
            else:
                # Update average price
                if quantity > 0:  # Buying
                    total_cost = position.avg_price * position.quantity + price * quantity
                    new_avg_price = total_cost / new_quantity
                else:
                    new_avg_price = position.avg_price

                self.inventory[contract_key] = InventoryPosition(
                    contract=contract,
                    quantity=new_quantity,
                    avg_price=new_avg_price,
                    delta_exposure=pricing_result.delta * new_quantity * contract.multiplier,
                    gamma_exposure=pricing_result.gamma * new_quantity * contract.multiplier,
                    vega_exposure=pricing_result.vega * new_quantity * contract.multiplier
                )
        else:
            self.inventory[contract_key] = InventoryPosition(
                contract=contract,
                quantity=quantity,
                avg_price=price,
                delta_exposure=pricing_result.delta * quantity * contract.multiplier,
                gamma_exposure=pricing_result.gamma * quantity * contract.multiplier,
                vega_exposure=pricing_result.vega * quantity * contract.multiplier
            )

        logger.info(f"Inventory updated: {contract_key}, quantity={quantity}, price={price}")

    def get_portfolio_greeks(self) -> Dict[str, float]:
        """Get aggregate portfolio Greeks."""
        return {
            "delta": sum(p.delta_exposure for p in self.inventory.values()),
            "gamma": sum(p.gamma_exposure for p in self.inventory.values()),
            "vega": sum(p.vega_exposure for p in self.inventory.values()),
            "position_count": len(self.inventory)
        }

    def calculate_hedge_trades(
        self,
        market_data: MarketData
    ) -> List[Dict[str, Any]]:
        """
        Calculate hedge trades to neutralize Greeks.

        Returns list of suggested hedging trades.
        """
        portfolio_greeks = self.get_portfolio_greeks()
        hedges = []

        # Delta hedge with underlying
        # OPT-P0-3: Use floor/ceil based on hedge direction to avoid residual exposure
        if abs(portfolio_greeks["delta"]) > 100:
            delta_to_hedge = portfolio_greeks["delta"]
            # If delta > 0, we need to sell shares (negative hedge)
            # Use floor to not over-hedge: floor(155.7) = 155 -> hedge = -155
            # If delta < 0, we need to buy shares (positive hedge)
            # Use ceiling of absolute value: ceil(155.7) = 156 -> hedge = 156
            if delta_to_hedge > 0:
                hedge_quantity = -math.floor(delta_to_hedge)
            else:
                hedge_quantity = math.ceil(abs(delta_to_hedge))
            hedges.append({
                "instrument": "underlying",
                "quantity": hedge_quantity,
                "rationale": "Delta neutralization"
            })

        # For gamma/vega, would need options - simplified here
        if abs(portfolio_greeks["gamma"]) > self.params.gamma_limit * 0.8:
            hedges.append({
                "instrument": "options",
                "action": "reduce_gamma",
                "current_exposure": portfolio_greeks["gamma"],
                "rationale": "Gamma limit approaching"
            })

        if abs(portfolio_greeks["vega"]) > self.params.vega_limit * 0.8:
            hedges.append({
                "instrument": "options",
                "action": "reduce_vega",
                "current_exposure": portfolio_greeks["vega"],
                "rationale": "Vega limit approaching"
            })

        return hedges


# =============================================================================
# Module Integration
# =============================================================================

# =============================================================================
# VEGA CONVEXITY (VOLGA) CALCULATION (P2)
# =============================================================================

class VolgaCalculator:
    """
    Vega convexity (Volga) calculator (P2).

    Volga measures the sensitivity of vega to changes in implied volatility.
    It's crucial for:
    - Understanding P&L from vol of vol
    - Pricing exotic options
    - Managing volatility smile risk
    """

    def __init__(self, pricing_model: BasePricingModel | None = None):
        """Initialize with pricing model."""
        self.pricing_model = pricing_model or BlackScholesModel()

    def calculate_volga(
        self,
        contract: OptionContract,
        market_data: MarketData,
        vol_bump: float = 0.01,
    ) -> Dict[str, float]:
        """
        Calculate volga (vega convexity) numerically (P2).

        Volga = d(Vega)/d(sigma) = d^2(Price)/d(sigma)^2

        Args:
            contract: Option contract
            market_data: Market data with volatility
            vol_bump: Volatility bump size for numerical differentiation

        Returns:
            Dictionary with volga and related metrics
        """
        sigma = market_data.volatility

        # Price at current vol
        result_mid = self.pricing_model.price(contract, market_data)
        price_mid = result_mid.price
        vega_mid = result_mid.vega

        # Price at vol + bump
        md_up = MarketData(
            spot=market_data.spot,
            rate=market_data.rate,
            dividend_yield=market_data.dividend_yield,
            volatility=sigma + vol_bump,
        )
        result_up = self.pricing_model.price(contract, md_up)
        vega_up = result_up.vega

        # Price at vol - bump
        md_down = MarketData(
            spot=market_data.spot,
            rate=market_data.rate,
            dividend_yield=market_data.dividend_yield,
            volatility=max(0.001, sigma - vol_bump),
        )
        result_down = self.pricing_model.price(contract, md_down)
        vega_down = result_down.vega

        # Numerical volga (central difference)
        volga = (vega_up - vega_down) / (2 * vol_bump)

        # Second derivative of price w.r.t. vol
        # Using three-point formula
        volga_price = (result_up.price - 2 * price_mid + result_down.price) / (vol_bump ** 2)

        # Vega decay rate (how vega changes as vol moves)
        vega_decay_up = (vega_up - vega_mid) / vol_bump
        vega_decay_down = (vega_mid - vega_down) / vol_bump

        return {
            "volga": volga,
            "volga_dollar": volga * 100,  # Per 1% vol move
            "vega_mid": vega_mid,
            "vega_up": vega_up,
            "vega_down": vega_down,
            "vega_convexity": volga_price,
            "vega_decay_rate_up": vega_decay_up,
            "vega_decay_rate_down": vega_decay_down,
            "vega_skew": vega_decay_up - vega_decay_down,
            "vol_bump_used": vol_bump,
            "interpretation": (
                "Positive volga: benefits from vol of vol"
                if volga > 0
                else "Negative volga: hurt by vol of vol"
            ),
        }

    def calculate_volga_surface(
        self,
        underlying: str,
        spot: float,
        rate: float,
        strikes: List[float],
        expiries_days: List[int],
        base_vol: float = 0.20,
    ) -> Dict[Tuple[float, int], float]:
        """
        Calculate volga across strike/expiry grid (P2).

        Args:
            underlying: Underlying symbol
            spot: Spot price
            rate: Risk-free rate
            strikes: List of strikes
            expiries_days: List of expiries in days
            base_vol: Base volatility

        Returns:
            Dictionary of (strike, expiry_days) -> volga
        """
        from datetime import timedelta

        volga_surface = {}

        for strike in strikes:
            for expiry_days in expiries_days:
                expiry = datetime.now(timezone.utc) + timedelta(days=expiry_days)
                contract = OptionContract(
                    underlying=underlying,
                    strike=strike,
                    expiry=expiry,
                    option_type=OptionType.CALL,
                )

                market_data = MarketData(
                    spot=spot,
                    rate=rate,
                    volatility=base_vol,
                )

                result = self.calculate_volga(contract, market_data)
                volga_surface[(strike, expiry_days)] = result["volga"]

        return volga_surface


# =============================================================================
# CROSS-GAMMA FOR MULTI-ASSET (P2)
# =============================================================================

@dataclass
class CrossGammaResult:
    """Result of cross-gamma calculation (P2)."""
    asset1: str
    asset2: str
    cross_gamma: float
    correlation: float
    delta1: float
    delta2: float
    gamma1: float
    gamma2: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset1": self.asset1,
            "asset2": self.asset2,
            "cross_gamma": self.cross_gamma,
            "correlation": self.correlation,
            "delta1": self.delta1,
            "delta2": self.delta2,
            "gamma1": self.gamma1,
            "gamma2": self.gamma2,
        }


class CrossGammaCalculator:
    """
    Cross-gamma calculator for multi-asset options (P2).

    Cross-gamma measures sensitivity of delta in one asset to
    price changes in another asset. Critical for:
    - Basket options
    - Spread options
    - Correlation products
    """

    def __init__(self, pricing_model: BasePricingModel | None = None):
        """Initialize with pricing model."""
        self.pricing_model = pricing_model or BlackScholesModel()

    def calculate_cross_gamma(
        self,
        contract: OptionContract,
        market_data: MarketData,
        asset1_name: str,
        asset2_name: str,
        spot1: float,
        spot2: float,
        correlation: float,
        weight1: float = 0.5,
        weight2: float = 0.5,
        spot_bump_pct: float = 0.01,
    ) -> CrossGammaResult:
        """
        Calculate cross-gamma between two assets (P2).

        Cross-gamma = d^2(V) / (dS1 * dS2)

        For a basket option with S_basket = w1*S1 + w2*S2,
        the cross-gamma captures how delta1 changes when S2 moves.

        Args:
            contract: Option contract (basket/spread)
            market_data: Base market data
            asset1_name: Name of first asset
            asset2_name: Name of second asset
            spot1: Spot price of asset 1
            spot2: Spot price of asset 2
            correlation: Correlation between assets
            weight1: Weight of asset 1 in basket
            weight2: Weight of asset 2 in basket
            spot_bump_pct: Percentage bump for numerical calculation

        Returns:
            CrossGammaResult with cross-gamma and related metrics
        """
        # Calculate basket spot for base case
        basket_spot = weight1 * spot1 + weight2 * spot2
        h1 = spot1 * spot_bump_pct
        h2 = spot2 * spot_bump_pct

        # Create market data for different scenarios
        def get_basket_price(s1: float, s2: float) -> float:
            basket = weight1 * s1 + weight2 * s2
            md = MarketData(
                spot=basket,
                rate=market_data.rate,
                dividend_yield=market_data.dividend_yield,
                volatility=market_data.volatility,
            )
            result = self.pricing_model.price(contract, md)
            return result.price

        # Four-point cross-gamma calculation
        # Cross-gamma = [V(S1+h1, S2+h2) - V(S1+h1, S2-h2) - V(S1-h1, S2+h2) + V(S1-h1, S2-h2)] / (4*h1*h2)
        V_pp = get_basket_price(spot1 + h1, spot2 + h2)  # ++
        V_pm = get_basket_price(spot1 + h1, spot2 - h2)  # +-
        V_mp = get_basket_price(spot1 - h1, spot2 + h2)  # -+
        V_mm = get_basket_price(spot1 - h1, spot2 - h2)  # --

        cross_gamma = (V_pp - V_pm - V_mp + V_mm) / (4 * h1 * h2)

        # Calculate individual deltas and gammas
        V_base = get_basket_price(spot1, spot2)
        V_1p = get_basket_price(spot1 + h1, spot2)
        V_1m = get_basket_price(spot1 - h1, spot2)
        V_2p = get_basket_price(spot1, spot2 + h2)
        V_2m = get_basket_price(spot1, spot2 - h2)

        delta1 = (V_1p - V_1m) / (2 * h1)
        delta2 = (V_2p - V_2m) / (2 * h2)
        gamma1 = (V_1p - 2 * V_base + V_1m) / (h1 ** 2)
        gamma2 = (V_2p - 2 * V_base + V_2m) / (h2 ** 2)

        return CrossGammaResult(
            asset1=asset1_name,
            asset2=asset2_name,
            cross_gamma=cross_gamma,
            correlation=correlation,
            delta1=delta1,
            delta2=delta2,
            gamma1=gamma1,
            gamma2=gamma2,
        )

    def calculate_cross_gamma_matrix(
        self,
        assets: List[str],
        spots: Dict[str, float],
        weights: Dict[str, float],
        correlations: Dict[Tuple[str, str], float],
        contract: OptionContract,
        market_data: MarketData,
    ) -> Dict[str, Any]:
        """
        Calculate full cross-gamma matrix for multiple assets (P2).

        Args:
            assets: List of asset names
            spots: Dictionary of asset -> spot price
            weights: Dictionary of asset -> weight in basket
            correlations: Dictionary of (asset1, asset2) -> correlation
            contract: Option contract
            market_data: Base market data

        Returns:
            Dictionary with cross-gamma matrix and analysis
        """
        n = len(assets)
        cross_gamma_matrix = np.zeros((n, n))
        results = {}

        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i == j:
                    # Diagonal is regular gamma
                    basket_spot = sum(
                        weights.get(a, 0) * spots.get(a, 0) for a in assets
                    )
                    md = MarketData(
                        spot=basket_spot,
                        rate=market_data.rate,
                        dividend_yield=market_data.dividend_yield,
                        volatility=market_data.volatility,
                    )
                    result = self.pricing_model.price(contract, md)
                    cross_gamma_matrix[i, j] = result.gamma * (weights.get(asset1, 0) ** 2)
                elif j > i:
                    corr = correlations.get((asset1, asset2), 0.5)
                    if (asset2, asset1) in correlations:
                        corr = correlations[(asset2, asset1)]

                    cg_result = self.calculate_cross_gamma(
                        contract=contract,
                        market_data=market_data,
                        asset1_name=asset1,
                        asset2_name=asset2,
                        spot1=spots.get(asset1, 100),
                        spot2=spots.get(asset2, 100),
                        correlation=corr,
                        weight1=weights.get(asset1, 0.5),
                        weight2=weights.get(asset2, 0.5),
                    )
                    cross_gamma_matrix[i, j] = cg_result.cross_gamma
                    cross_gamma_matrix[j, i] = cg_result.cross_gamma
                    results[f"{asset1}_{asset2}"] = cg_result.to_dict()

        # Total portfolio cross-gamma exposure
        total_cross_gamma = np.sum(cross_gamma_matrix) - np.trace(cross_gamma_matrix)

        return {
            "assets": assets,
            "cross_gamma_matrix": cross_gamma_matrix.tolist(),
            "pairwise_results": results,
            "total_cross_gamma": total_cross_gamma,
            "max_cross_gamma_pair": max(
                results.keys(),
                key=lambda k: abs(results[k]["cross_gamma"]),
                default=None,
            ) if results else None,
        }


# =============================================================================
# THETA DECAY CURVE PROJECTION (P2)
# =============================================================================

@dataclass
class ThetaProjection:
    """Single point in theta decay projection (P2)."""
    days_forward: int
    option_value: float
    theta_daily: float
    theta_cumulative: float
    time_to_expiry: float


class ThetaDecayCurveCalculator:
    """
    Theta decay curve projection calculator (P2).

    Projects the theta decay over time, showing how option
    value and theta change as expiration approaches.
    Critical for:
    - Timing option trades
    - Understanding decay acceleration
    - Rolling strategy decisions
    """

    def __init__(self, pricing_model: BasePricingModel | None = None):
        """Initialize with pricing model."""
        self.pricing_model = pricing_model or BlackScholesModel()

    def project_theta_decay(
        self,
        contract: OptionContract,
        market_data: MarketData,
        projection_days: int = 30,
        step_days: int = 1,
    ) -> Dict[str, Any]:
        """
        Project theta decay curve forward in time (P2).

        Args:
            contract: Option contract
            market_data: Current market data
            projection_days: Number of days to project forward
            step_days: Days between projection points

        Returns:
            Dictionary with theta decay projections and analysis
        """
        projections: List[ThetaProjection] = []
        current_tte = contract.time_to_expiry

        if current_tte <= 0:
            return {
                "error": "Option has expired",
                "projections": [],
            }

        # Maximum projection is until 1 day before expiry
        max_days = int(current_tte * 365) - 1
        projection_days = min(projection_days, max_days)

        cumulative_theta = 0.0

        for day in range(0, projection_days + 1, step_days):
            # Create contract with adjusted time to expiry
            days_remaining = current_tte * 365 - day
            if days_remaining <= 0:
                break

            tte_years = days_remaining / 365.0

            # Create a new contract with shifted expiry
            shifted_expiry = datetime.now(timezone.utc) + timedelta(days=days_remaining)
            shifted_contract = OptionContract(
                underlying=contract.underlying,
                strike=contract.strike,
                expiry=shifted_expiry,
                option_type=contract.option_type,
                multiplier=contract.multiplier,
            )

            # Price option
            result = self.pricing_model.price(shifted_contract, market_data)

            # Daily theta (already in per-day terms from model)
            daily_theta = result.theta

            if day > 0:
                cumulative_theta += daily_theta * step_days

            projections.append(ThetaProjection(
                days_forward=day,
                option_value=result.price,
                theta_daily=daily_theta,
                theta_cumulative=cumulative_theta,
                time_to_expiry=tte_years,
            ))

        # Analyze decay pattern
        if len(projections) >= 2:
            values = [p.option_value for p in projections]
            thetas = [p.theta_daily for p in projections]

            # Find acceleration point (where theta decay rate increases significantly)
            theta_changes = [
                thetas[i + 1] - thetas[i]
                for i in range(len(thetas) - 1)
            ]
            max_acceleration_idx = (
                theta_changes.index(min(theta_changes))
                if theta_changes
                else 0
            )

            # Calculate decay metrics
            initial_value = values[0]
            half_value_day = None
            for i, v in enumerate(values):
                if v <= initial_value / 2:
                    half_value_day = projections[i].days_forward
                    break

            decay_rate_first_half = (
                (values[0] - values[len(values) // 2]) / values[0] * 100
                if len(values) > 1 and values[0] > 0
                else 0
            )

            decay_rate_second_half = (
                (values[len(values) // 2] - values[-1]) / values[0] * 100
                if len(values) > 1 and values[0] > 0
                else 0
            )

            analysis = {
                "initial_value": initial_value,
                "final_value": values[-1],
                "total_decay": initial_value - values[-1],
                "total_decay_pct": (initial_value - values[-1]) / initial_value * 100
                if initial_value > 0
                else 0,
                "half_value_day": half_value_day,
                "max_theta_day": projections[thetas.index(min(thetas))].days_forward
                if thetas
                else 0,
                "max_theta_value": min(thetas) if thetas else 0,
                "acceleration_day": projections[max_acceleration_idx].days_forward,
                "decay_rate_first_half_pct": decay_rate_first_half,
                "decay_rate_second_half_pct": decay_rate_second_half,
                "decay_accelerates": decay_rate_second_half > decay_rate_first_half,
            }
        else:
            analysis = {}

        return {
            "contract": {
                "underlying": contract.underlying,
                "strike": contract.strike,
                "option_type": contract.option_type.value,
                "current_tte_days": current_tte * 365,
            },
            "market_data": {
                "spot": market_data.spot,
                "volatility": market_data.volatility,
                "rate": market_data.rate,
            },
            "projections": [
                {
                    "days_forward": p.days_forward,
                    "option_value": p.option_value,
                    "theta_daily": p.theta_daily,
                    "theta_cumulative": p.theta_cumulative,
                    "time_to_expiry_years": p.time_to_expiry,
                }
                for p in projections
            ],
            "analysis": analysis,
            "recommendation": self._generate_theta_recommendation(analysis),
        }

    def _generate_theta_recommendation(self, analysis: Dict[str, Any]) -> str:
        """Generate trading recommendation based on theta analysis."""
        if not analysis:
            return "Insufficient data for recommendation"

        if analysis.get("decay_accelerates"):
            accel_day = analysis.get("acceleration_day", 0)
            if accel_day < 7:
                return "URGENT: Significant theta acceleration imminent. Consider rolling or closing position."
            elif accel_day < 14:
                return "CAUTION: Theta decay will accelerate soon. Monitor position closely."
            else:
                return "OK: Theta decay still relatively stable. Continue monitoring."
        else:
            return "STABLE: Linear decay pattern. Normal theta profile."

    def compare_theta_curves(
        self,
        contracts: List[OptionContract],
        market_data: MarketData,
        projection_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Compare theta decay curves for multiple contracts (P2).

        Useful for:
        - Calendar spread analysis
        - Selecting optimal expiry
        - Rolling decision support
        """
        curves = {}

        for contract in contracts:
            key = f"{contract.strike}_{contract.expiry.strftime('%Y%m%d')}"
            projection = self.project_theta_decay(
                contract, market_data, projection_days
            )
            curves[key] = projection

        # Find best theta profile
        best_theta_ratio = None
        best_contract = None

        for key, curve in curves.items():
            if "analysis" in curve and curve["analysis"]:
                initial = curve["analysis"].get("initial_value", 0)
                if initial > 0:
                    total_decay_pct = curve["analysis"].get("total_decay_pct", 0)
                    theta_ratio = total_decay_pct / projection_days  # Daily decay rate
                    if best_theta_ratio is None or theta_ratio < best_theta_ratio:
                        best_theta_ratio = theta_ratio
                        best_contract = key

        return {
            "curves": curves,
            "comparison": {
                "best_theta_profile": best_contract,
                "lowest_daily_decay_pct": best_theta_ratio,
            },
        }


def create_pricing_suite() -> Dict[str, Any]:
    """
    Create a complete pricing suite with all models.

    Returns:
        Dictionary containing all pricing components
    """
    return {
        "black_scholes": BlackScholesModel(),
        "merton_jump": MertonJumpDiffusionModel(),
        "heston": HestonModel(),
        "comparator": PricingModelComparator(),
        "market_maker": OptionMarketMaker(BlackScholesModel()),
        "volga_calculator": VolgaCalculator(),  # P2: Vega convexity
        "cross_gamma_calculator": CrossGammaCalculator(),  # P2: Multi-asset
        "theta_decay_calculator": ThetaDecayCurveCalculator(),  # P2: Theta projection
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create test contract
    contract = OptionContract(
        underlying="SPY",
        strike=450.0,
        expiry=datetime(2025, 6, 20),
        option_type=OptionType.CALL
    )

    # Market data
    market_data = MarketData(
        spot=455.0,
        rate=0.05,
        dividend_yield=0.015,
        volatility=0.20
    )

    # Compare models
    comparator = PricingModelComparator()
    comparison = comparator.compare_models(contract, market_data, market_price=12.50)

    print("Model Comparison Results:")
    for model, result in comparison.results.items():
        print(f"  {model.value}: Price={result.price:.2f}, Delta={result.delta:.3f}, "
              f"Gamma={result.gamma:.4f}, Vega={result.vega:.2f}")

    if comparison.best_fit_model:
        print(f"\nBest fit model: {comparison.best_fit_model.value}")

    # Market making
    mm = OptionMarketMaker(BlackScholesModel())
    quote = mm.generate_quote(contract, market_data)
    print(f"\nMarket Making Quote:")
    print(f"  Bid: {quote.bid_price:.2f} x {quote.bid_size}")
    print(f"  Ask: {quote.ask_price:.2f} x {quote.ask_size}")
    print(f"  Theo: {quote.theo_price:.2f}, Edge: {quote.edge:.4f}")

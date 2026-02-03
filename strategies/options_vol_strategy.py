"""
Options Volatility Strategy
===========================

Implements options and volatility-based trading logic.

MATURITY: BETA
--------------
Status: Comprehensive Greeks and volatility surface implementation
- [x] Black-Scholes pricing with dividends
- [x] Full Greeks calculation (delta, gamma, theta, vega, rho)
- [x] Implied volatility (Newton-Raphson)
- [x] IV percentile ranking
- [x] Vol surface construction (#O4)
- [x] Skew analysis (risk reversal, butterfly) (#O5)
- [x] Early exercise boundary (binomial) (#O3)
- [x] Vanna/Volga adjustments (#O11)
- [x] Option validation (#O2)
- [x] Spread strategies (verticals, iron condors) (#O7)
- [x] Pin risk detection (#O8)
- [x] Assignment risk calculation (#O9)
- [x] Gamma scalping support (#O10)
- [ ] SABR model (TODO)
- [ ] Local volatility surface (TODO)
- [ ] Variance swap replication (TODO)

Production Readiness:
- Unit tests: Good coverage for Greeks
- Validation: Option contract validation implemented
- Greeks bounds checking: Implemented

Use in production: WITH CAUTION
- IV solver may not converge for extreme values
- Early exercise uses binomial tree approximation
- Verify Greeks against broker values before trading
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm


logger = logging.getLogger(__name__)


class OptionValidationError(Exception):
    """Raised when option contract validation fails."""
    pass


@dataclass
class OptionData:
    """
    Option contract data with validation (#O2).

    Validates strike prices and expiration dates to ensure contract integrity.
    """
    symbol: str
    underlying: str
    strike: float
    expiry_days: int
    is_call: bool
    bid: float
    ask: float
    implied_vol: float
    delta: float
    gamma: float
    theta: float
    vega: float

    # Validation constants
    MIN_STRIKE: float = 0.01
    MAX_STRIKE: float = 1_000_000.0
    MIN_EXPIRY_DAYS: int = 0
    MAX_EXPIRY_DAYS: int = 3650  # ~10 years
    MAX_IMPLIED_VOL: float = 10.0  # 1000% vol

    def __post_init__(self):
        """Validate option contract data (#O2)."""
        self.validate()

    def validate(self) -> None:
        """
        Validate option contract parameters (#O2).

        Raises:
            OptionValidationError: If validation fails
        """
        errors = []

        # Validate strike price
        if self.strike <= 0:
            errors.append(f"Strike price must be positive: {self.strike}")
        elif self.strike < self.MIN_STRIKE:
            errors.append(f"Strike price {self.strike} below minimum {self.MIN_STRIKE}")
        elif self.strike > self.MAX_STRIKE:
            errors.append(f"Strike price {self.strike} above maximum {self.MAX_STRIKE}")

        # Validate expiry
        if self.expiry_days < self.MIN_EXPIRY_DAYS:
            errors.append(f"Expiry days cannot be negative: {self.expiry_days}")
        elif self.expiry_days > self.MAX_EXPIRY_DAYS:
            errors.append(f"Expiry days {self.expiry_days} exceeds maximum {self.MAX_EXPIRY_DAYS}")

        # Validate bid/ask
        if self.bid < 0:
            errors.append(f"Bid price cannot be negative: {self.bid}")
        if self.ask < 0:
            errors.append(f"Ask price cannot be negative: {self.ask}")
        if self.bid > self.ask and self.ask > 0:
            errors.append(f"Bid {self.bid} cannot exceed ask {self.ask}")

        # Validate implied volatility
        if self.implied_vol < 0:
            errors.append(f"Implied volatility cannot be negative: {self.implied_vol}")
        elif self.implied_vol > self.MAX_IMPLIED_VOL:
            errors.append(f"Implied volatility {self.implied_vol} exceeds maximum {self.MAX_IMPLIED_VOL}")

        # Validate Greeks bounds
        if self.is_call:
            if not -0.01 <= self.delta <= 1.01:
                errors.append(f"Call delta {self.delta} out of bounds [0, 1]")
        else:
            if not -1.01 <= self.delta <= 0.01:
                errors.append(f"Put delta {self.delta} out of bounds [-1, 0]")

        if self.gamma < 0:
            errors.append(f"Gamma cannot be negative: {self.gamma}")

        # Vega should be positive for long options
        if self.vega < 0:
            errors.append(f"Vega cannot be negative: {self.vega}")

        if errors:
            error_msg = f"Option validation failed for {self.symbol}: " + "; ".join(errors)
            logger.error(error_msg)
            raise OptionValidationError(error_msg)

    @property
    def is_expired(self) -> bool:
        """Check if option is expired."""
        return self.expiry_days <= 0

    @property
    def is_atm(self) -> bool:
        """Check if option is approximately at-the-money (within 2% of strike)."""
        # Note: This requires underlying price which isn't stored
        # Would need to compare with current underlying price
        return False  # Placeholder

    @property
    def mid_price(self) -> float:
        """Get mid-market price."""
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        """Get bid-ask spread as percentage of mid price."""
        mid = self.mid_price
        if mid <= 0:
            return float('inf')
        return (self.ask - self.bid) / mid

    @classmethod
    def create_validated(
        cls,
        symbol: str,
        underlying: str,
        strike: float,
        expiry_days: int,
        is_call: bool,
        bid: float,
        ask: float,
        implied_vol: float,
        delta: float,
        gamma: float,
        theta: float,
        vega: float,
    ) -> "OptionData":
        """
        Factory method to create validated option data.

        Returns:
            Validated OptionData instance

        Raises:
            OptionValidationError: If validation fails
        """
        return cls(
            symbol=symbol,
            underlying=underlying,
            strike=strike,
            expiry_days=expiry_days,
            is_call=is_call,
            bid=bid,
            ask=ask,
            implied_vol=implied_vol,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
        )


@dataclass
class VolSignal:
    """Volatility strategy signal."""
    underlying: str
    strategy: str  # "sell_strangle", "buy_straddle", "iron_condor", etc.
    direction: str  # "short_vol", "long_vol", "neutral"
    strength: float
    iv_percentile: float
    vol_premium: float
    legs: list[dict]  # Option legs for the trade
    # P1-13: Stop-loss for options strategies (typically % of premium or delta)
    max_loss_pct: float | None = None  # Max acceptable loss as % of premium
    stop_loss_underlying_move: float | None = None  # Underlying move that triggers stop


class OptionsVolStrategy:
    """
    Options Volatility Strategy Implementation.

    Implements:
    1. IV percentile ranking
    2. Volatility risk premium analysis
    3. Skew trading
    4. Term structure analysis

    TODO: Implement proper models:
    - SABR for smile dynamics
    - Local volatility surface
    - Variance swap replication
    - Dispersion trading
    """

    def __init__(self, config: dict[str, Any]):
        self._iv_percentile_threshold = config.get("iv_percentile_threshold", 80)
        self._min_dte = config.get("min_dte", 7)
        self._max_dte = config.get("max_dte", 45)
        self._delta_range = config.get("delta_range", [0.20, 0.40])
        self._min_vol_premium = config.get("min_vol_premium", 0.02)

    def black_scholes_price(
        self,
        S: float,  # Spot price
        K: float,  # Strike
        T: float,  # Time to expiry (years)
        r: float,  # Risk-free rate
        sigma: float,  # Volatility
        is_call: bool,
        q: float = 0.0,  # Continuous dividend yield
    ) -> float:
        """
        Calculate Black-Scholes option price with continuous dividend yield.

        The Black-Scholes model (1973) provides a closed-form solution for
        European option prices under the following assumptions:
        - Log-normal distribution of stock prices
        - Constant volatility and interest rates
        - No transaction costs or taxes
        - Continuous trading possible
        - No arbitrage opportunities

        Black-Scholes Formula (with dividend yield q):
            Call: C = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
            Put:  P = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)

        Where:
            d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)

            N(x) = cumulative standard normal distribution function
            e^(-rT) = discount factor for strike payment
            e^(-qT) = discount factor for dividend-paying stock

        Interpretation:
            - N(d2) is approximately the probability the option expires ITM
              (risk-neutral probability of exercise)
            - N(d1) is delta-related: probability-weighted exercise value

        Args:
            S: Current spot price of the underlying asset
            K: Strike (exercise) price
            T: Time to expiry in years (e.g., 30 days = 30/365 = 0.0822)
            r: Risk-free interest rate (annualized decimal, e.g., 0.05 for 5%)
            sigma: Implied volatility (annualized decimal, e.g., 0.20 for 20%)
            is_call: True for call option, False for put option
            q: Continuous dividend yield (annualized, e.g., 0.02 for 2%)

        Returns:
            Theoretical option price in the same currency as S and K

        Example:
            S=100, K=100, T=0.25 (3 months), r=0.05, sigma=0.20, call
            d1 = [ln(1) + (0.05 + 0.02) * 0.25] / (0.20 * 0.5) = 0.175
            d2 = 0.175 - 0.10 = 0.075
            C = 100 * N(0.175) - 100 * e^(-0.0125) * N(0.075) = ~5.88

        Note:
            For stocks with discrete dividends, convert to continuous yield:
            q = -ln(1 - PV(dividends)/S) / T
            where PV(dividends) is the present value of expected dividends.
        """
        if T <= 0 or sigma <= 0:
            # At or past expiry: return intrinsic value only
            return max(0, S - K) if is_call else max(0, K - S)

        sqrt_T = math.sqrt(T)

        # Calculate d1 and d2 (key Black-Scholes parameters)
        # d1 incorporates: moneyness (ln(S/K)), cost of carry (r-q), and volatility (sigma^2/2)
        d1 = (math.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * sqrt_T)
        # d2 = d1 - sigma*sqrt(T) represents the probability of exercise
        d2 = d1 - sigma * sqrt_T

        # Discount factors for present value calculations
        discount_factor = math.exp(-r * T)  # PV of $1 received at expiry
        dividend_discount = math.exp(-q * T)  # Adjustment for dividend-paying assets

        # Apply Black-Scholes formula
        if is_call:
            # Call = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
            price = S * dividend_discount * norm.cdf(d1) - K * discount_factor * norm.cdf(d2)
        else:
            # Put = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
            # Using put-call parity: P = C - S*e^(-qT) + K*e^(-rT)
            price = K * discount_factor * norm.cdf(-d2) - S * dividend_discount * norm.cdf(-d1)

        return price

    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        is_call: bool,
        q: float = 0.0,  # Continuous dividend yield
    ) -> dict[str, float]:
        """
        Calculate option Greeks (sensitivities) with dividend yield.

        Greeks measure how option prices change with respect to various factors.
        They are essential for hedging and risk management.

        First-Order Greeks (price sensitivities):
            Delta: dV/dS - sensitivity to underlying price
            Vega:  dV/d(sigma) - sensitivity to volatility
            Theta: dV/dt - time decay (negative for long options)
            Rho:   dV/dr - sensitivity to interest rate

        Second-Order Greek:
            Gamma: d(Delta)/dS = d^2V/dS^2 - rate of delta change

        Formulas (Black-Scholes with continuous dividend yield q):
            d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)

            Delta_call = e^(-qT) * N(d1)
            Delta_put  = -e^(-qT) * N(-d1)

            Gamma = e^(-qT) * N'(d1) / (S * sigma * sqrt(T))

            Theta = -e^(-qT) * S * N'(d1) * sigma / (2 * sqrt(T))
                    - r * K * e^(-rT) * N(d2)  [call]
                    + q * S * e^(-qT) * N(d1)  [call]

            Vega = S * e^(-qT) * sqrt(T) * N'(d1)

            Rho_call = K * T * e^(-rT) * N(d2)
            Rho_put  = -K * T * e^(-rT) * N(-d2)

        Where N(x) is the standard normal CDF and N'(x) is the PDF.

        Args:
            S: Spot price of the underlying
            K: Strike price
            T: Time to expiry in years (e.g., 30 days = 30/365)
            r: Risk-free interest rate (annualized, e.g., 0.05 for 5%)
            sigma: Implied volatility (annualized, e.g., 0.20 for 20%)
            is_call: True for call option, False for put option
            q: Continuous dividend yield (annualized, e.g., 0.02 for 2%)

        Returns:
            Dictionary with:
                delta: Change in option price per $1 move in underlying
                gamma: Change in delta per $1 move in underlying
                theta: Daily time decay in dollars (negative for long options)
                vega: Change in price per 1% (100bp) move in volatility
                rho: Change in price per 1% (100bp) move in interest rate
        """
        # Handle edge cases: expired or zero volatility options
        if T <= 0 or sigma <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

        sqrt_T = math.sqrt(T)

        # Calculate d1 and d2 (fundamental Black-Scholes parameters)
        # d1 represents the moneyness adjusted for drift and volatility
        d1 = (math.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * sqrt_T)
        # d2 = d1 - sigma*sqrt(T), represents the probability of exercise
        d2 = d1 - sigma * sqrt_T

        # Discount factors for present value calculations
        discount_factor = math.exp(-r * T)  # For discounting strike
        dividend_discount = math.exp(-q * T)  # For dividend-paying assets

        # DELTA: Sensitivity of option price to underlying price
        # Call delta is in [0, 1], Put delta is in [-1, 0]
        # For dividend-paying assets, multiply by e^(-qT)
        if is_call:
            delta = dividend_discount * norm.cdf(d1)
        else:
            # Put-call parity: delta_put = delta_call - e^(-qT)
            delta = -dividend_discount * norm.cdf(-d1)

        # GAMMA: Rate of change of delta (curvature)
        # Gamma is always positive for long options
        # Highest for ATM options near expiry
        gamma = dividend_discount * norm.pdf(d1) / (S * sigma * sqrt_T)

        # THETA: Time decay (value lost per day)
        # Composed of three terms:
        # Term 1: Pure time decay (always negative for long options)
        theta_term1 = -S * dividend_discount * norm.pdf(d1) * sigma / (2 * sqrt_T)
        if is_call:
            # Term 2: Interest cost of holding strike (negative for calls)
            theta_term2 = -r * K * discount_factor * norm.cdf(d2)
            # Term 3: Dividend benefit for call holders (positive)
            theta_term3 = q * S * dividend_discount * norm.cdf(d1)
        else:
            # Term 2: Interest benefit of put (positive for puts)
            theta_term2 = r * K * discount_factor * norm.cdf(-d2)
            # Term 3: Dividend cost for put holders (negative)
            theta_term3 = -q * S * dividend_discount * norm.cdf(-d1)
        # Convert to per-day (annual theta / 365)
        theta = (theta_term1 + theta_term2 + theta_term3) / 365

        # VEGA: Sensitivity to volatility
        # Scaled to show P&L per 1% (100 basis point) vol change
        # Always positive for long options
        vega = S * dividend_discount * sqrt_T * norm.pdf(d1) / 100

        # RHO: Sensitivity to interest rate
        # Scaled to show P&L per 1% (100 basis point) rate change
        # Calls have positive rho, puts have negative rho
        if is_call:
            rho = K * T * discount_factor * norm.cdf(d2) / 100
        else:
            rho = -K * T * discount_factor * norm.cdf(-d2) / 100

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
        }

    def implied_volatility(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        is_call: bool,
        q: float = 0.0,  # Continuous dividend yield
        precision: float = 0.0001,
        max_iterations: int = 100,
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.

        Implied volatility (IV) is the volatility value that, when input into
        Black-Scholes, produces the observed market price. It represents the
        market's expectation of future volatility.

        Newton-Raphson Method:
            Given a function f(x) = 0, iterate:
            x_{n+1} = x_n - f(x_n) / f'(x_n)

            For IV: f(sigma) = BS_price(sigma) - market_price = 0
            Derivative: f'(sigma) = vega = dPrice/dSigma

            Therefore: sigma_{n+1} = sigma_n - (BS_price - market_price) / vega

        Convergence:
            Newton-Raphson converges quadratically when the initial guess
            is close to the solution. Convergence issues can occur for:
            - Deep ITM/OTM options (low vega)
            - Very short-dated options (rapid delta change)
            - Options trading below intrinsic value

        Args:
            price: Observed market price of the option
            S: Current spot price of underlying
            K: Strike price
            T: Time to expiry in years
            r: Risk-free interest rate (annualized)
            is_call: True for call, False for put
            q: Continuous dividend yield (annualized)
            precision: Price precision for convergence (default: $0.0001)
            max_iterations: Maximum Newton-Raphson iterations (default: 100)

        Returns:
            Implied volatility as decimal (e.g., 0.25 for 25%)
            Bounded between 0.01 (1%) and 2.0 (200%)

        Note:
            For production use, consider Brent's method or bisection with bounds
            for more robust convergence on extreme options.
        """
        # Initial guess: 20% volatility is reasonable for most equity options
        sigma = 0.2

        for _ in range(max_iterations):
            # Calculate Black-Scholes price at current sigma guess
            bs_price = self.black_scholes_price(S, K, T, r, sigma, is_call, q)

            # Get vega (sensitivity of price to volatility)
            # Multiply by 100 because our vega is scaled for 1% vol change
            vega = self.calculate_greeks(S, K, T, r, sigma, is_call, q)["vega"] * 100

            # Check for near-zero vega (convergence issues)
            # This happens for deep ITM/OTM or very short-dated options
            if abs(vega) < 1e-8:
                break

            # Newton-Raphson update: sigma_new = sigma - f(sigma) / f'(sigma)
            # Where f(sigma) = BS_price - market_price and f'(sigma) = vega
            sigma = sigma - (bs_price - price) / vega

            # Prevent sigma from going negative (volatility must be positive)
            if sigma <= 0:
                sigma = 0.01

            # Check for convergence (price difference within tolerance)
            if abs(bs_price - price) < precision:
                break

        # Bound the result to reasonable volatility range
        # IV below 1% or above 200% is typically invalid/suspicious
        return max(0.01, min(2.0, sigma))

    def calculate_iv_percentile(
        self,
        current_iv: float,
        iv_history: list[float],
    ) -> float:
        """
        Calculate IV percentile rank.
        """
        if not iv_history:
            return 50.0

        percentile = sum(1 for iv in iv_history if iv < current_iv) / len(iv_history) * 100
        return percentile

    def analyze_vol_surface(
        self,
        options: list[OptionData],
        spot_price: float,
    ) -> dict[str, Any]:
        """
        Analyze volatility surface characteristics.

        Returns analysis of:
        - ATM IV
        - Skew (25 delta put - 25 delta call IV)
        - Term structure
        """
        if not options:
            return {}

        # Group by expiry
        by_expiry: dict[int, list[OptionData]] = {}
        for opt in options:
            if opt.expiry_days not in by_expiry:
                by_expiry[opt.expiry_days] = []
            by_expiry[opt.expiry_days].append(opt)

        analysis = {
            "atm_iv_by_expiry": {},
            "skew_by_expiry": {},
            "term_structure": [],
        }

        for expiry, opts in sorted(by_expiry.items()):
            # Find ATM
            atm_opts = [o for o in opts if abs(o.strike - spot_price) / spot_price < 0.02]
            if atm_opts:
                atm_iv = np.mean([o.implied_vol for o in atm_opts])
                analysis["atm_iv_by_expiry"][expiry] = atm_iv
                analysis["term_structure"].append((expiry, atm_iv))

            # Calculate skew (25 delta)
            puts = [o for o in opts if not o.is_call and -0.30 < o.delta < -0.20]
            calls = [o for o in opts if o.is_call and 0.20 < o.delta < 0.30]

            if puts and calls:
                put_iv = np.mean([o.implied_vol for o in puts])
                call_iv = np.mean([o.implied_vol for o in calls])
                skew = put_iv - call_iv
                analysis["skew_by_expiry"][expiry] = skew

        return analysis

    def generate_signal(
        self,
        underlying: str,
        current_iv: float,
        iv_history: list[float],
        realized_vol: float,
        options: list[OptionData] | None = None,
    ) -> VolSignal | None:
        """
        Generate volatility trading signal.
        """
        iv_percentile = self.calculate_iv_percentile(current_iv, iv_history)
        vol_premium = current_iv - realized_vol

        # High IV percentile - sell premium
        if iv_percentile > self._iv_percentile_threshold:
            if vol_premium > self._min_vol_premium:
                return VolSignal(
                    underlying=underlying,
                    strategy="sell_strangle",
                    direction="short_vol",
                    strength=0.7,
                    iv_percentile=iv_percentile,
                    vol_premium=vol_premium,
                    legs=[
                        {"type": "put", "delta": -0.20, "action": "sell"},
                        {"type": "call", "delta": 0.20, "action": "sell"},
                    ],
                )

        # Low IV percentile - buy premium
        elif iv_percentile < (100 - self._iv_percentile_threshold):
            return VolSignal(
                underlying=underlying,
                strategy="buy_straddle",
                direction="long_vol",
                strength=0.5,
                iv_percentile=iv_percentile,
                vol_premium=vol_premium,
                legs=[
                    {"type": "put", "delta": -0.50, "action": "buy"},
                    {"type": "call", "delta": 0.50, "action": "buy"},
                ],
            )

        return None

    # =========================================================================
    # AMERICAN OPTION EARLY EXERCISE MODELING (#O3)
    # =========================================================================

    def calculate_early_exercise_boundary(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        is_call: bool,
        q: float = 0.0,
        n_steps: int = 100
    ) -> dict:
        """
        Calculate early exercise boundary for American options (#O3).

        Uses binomial tree approximation for early exercise premium.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            is_call: True for call, False for put
            q: Dividend yield
            n_steps: Number of time steps in binomial tree

        Returns:
            Early exercise analysis including boundary and premium
        """
        if T <= 0:
            intrinsic = max(0, S - K) if is_call else max(0, K - S)
            return {
                "early_exercise_premium": 0.0,
                "european_price": intrinsic,
                "american_price": intrinsic,
                "exercise_boundary": K,
                "optimal_to_exercise": intrinsic > 0,
            }

        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)

        # Validate probability bounds - if p is outside [0, 1], model assumptions are violated
        if p < 0 or p > 1:
            return {
                "error": "invalid_probability",
                "message": f"Risk-neutral probability p={p:.4f} is outside [0,1]. "
                           "Check model parameters: r-q may be too extreme relative to volatility.",
                "early_exercise_premium": None,
                "european_price": None,
                "american_price": None,
                "exercise_boundary": None,
                "optimal_to_exercise": None,
            }

        # Build price tree
        prices = np.zeros((n_steps + 1, n_steps + 1))
        for i in range(n_steps + 1):
            for j in range(i + 1):
                prices[j, i] = S * (u ** (i - j)) * (d ** j)

        # Calculate option values backwards
        option_values = np.zeros((n_steps + 1, n_steps + 1))
        exercise_nodes = np.zeros((n_steps + 1, n_steps + 1), dtype=bool)

        # Terminal values
        for j in range(n_steps + 1):
            if is_call:
                option_values[j, n_steps] = max(0, prices[j, n_steps] - K)
            else:
                option_values[j, n_steps] = max(0, K - prices[j, n_steps])

        # Work backwards
        discount = np.exp(-r * dt)
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                hold_value = discount * (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1])

                if is_call:
                    exercise_value = max(0, prices[j, i] - K)
                else:
                    exercise_value = max(0, K - prices[j, i])

                # American option: can exercise early
                if exercise_value > hold_value:
                    option_values[j, i] = exercise_value
                    exercise_nodes[j, i] = True
                else:
                    option_values[j, i] = hold_value

        american_price = option_values[0, 0]

        # Calculate European price for comparison
        european_price = self.black_scholes_price(S, K, T, r, sigma, is_call, q)

        # Early exercise premium
        early_exercise_premium = american_price - european_price

        # Find exercise boundary (first price where exercise is optimal)
        exercise_boundary = None
        for i in range(n_steps):
            for j in range(i + 1):
                if exercise_nodes[j, i]:
                    exercise_boundary = prices[j, i]
                    break
            if exercise_boundary:
                break

        return {
            "american_price": american_price,
            "european_price": european_price,
            "early_exercise_premium": early_exercise_premium,
            "premium_pct": (early_exercise_premium / european_price * 100) if european_price > 0 else 0,
            "exercise_boundary": exercise_boundary,
            "optimal_to_exercise_now": is_call and S > K and q > r,  # Simplified rule
            "n_exercise_nodes": np.sum(exercise_nodes),
        }

    def should_exercise_early(
        self,
        option: OptionData,
        underlying_price: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        days_to_ex_div: int | None = None,
        expected_dividend: float | None = None
    ) -> dict:
        """
        Determine if early exercise is optimal (#O3).

        For calls: Exercise before ex-dividend if dividend > time value
        For puts: Exercise if time value < interest earned on proceeds

        Args:
            option: Option contract data
            underlying_price: Current underlying price
            risk_free_rate: Risk-free rate
            dividend_yield: Continuous dividend yield
            days_to_ex_div: Days until ex-dividend date
            expected_dividend: Expected dividend amount

        Returns:
            Early exercise recommendation
        """
        T = option.expiry_days / 365.0
        intrinsic = max(0, underlying_price - option.strike) if option.is_call else max(0, option.strike - underlying_price)
        time_value = option.mid_price - intrinsic

        result = {
            "should_exercise": False,
            "reason": None,
            "intrinsic_value": intrinsic,
            "time_value": time_value,
            "time_value_pct": (time_value / intrinsic * 100) if intrinsic > 0 else float('inf'),
        }

        if intrinsic <= 0:
            result["reason"] = "out_of_the_money"
            return result

        if option.is_call:
            # For calls: exercise before ex-div if dividend > time value
            if days_to_ex_div is not None and expected_dividend is not None:
                if days_to_ex_div <= 1 and expected_dividend > time_value:
                    result["should_exercise"] = True
                    result["reason"] = f"dividend ${expected_dividend:.2f} exceeds time value ${time_value:.2f}"
                    return result

            # Deep ITM with low time value
            moneyness = underlying_price / option.strike
            if moneyness > 1.3 and time_value < intrinsic * 0.01:
                result["should_exercise"] = True
                result["reason"] = "deep_itm_minimal_time_value"
                return result

        else:  # Put
            # For puts: compare time value to interest on strike
            days_interest = option.strike * risk_free_rate * (option.expiry_days / 365)
            if time_value < days_interest:
                result["should_exercise"] = True
                result["reason"] = f"time value ${time_value:.2f} < interest earned ${days_interest:.2f}"
                return result

            # Deep ITM put
            moneyness = option.strike / underlying_price
            if moneyness > 1.3 and time_value < intrinsic * 0.01:
                result["should_exercise"] = True
                result["reason"] = "deep_itm_minimal_time_value"
                return result

        result["reason"] = "hold_position"
        return result

    # =========================================================================
    # IMPLIED VOLATILITY SURFACE (#O4)
    # =========================================================================

    def build_vol_surface(
        self,
        options: list[OptionData],
        spot_price: float,
        risk_free_rate: float = 0.05
    ) -> dict:
        """
        Build implied volatility surface (#O4).

        Creates a grid of IV by strike (moneyness) and expiry.

        Args:
            options: List of option data
            spot_price: Current underlying price
            risk_free_rate: Risk-free rate

        Returns:
            Volatility surface data structure
        """
        if not options:
            return {"error": "no_options_provided"}

        # Extract unique expiries and moneyness levels
        expiries = sorted(set(o.expiry_days for o in options))
        moneyness_levels = []
        iv_grid = {}

        for opt in options:
            moneyness = np.log(opt.strike / spot_price)  # Log-moneyness
            moneyness_levels.append(moneyness)

            key = (opt.expiry_days, round(moneyness, 3))
            if key not in iv_grid:
                iv_grid[key] = []
            iv_grid[key].append(opt.implied_vol)

        # Average IVs at each point
        for key in iv_grid:
            iv_grid[key] = np.mean(iv_grid[key])

        # Create surface arrays
        unique_moneyness = sorted(set(round(m, 3) for m in moneyness_levels))

        surface = np.full((len(expiries), len(unique_moneyness)), np.nan)
        for i, exp in enumerate(expiries):
            for j, mon in enumerate(unique_moneyness):
                key = (exp, mon)
                if key in iv_grid:
                    surface[i, j] = iv_grid[key]

        # Calculate surface metrics
        atm_term_structure = []
        for i, exp in enumerate(expiries):
            # Find ATM (moneyness closest to 0)
            atm_idx = np.argmin([abs(m) for m in unique_moneyness])
            if not np.isnan(surface[i, atm_idx]):
                atm_term_structure.append((exp, surface[i, atm_idx]))

        return {
            "expiries": expiries,
            "moneyness": unique_moneyness,
            "surface": surface.tolist(),
            "atm_term_structure": atm_term_structure,
            "spot_price": spot_price,
            "n_points": len(iv_grid),
        }

    def interpolate_vol(
        self,
        surface: dict,
        target_expiry: int,
        target_moneyness: float
    ) -> float | None:
        """
        Interpolate volatility from surface (#O4).

        Uses bilinear interpolation for intermediate points.

        Args:
            surface: Vol surface from build_vol_surface
            target_expiry: Target expiry in days
            target_moneyness: Target log-moneyness

        Returns:
            Interpolated implied volatility
        """
        expiries = surface.get("expiries", [])
        moneyness = surface.get("moneyness", [])
        grid = np.array(surface.get("surface", []))

        if len(expiries) == 0 or len(moneyness) == 0:
            return None

        # Find bracketing expiries
        exp_idx = np.searchsorted(expiries, target_expiry)
        mon_idx = np.searchsorted(moneyness, target_moneyness)

        # Clamp to valid range
        exp_idx = max(1, min(exp_idx, len(expiries) - 1))
        mon_idx = max(1, min(mon_idx, len(moneyness) - 1))

        # Get four corner points
        e1, e2 = expiries[exp_idx - 1], expiries[exp_idx]
        m1, m2 = moneyness[mon_idx - 1], moneyness[mon_idx]

        v11 = grid[exp_idx - 1, mon_idx - 1]
        v12 = grid[exp_idx - 1, mon_idx]
        v21 = grid[exp_idx, mon_idx - 1]
        v22 = grid[exp_idx, mon_idx]

        # Check for NaN
        if any(np.isnan([v11, v12, v21, v22])):
            # Fall back to nearest neighbor
            valid_vals = [v for v in [v11, v12, v21, v22] if not np.isnan(v)]
            return np.mean(valid_vals) if valid_vals else None

        # Bilinear interpolation
        if e2 != e1:
            t = (target_expiry - e1) / (e2 - e1)
        else:
            t = 0.5

        if m2 != m1:
            u = (target_moneyness - m1) / (m2 - m1)
        else:
            u = 0.5

        # Interpolate
        vol = (1 - t) * (1 - u) * v11 + (1 - t) * u * v12 + t * (1 - u) * v21 + t * u * v22

        return vol

    # =========================================================================
    # VOLATILITY SMILE/SKEW HANDLING (#O5)
    # =========================================================================

    def analyze_skew(
        self,
        options: list[OptionData],
        spot_price: float,
        expiry_days: int | None = None
    ) -> dict:
        """
        Analyze volatility skew at given expiry (#O5).

        Calculates:
        - 25-delta risk reversal (put IV - call IV)
        - Butterfly (wing average - ATM)
        - Skew slope

        Args:
            options: List of option data
            spot_price: Current underlying price
            expiry_days: Specific expiry or None for all

        Returns:
            Skew analysis metrics
        """
        if expiry_days is not None:
            options = [o for o in options if o.expiry_days == expiry_days]

        if not options:
            return {"error": "no_options_for_expiry"}

        # Separate calls and puts
        calls = [o for o in options if o.is_call]
        puts = [o for o in options if not o.is_call]

        # Calculate moneyness for each
        def get_by_delta(opts: list[OptionData], target_delta: float, tolerance: float = 0.05):
            matches = [o for o in opts if abs(abs(o.delta) - abs(target_delta)) < tolerance]
            return np.mean([o.implied_vol for o in matches]) if matches else None

        # Get delta-bucketed IVs
        put_25d_iv = get_by_delta(puts, 0.25)
        call_25d_iv = get_by_delta(calls, 0.25)
        put_10d_iv = get_by_delta(puts, 0.10)
        call_10d_iv = get_by_delta(calls, 0.10)

        # ATM (50 delta)
        atm_opts = [o for o in options if abs(o.strike - spot_price) / spot_price < 0.02]
        atm_iv = np.mean([o.implied_vol for o in atm_opts]) if atm_opts else None

        # Risk reversal (25 delta)
        rr_25d = (put_25d_iv - call_25d_iv) if (put_25d_iv and call_25d_iv) else None

        # Butterfly (25 delta)
        if put_25d_iv and call_25d_iv and atm_iv:
            bf_25d = 0.5 * (put_25d_iv + call_25d_iv) - atm_iv
        else:
            bf_25d = None

        # Skew slope (linear regression of IV vs moneyness)
        moneyness = []
        ivs = []
        for o in options:
            m = np.log(o.strike / spot_price)
            moneyness.append(m)
            ivs.append(o.implied_vol)

        if len(moneyness) >= 3:
            slope, intercept = np.polyfit(moneyness, ivs, 1)
        else:
            slope, intercept = None, None

        return {
            "atm_iv": atm_iv,
            "put_25d_iv": put_25d_iv,
            "call_25d_iv": call_25d_iv,
            "put_10d_iv": put_10d_iv,
            "call_10d_iv": call_10d_iv,
            "risk_reversal_25d": rr_25d,  # Positive = puts more expensive (crash protection)
            "butterfly_25d": bf_25d,  # Positive = smile (wings expensive)
            "skew_slope": slope,  # Negative = typical equity skew
            "skew_intercept": intercept,
        }

    def detect_skew_anomaly(
        self,
        current_skew: dict,
        historical_skew: list[dict],
        z_threshold: float = 2.0
    ) -> dict:
        """
        Detect anomalies in current skew vs history (#O5).

        Args:
            current_skew: Current skew analysis
            historical_skew: List of historical skew analyses
            z_threshold: Z-score threshold for anomaly

        Returns:
            Anomaly detection results
        """
        if len(historical_skew) < 10:
            return {"error": "insufficient_history", "anomalies": []}

        anomalies = []

        metrics = ["risk_reversal_25d", "butterfly_25d", "skew_slope"]
        for metric in metrics:
            current_val = current_skew.get(metric)
            if current_val is None:
                continue

            historical_vals = [h.get(metric) for h in historical_skew if h.get(metric) is not None]
            if len(historical_vals) < 5:
                continue

            mean = np.mean(historical_vals)
            std = np.std(historical_vals)

            if std > 0:
                z_score = (current_val - mean) / std
                if abs(z_score) > z_threshold:
                    anomalies.append({
                        "metric": metric,
                        "current": current_val,
                        "mean": mean,
                        "std": std,
                        "z_score": z_score,
                        "direction": "high" if z_score > 0 else "low",
                    })

        return {
            "has_anomaly": len(anomalies) > 0,
            "anomalies": anomalies,
            "z_threshold": z_threshold,
        }

    # =========================================================================
    # GREEKS TERM STRUCTURE (#O6)
    # =========================================================================

    def calculate_greeks_term_structure(
        self,
        options: list[OptionData],
        spot_price: float
    ) -> dict:
        """
        Calculate Greeks term structure across expiries (#O6).

        Shows how Greeks evolve with time to expiration.

        Args:
            options: List of option data
            spot_price: Current underlying price

        Returns:
            Greeks term structure by expiry
        """
        # Group by expiry
        by_expiry: dict[int, list[OptionData]] = {}
        for opt in options:
            if opt.expiry_days not in by_expiry:
                by_expiry[opt.expiry_days] = []
            by_expiry[opt.expiry_days].append(opt)

        term_structure = []

        for expiry, opts in sorted(by_expiry.items()):
            # Find ATM options
            atm_opts = [o for o in opts if abs(o.strike - spot_price) / spot_price < 0.05]
            if not atm_opts:
                continue

            # Average Greeks at ATM
            avg_gamma = np.mean([o.gamma for o in atm_opts])
            avg_theta = np.mean([o.theta for o in atm_opts])
            avg_vega = np.mean([o.vega for o in atm_opts])

            # Calculate gamma/theta ratio
            gamma_theta_ratio = abs(avg_gamma / avg_theta) if avg_theta != 0 else None

            term_structure.append({
                "expiry_days": expiry,
                "atm_gamma": avg_gamma,
                "atm_theta": avg_theta,
                "atm_vega": avg_vega,
                "gamma_theta_ratio": gamma_theta_ratio,
                "n_options": len(atm_opts),
            })

        # Calculate aggregate portfolio Greeks if we had positions
        total_gamma = sum(ts["atm_gamma"] for ts in term_structure)
        total_theta = sum(ts["atm_theta"] for ts in term_structure)
        total_vega = sum(ts["atm_vega"] for ts in term_structure)

        return {
            "term_structure": term_structure,
            "total_atm_gamma": total_gamma,
            "total_atm_theta": total_theta,
            "total_atm_vega": total_vega,
            "gamma_weighted_avg_dte": sum(
                ts["expiry_days"] * ts["atm_gamma"] for ts in term_structure
            ) / total_gamma if total_gamma > 0 else None,
        }

    # =========================================================================
    # OPTION SPREAD STRATEGIES (#O7)
    # =========================================================================

    def create_vertical_spread(
        self,
        options: list[OptionData],
        spread_type: str,  # "bull_call", "bear_call", "bull_put", "bear_put"
        target_delta: float = 0.30,
        width_pct: float = 0.05
    ) -> dict | None:
        """
        Create vertical spread strategy (#O7).

        Args:
            options: Available options
            spread_type: Type of spread
            target_delta: Delta for short leg
            width_pct: Spread width as % of underlying

        Returns:
            Spread definition or None if not possible
        """
        is_call_spread = "call" in spread_type
        is_bullish = "bull" in spread_type

        # Filter to calls or puts
        candidates = [o for o in options if o.is_call == is_call_spread]
        if len(candidates) < 2:
            return None

        # Find legs
        if is_bullish and is_call_spread:  # Bull call spread: buy lower, sell higher
            long_opts = sorted(candidates, key=lambda o: abs(o.delta - target_delta))
            long_leg = long_opts[0] if long_opts else None
            if long_leg:
                short_candidates = [o for o in candidates if o.strike > long_leg.strike]
                short_leg = min(short_candidates, key=lambda o: abs(o.strike - long_leg.strike * (1 + width_pct))) if short_candidates else None
        elif not is_bullish and is_call_spread:  # Bear call spread: sell lower, buy higher
            short_opts = sorted(candidates, key=lambda o: abs(o.delta - target_delta))
            short_leg = short_opts[0] if short_opts else None
            if short_leg:
                long_candidates = [o for o in candidates if o.strike > short_leg.strike]
                long_leg = min(long_candidates, key=lambda o: abs(o.strike - short_leg.strike * (1 + width_pct))) if long_candidates else None
        elif is_bullish and not is_call_spread:  # Bull put spread: sell higher, buy lower
            short_opts = sorted(candidates, key=lambda o: abs(abs(o.delta) - target_delta))
            short_leg = short_opts[0] if short_opts else None
            if short_leg:
                long_candidates = [o for o in candidates if o.strike < short_leg.strike]
                long_leg = max(long_candidates, key=lambda o: o.strike) if long_candidates else None
        else:  # Bear put spread: buy higher, sell lower
            long_opts = sorted(candidates, key=lambda o: abs(abs(o.delta) - target_delta))
            long_leg = long_opts[0] if long_opts else None
            if long_leg:
                short_candidates = [o for o in candidates if o.strike < long_leg.strike]
                short_leg = max(short_candidates, key=lambda o: o.strike) if short_candidates else None

        if not long_leg or not short_leg:
            return None

        # Calculate spread metrics
        net_premium = long_leg.mid_price - short_leg.mid_price
        max_profit = abs(short_leg.strike - long_leg.strike) - abs(net_premium) if net_premium > 0 else abs(net_premium)
        max_loss = abs(net_premium) if net_premium > 0 else abs(short_leg.strike - long_leg.strike) - abs(net_premium)
        net_delta = long_leg.delta - short_leg.delta
        net_theta = long_leg.theta - short_leg.theta

        return {
            "spread_type": spread_type,
            "long_leg": {
                "symbol": long_leg.symbol,
                "strike": long_leg.strike,
                "delta": long_leg.delta,
                "price": long_leg.mid_price,
            },
            "short_leg": {
                "symbol": short_leg.symbol,
                "strike": short_leg.strike,
                "delta": short_leg.delta,
                "price": short_leg.mid_price,
            },
            "net_premium": net_premium,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "risk_reward": max_profit / max_loss if max_loss > 0 else float('inf'),
            "net_delta": net_delta,
            "net_theta": net_theta,
            "breakeven": long_leg.strike + net_premium if is_call_spread else short_leg.strike - net_premium,
        }

    def create_iron_condor(
        self,
        options: list[OptionData],
        put_delta: float = 0.20,
        call_delta: float = 0.20,
        wing_width_pct: float = 0.03
    ) -> dict | None:
        """
        Create iron condor strategy (#O7).

        Sells put spread below market, sells call spread above market.

        Args:
            options: Available options
            put_delta: Delta for short put
            call_delta: Delta for short call
            wing_width_pct: Wing width as % of strike

        Returns:
            Iron condor definition
        """
        calls = [o for o in options if o.is_call]
        puts = [o for o in options if not o.is_call]

        if len(calls) < 2 or len(puts) < 2:
            return None

        # Find short strikes
        short_put = min(puts, key=lambda o: abs(abs(o.delta) - put_delta))
        short_call = min(calls, key=lambda o: abs(o.delta - call_delta))

        # Find wings
        long_puts = [o for o in puts if o.strike < short_put.strike]
        long_calls = [o for o in calls if o.strike > short_call.strike]

        if not long_puts or not long_calls:
            return None

        long_put = max(long_puts, key=lambda o: o.strike)
        long_call = min(long_calls, key=lambda o: o.strike)

        # Calculate metrics
        credit = (short_put.mid_price + short_call.mid_price) - (long_put.mid_price + long_call.mid_price)
        put_spread_width = short_put.strike - long_put.strike
        call_spread_width = long_call.strike - short_call.strike
        max_risk = max(put_spread_width, call_spread_width) - credit

        net_delta = short_put.delta + short_call.delta - long_put.delta - long_call.delta
        net_theta = short_put.theta + short_call.theta - long_put.theta - long_call.theta

        return {
            "strategy": "iron_condor",
            "legs": {
                "long_put": {"strike": long_put.strike, "delta": long_put.delta, "price": long_put.mid_price},
                "short_put": {"strike": short_put.strike, "delta": short_put.delta, "price": short_put.mid_price},
                "short_call": {"strike": short_call.strike, "delta": short_call.delta, "price": short_call.mid_price},
                "long_call": {"strike": long_call.strike, "delta": long_call.delta, "price": long_call.mid_price},
            },
            "credit": credit,
            "max_profit": credit,
            "max_loss": max_risk,
            "profit_range": (short_put.strike, short_call.strike),
            "net_delta": net_delta,
            "net_theta": net_theta,
            "probability_of_profit": 1 - abs(short_put.delta) - abs(short_call.delta),  # Approximation using delta as ITM proxy
        }

    # =========================================================================
    # PIN RISK DETECTION (#O8)
    # =========================================================================

    def detect_pin_risk(
        self,
        options: list[OptionData],
        spot_price: float,
        position_size: int,
        pin_threshold_pct: float = 1.0
    ) -> dict:
        """
        Detect pin risk near expiration (#O8).

        Pin risk occurs when underlying settles near a strike at expiration,
        making assignment uncertain.

        Args:
            options: Option positions
            spot_price: Current underlying price
            position_size: Number of contracts (negative = short)
            pin_threshold_pct: % from strike to flag as at-risk

        Returns:
            Pin risk analysis
        """
        at_risk = []
        total_pin_exposure = 0

        for opt in options:
            # Only check near-expiry options
            if opt.expiry_days > 5:
                continue

            # Check if near a strike
            distance_pct = abs(spot_price - opt.strike) / opt.strike * 100

            if distance_pct < pin_threshold_pct:
                # Estimate assignment probability based on proximity
                assignment_prob = max(0, 1 - distance_pct / pin_threshold_pct) * 0.5

                # Calculate potential exposure
                shares_at_risk = abs(position_size) * 100  # 100 shares per contract

                at_risk.append({
                    "symbol": opt.symbol,
                    "strike": opt.strike,
                    "expiry_days": opt.expiry_days,
                    "is_call": opt.is_call,
                    "distance_from_strike_pct": distance_pct,
                    "assignment_probability": assignment_prob,
                    "shares_at_risk": shares_at_risk,
                    "value_at_risk": shares_at_risk * spot_price,
                })
                total_pin_exposure += shares_at_risk * spot_price * assignment_prob

        return {
            "has_pin_risk": len(at_risk) > 0,
            "positions_at_risk": at_risk,
            "total_pin_exposure": total_pin_exposure,
            "recommendation": "Consider rolling or closing positions" if at_risk else "No immediate pin risk",
        }

    # =========================================================================
    # ASSIGNMENT RISK CALCULATION (#O9)
    # =========================================================================

    def calculate_assignment_risk(
        self,
        option: OptionData,
        spot_price: float,
        position_size: int,
        days_to_ex_div: int | None = None,
        dividend_amount: float | None = None
    ) -> dict:
        """
        Calculate assignment risk for short options (#O9).

        Assignment is most likely for:
        - Deep ITM options (high intrinsic value)
        - Near expiration
        - Before ex-dividend for calls

        Args:
            option: Option contract
            spot_price: Current underlying price
            position_size: Number of contracts (negative = short)
            days_to_ex_div: Days to ex-dividend
            dividend_amount: Expected dividend

        Returns:
            Assignment risk analysis
        """
        if position_size >= 0:
            return {
                "assignment_risk": 0,
                "reason": "long_position_cannot_be_assigned",
            }

        # Calculate intrinsic value
        if option.is_call:
            intrinsic = max(0, spot_price - option.strike)
            moneyness = spot_price / option.strike
        else:
            intrinsic = max(0, option.strike - spot_price)
            moneyness = option.strike / spot_price

        time_value = option.mid_price - intrinsic

        # Base assignment probability
        if intrinsic <= 0:
            base_prob = 0.0
        elif moneyness > 1.10:  # > 10% ITM
            base_prob = 0.7
        elif moneyness > 1.05:  # 5-10% ITM
            base_prob = 0.3
        else:  # < 5% ITM
            base_prob = 0.1

        # Time adjustment (higher risk near expiry)
        if option.expiry_days <= 1:
            time_factor = 2.0
        elif option.expiry_days <= 5:
            time_factor = 1.5
        elif option.expiry_days <= 10:
            time_factor = 1.2
        else:
            time_factor = 1.0

        # Dividend adjustment (calls before ex-div)
        div_factor = 1.0
        if option.is_call and days_to_ex_div is not None and dividend_amount is not None:
            if days_to_ex_div <= 1 and dividend_amount > time_value:
                div_factor = 3.0  # High assignment risk

        # Calculate final probability
        assignment_prob = min(0.95, base_prob * time_factor * div_factor)

        # Calculate exposure
        shares_exposure = abs(position_size) * 100
        dollar_exposure = shares_exposure * spot_price

        return {
            "assignment_probability": assignment_prob,
            "moneyness": moneyness,
            "intrinsic_value": intrinsic,
            "time_value": time_value,
            "shares_exposure": shares_exposure,
            "dollar_exposure": dollar_exposure,
            "expected_assignment_cost": dollar_exposure * assignment_prob,
            "risk_level": "HIGH" if assignment_prob > 0.5 else ("MEDIUM" if assignment_prob > 0.2 else "LOW"),
            "factors": {
                "base_prob": base_prob,
                "time_factor": time_factor,
                "dividend_factor": div_factor,
            },
        }

    # =========================================================================
    # GAMMA SCALPING SUPPORT (#O10)
    # =========================================================================

    def calculate_gamma_scalp_parameters(
        self,
        option: OptionData,
        position_size: int,
        spot_price: float,
        realized_vol: float,
        hedge_interval_seconds: int = 300
    ) -> dict:
        """
        Calculate parameters for gamma scalping strategy (#O10).

        Gamma scalping profits from realized volatility exceeding implied
        by delta-hedging an options position.

        Args:
            option: Option contract
            position_size: Number of contracts (positive = long gamma)
            spot_price: Current underlying price
            realized_vol: Current realized volatility
            hedge_interval_seconds: Seconds between hedges

        Returns:
            Gamma scalping parameters
        """
        if position_size <= 0:
            return {
                "strategy_viable": False,
                "reason": "need_long_gamma_position",
            }

        # Position Greeks
        position_gamma = option.gamma * position_size * 100
        position_theta = option.theta * position_size * 100

        # Gamma P&L per unit price move
        gamma_pnl_per_pct = 0.5 * position_gamma * (spot_price * 0.01) ** 2

        # Daily theta cost
        daily_theta_cost = abs(position_theta)

        # Break-even daily move
        # Theta = 0.5 * Gamma * S^2 * sigma^2 * dt
        # Break-even sigma = sqrt(2 * Theta / (Gamma * S^2 * dt))
        dt = 1 / 252  # Daily
        # Guard against division by zero when gamma is very small
        if position_gamma > 1e-8:
            breakeven_vol = np.sqrt(2 * abs(position_theta) / (position_gamma * spot_price ** 2 * dt))
        else:
            breakeven_vol = float('inf')

        # Expected daily P&L
        # P&L = 0.5 * Gamma * S^2 * (RV^2 - IV^2) * dt
        expected_daily_pnl = 0.5 * position_gamma * spot_price ** 2 * (
            realized_vol ** 2 - option.implied_vol ** 2
        ) * dt

        # Hedge size calculation
        # Hedge when delta exceeds threshold
        delta_threshold = 0.01  # 1% of spot
        hedge_trigger_move = delta_threshold * spot_price / (position_gamma / 100) if position_gamma > 0 else float('inf')

        return {
            "strategy_viable": realized_vol > option.implied_vol,
            "position_gamma": position_gamma,
            "position_theta": position_theta,
            "gamma_pnl_per_1pct_move": gamma_pnl_per_pct,
            "daily_theta_cost": daily_theta_cost,
            "breakeven_vol": breakeven_vol,
            "implied_vol": option.implied_vol,
            "realized_vol": realized_vol,
            "vol_edge": realized_vol - option.implied_vol,
            "expected_daily_pnl": expected_daily_pnl,
            "hedge_trigger_move": hedge_trigger_move,
            "hedge_frequency_estimate": hedge_interval_seconds,
            "recommendation": "scalp_gamma" if realized_vol > breakeven_vol else "avoid_theta_decay",
        }

    def calculate_delta_hedge(
        self,
        options: list[tuple[OptionData, int]],  # (option, position_size)
        spot_price: float,
        current_hedge_shares: int = 0
    ) -> dict:
        """
        Calculate required delta hedge (#O10).

        Args:
            options: List of (option, position) tuples
            spot_price: Current underlying price
            current_hedge_shares: Current hedge position

        Returns:
            Hedge adjustment needed
        """
        # Calculate total portfolio delta
        total_delta = 0
        for opt, size in options:
            # Delta per contract * 100 shares * number of contracts
            total_delta += opt.delta * 100 * size

        # Net delta including current hedge
        net_delta = total_delta + current_hedge_shares

        # OPT-P0-3: Hedge adjustment using floor/ceil based on direction
        # to avoid residual exposure from rounding errors
        # If net_delta > 0, we need to sell shares: floor(155.7) = 155 -> hedge = -155
        # If net_delta < 0, we need to buy shares: ceil(155.7) = 156 -> hedge = 156
        if net_delta > 0:
            hedge_adjustment = -math.floor(net_delta)
        elif net_delta < 0:
            hedge_adjustment = math.ceil(abs(net_delta))
        else:
            hedge_adjustment = 0

        return {
            "option_delta": total_delta,
            "current_hedge": current_hedge_shares,
            "net_delta": net_delta,
            "hedge_adjustment_needed": hedge_adjustment,
            "new_hedge_position": current_hedge_shares + hedge_adjustment,
            "is_delta_neutral": abs(net_delta) < 10,  # Within 10 shares
        }

    # =========================================================================
    # VANNA/VOLGA ADJUSTMENTS (#O11)
    # =========================================================================

    def calculate_vanna(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate Vanna (dDelta/dVol or dVega/dSpot) (#O11).

        Vanna measures sensitivity of delta to volatility changes.

        Args:
            S: Spot price
            K: Strike
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Vanna value
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Vanna = -e^(-qT) * N'(d1) * d2 / sigma
        vanna = -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma

        return vanna

    def calculate_volga(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate Volga (dVega/dVol, also called Vomma) (#O11).

        Volga measures convexity of vega with respect to volatility.

        Args:
            S: Spot price
            K: Strike
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Volga value
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Vega first
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        # Volga = Vega * d1 * d2 / sigma
        volga = vega * d1 * d2 / sigma

        return volga

    def apply_vanna_volga_adjustment(
        self,
        bs_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        atm_vol: float,
        put_25d_vol: float,
        call_25d_vol: float,
        q: float = 0.0
    ) -> dict:
        """
        Apply Vanna-Volga adjustment to Black-Scholes price (#O11).

        Adjusts BS price to account for smile dynamics using
        market-implied risk reversals and butterflies.

        Args:
            bs_price: Black-Scholes price
            S: Spot price
            K: Strike
            T: Time to expiry
            r: Risk-free rate
            atm_vol: ATM implied volatility
            put_25d_vol: 25-delta put IV
            call_25d_vol: 25-delta call IV
            q: Dividend yield

        Returns:
            Adjusted price and components
        """
        # Calculate Greeks at ATM vol
        vanna = self.calculate_vanna(S, K, T, r, atm_vol, q)
        volga = self.calculate_volga(S, K, T, r, atm_vol, q)

        # Market smile parameters
        risk_reversal = put_25d_vol - call_25d_vol  # Skew
        butterfly = 0.5 * (put_25d_vol + call_25d_vol) - atm_vol  # Convexity

        # Vanna-Volga adjustment
        # Price adjustment = Vanna * RR_adjustment + Volga * BF_adjustment
        vanna_adj = vanna * risk_reversal * 0.5  # Simplified scaling
        volga_adj = volga * butterfly * 0.5

        total_adjustment = vanna_adj + volga_adj
        adjusted_price = bs_price + total_adjustment

        return {
            "bs_price": bs_price,
            "vanna": vanna,
            "volga": volga,
            "risk_reversal": risk_reversal,
            "butterfly": butterfly,
            "vanna_adjustment": vanna_adj,
            "volga_adjustment": volga_adj,
            "total_adjustment": total_adjustment,
            "adjusted_price": adjusted_price,
            "adjustment_pct": (total_adjustment / bs_price * 100) if bs_price > 0 else 0,
        }

    # =========================================================================
    # VOL SURFACE INTERPOLATION METHODS (P3)
    # =========================================================================

    def interpolate_vol_cubic(
        self,
        surface: dict,
        target_expiry: int,
        target_moneyness: float
    ) -> float | None:
        """
        Interpolate volatility using cubic spline interpolation (P3).

        More accurate than bilinear for smooth surfaces.

        Args:
            surface: Vol surface from build_vol_surface
            target_expiry: Target expiry in days
            target_moneyness: Target log-moneyness

        Returns:
            Interpolated implied volatility
        """
        expiries = surface.get("expiries", [])
        moneyness = surface.get("moneyness", [])
        grid = np.array(surface.get("surface", []))

        if len(expiries) < 2 or len(moneyness) < 2:
            return self.interpolate_vol(surface, target_expiry, target_moneyness)

        try:
            from scipy.interpolate import RectBivariateSpline

            # Remove NaN values for interpolation
            valid_mask = ~np.isnan(grid)
            if not np.any(valid_mask):
                return None

            # Fill NaN with nearest neighbor for spline fitting
            grid_filled = np.copy(grid)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if np.isnan(grid_filled[i, j]):
                        # Find nearest non-NaN value
                        valid_vals = grid[valid_mask]
                        if len(valid_vals) > 0:
                            grid_filled[i, j] = np.mean(valid_vals)

            # Create spline interpolator
            spline = RectBivariateSpline(
                expiries, moneyness, grid_filled, kx=min(3, len(expiries) - 1), ky=min(3, len(moneyness) - 1)
            )

            # Interpolate
            vol = spline(target_expiry, target_moneyness)[0, 0]
            return float(vol)

        except (ImportError, ValueError) as e:
            logger.warning(f"Cubic interpolation failed, falling back to bilinear: {e}")
            return self.interpolate_vol(surface, target_expiry, target_moneyness)

    def fit_svi_slice(
        self,
        strikes: list[float],
        ivs: list[float],
        forward: float,
        time_to_expiry: float
    ) -> dict | None:
        """
        Fit SVI (Stochastic Volatility Inspired) parameterization to a vol slice (P3).

        SVI: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        where w = total variance = iv^2 * T, k = log(K/F)

        Args:
            strikes: Strike prices
            ivs: Implied volatilities at each strike
            forward: Forward price
            time_to_expiry: Time to expiry in years

        Returns:
            SVI parameters {a, b, rho, m, sigma} or None if fit fails
        """
        if len(strikes) < 5 or time_to_expiry <= 0:
            return None

        try:
            from scipy.optimize import minimize

            # Convert to log-moneyness and total variance
            k = np.log(np.array(strikes) / forward)
            w = np.array(ivs) ** 2 * time_to_expiry

            def svi(params, k):
                a, b, rho, m, sigma = params
                return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

            def objective(params):
                pred = svi(params, k)
                return np.sum((pred - w) ** 2)

            # Initial guess
            a0 = np.mean(w)
            b0 = 0.1
            rho0 = -0.3
            m0 = 0.0
            sigma0 = 0.1

            # Constraints: b >= 0, |rho| < 1, sigma > 0, a + b*sigma*sqrt(1-rho^2) >= 0
            bounds = [
                (None, None),  # a
                (0.001, None),  # b > 0
                (-0.99, 0.99),  # |rho| < 1
                (None, None),  # m
                (0.001, None),  # sigma > 0
            ]

            result = minimize(
                objective,
                [a0, b0, rho0, m0, sigma0],
                method='L-BFGS-B',
                bounds=bounds
            )

            if result.success:
                a, b, rho, m, sigma = result.x
                return {
                    "a": a,
                    "b": b,
                    "rho": rho,
                    "m": m,
                    "sigma": sigma,
                    "fit_error": result.fun,
                    "time_to_expiry": time_to_expiry,
                }
            return None

        except (ImportError, Exception) as e:
            logger.warning(f"SVI fitting failed: {e}")
            return None

    def interpolate_vol_svi(
        self,
        svi_params: dict,
        target_moneyness: float,
        time_to_expiry: float
    ) -> float | None:
        """
        Interpolate volatility using fitted SVI parameters (P3).

        Args:
            svi_params: SVI parameters from fit_svi_slice
            target_moneyness: Target log-moneyness (log(K/F))
            time_to_expiry: Time to expiry in years

        Returns:
            Implied volatility at target strike
        """
        if not svi_params or time_to_expiry <= 0:
            return None

        a = svi_params["a"]
        b = svi_params["b"]
        rho = svi_params["rho"]
        m = svi_params["m"]
        sigma = svi_params["sigma"]

        # Calculate total variance
        k = target_moneyness
        w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

        # Convert to IV
        if w <= 0:
            return None
        iv = np.sqrt(w / time_to_expiry)
        return float(iv)

    # =========================================================================
    # TERM STRUCTURE ANALYSIS (P3)
    # =========================================================================

    def analyze_term_structure(
        self,
        surface: dict,
        spot_price: float
    ) -> dict:
        """
        Analyze volatility term structure characteristics (P3).

        Examines how ATM volatility changes with expiry and identifies
        term structure patterns (contango, backwardation, humped).

        Args:
            surface: Vol surface from build_vol_surface
            spot_price: Current underlying price

        Returns:
            Term structure analysis
        """
        atm_term_structure = surface.get("atm_term_structure", [])

        if len(atm_term_structure) < 2:
            return {"error": "insufficient_data", "expiries": len(atm_term_structure)}

        expiries = [x[0] for x in atm_term_structure]
        ivs = [x[1] for x in atm_term_structure]

        # Calculate term structure slope
        if len(expiries) >= 2:
            slope, intercept = np.polyfit(expiries, ivs, 1)
        else:
            slope, intercept = 0, ivs[0] if ivs else 0

        # Identify term structure shape
        front_iv = ivs[0] if ivs else 0
        back_iv = ivs[-1] if ivs else 0

        if slope > 0.0001:
            shape = "contango"  # Back months higher IV
        elif slope < -0.0001:
            shape = "backwardation"  # Front months higher IV
        else:
            shape = "flat"

        # Check for hump
        is_humped = False
        hump_expiry = None
        if len(ivs) >= 3:
            max_idx = np.argmax(ivs)
            if 0 < max_idx < len(ivs) - 1:
                is_humped = True
                hump_expiry = expiries[max_idx]

        # Calculate roll yield proxy (front to second month)
        roll_yield = None
        if len(ivs) >= 2:
            roll_yield = (ivs[1] - ivs[0]) / ivs[0] * 100 if ivs[0] > 0 else 0

        # Forward volatility (variance interpolation)
        forward_vols = []
        for i in range(len(expiries) - 1):
            t1, v1 = expiries[i], ivs[i]
            t2, v2 = expiries[i + 1], ivs[i + 1]

            if t1 > 0 and t2 > t1:
                # Forward variance = (v2^2 * t2 - v1^2 * t1) / (t2 - t1)
                fwd_var = (v2 ** 2 * t2 / 365 - v1 ** 2 * t1 / 365) / ((t2 - t1) / 365)
                if fwd_var > 0:
                    fwd_vol = np.sqrt(fwd_var)
                    forward_vols.append({
                        "period": f"{t1}d-{t2}d",
                        "forward_vol": fwd_vol,
                    })

        return {
            "shape": shape,
            "slope": slope,
            "intercept": intercept,
            "front_iv": front_iv,
            "back_iv": back_iv,
            "is_humped": is_humped,
            "hump_expiry": hump_expiry,
            "roll_yield_pct": roll_yield,
            "forward_vols": forward_vols,
            "n_expiries": len(expiries),
            "term_structure": atm_term_structure,
        }

    def calculate_vol_carry(
        self,
        current_iv: float,
        realized_vol: float,
        days_to_expiry: int
    ) -> dict:
        """
        Calculate volatility carry/theta from vol premium (P3).

        Estimates daily P&L from being short volatility.

        Args:
            current_iv: Current implied volatility
            realized_vol: Current realized volatility
            days_to_expiry: Days to option expiry

        Returns:
            Vol carry analysis
        """
        vol_premium = current_iv - realized_vol

        # Daily variance carry (simplified)
        # Total variance = IV^2 * T, realized variance = RV^2 * T
        # Daily carry = (IV^2 - RV^2) / 365
        daily_variance_carry = (current_iv ** 2 - realized_vol ** 2) / 365

        # Annualized carry
        annualized_carry = vol_premium

        # Break-even days (how long RV can be at current level before losing)
        break_even_days = None
        if daily_variance_carry != 0:
            total_premium = vol_premium * days_to_expiry / 365
            break_even_days = abs(total_premium / daily_variance_carry) if daily_variance_carry != 0 else None

        return {
            "vol_premium": vol_premium,
            "daily_variance_carry": daily_variance_carry,
            "annualized_carry": annualized_carry,
            "days_to_expiry": days_to_expiry,
            "break_even_days": break_even_days,
            "carry_direction": "positive" if vol_premium > 0 else "negative",
            "recommendation": "sell_vol" if vol_premium > 0.02 else ("buy_vol" if vol_premium < -0.02 else "neutral"),
        }

    # =========================================================================
    # SKEW TRADING SIGNALS (P3)
    # =========================================================================

    def generate_skew_signal(
        self,
        current_skew: dict,
        historical_skew: list[dict],
        underlying: str
    ) -> dict | None:
        """
        Generate trading signal from skew analysis (P3).

        Identifies when skew is at extremes relative to history.

        Args:
            current_skew: Current skew from analyze_skew
            historical_skew: Historical skew readings
            underlying: Underlying symbol

        Returns:
            Skew trading signal or None
        """
        anomalies = self.detect_skew_anomaly(current_skew, historical_skew)

        if not anomalies.get("has_anomaly"):
            return None

        signals = []

        for anomaly in anomalies.get("anomalies", []):
            metric = anomaly["metric"]
            z_score = anomaly["z_score"]
            direction = anomaly["direction"]

            if metric == "risk_reversal_25d":
                # High RR = puts expensive, low RR = calls expensive
                if direction == "high" and z_score > 2:
                    # Puts very expensive - sell put spread, buy call spread
                    signals.append({
                        "trade": "sell_risk_reversal",
                        "action": "Sell OTM puts, buy OTM calls",
                        "rationale": f"Put skew at {z_score:.1f} sigma - puts overpriced",
                        "strength": min(1.0, z_score / 3),
                    })
                elif direction == "low" and z_score < -2:
                    # Calls expensive - opposite
                    signals.append({
                        "trade": "buy_risk_reversal",
                        "action": "Buy OTM puts, sell OTM calls",
                        "rationale": f"Call skew at {abs(z_score):.1f} sigma - calls overpriced",
                        "strength": min(1.0, abs(z_score) / 3),
                    })

            elif metric == "butterfly_25d":
                # High butterfly = wings expensive
                if direction == "high" and z_score > 2:
                    signals.append({
                        "trade": "sell_butterfly",
                        "action": "Sell wings, buy ATM (short butterfly)",
                        "rationale": f"Butterfly at {z_score:.1f} sigma - wings overpriced",
                        "strength": min(1.0, z_score / 3),
                    })
                elif direction == "low" and z_score < -2:
                    signals.append({
                        "trade": "buy_butterfly",
                        "action": "Buy wings, sell ATM (long butterfly)",
                        "rationale": f"Butterfly at {abs(z_score):.1f} sigma - wings cheap",
                        "strength": min(1.0, abs(z_score) / 3),
                    })

        if not signals:
            return None

        # Return strongest signal
        best_signal = max(signals, key=lambda s: s["strength"])

        return {
            "underlying": underlying,
            "signal_type": "skew_trade",
            "trade": best_signal["trade"],
            "action": best_signal["action"],
            "rationale": best_signal["rationale"],
            "strength": best_signal["strength"],
            "current_skew": {
                "risk_reversal_25d": current_skew.get("risk_reversal_25d"),
                "butterfly_25d": current_skew.get("butterfly_25d"),
            },
            "all_signals": signals,
        }

    def create_skew_trade(
        self,
        options: list[OptionData],
        trade_type: str,  # "risk_reversal", "butterfly"
        target_delta: float = 0.25
    ) -> dict | None:
        """
        Create a skew trade structure (P3).

        Args:
            options: Available options
            trade_type: Type of skew trade
            target_delta: Target delta for legs

        Returns:
            Trade structure or None
        """
        calls = [o for o in options if o.is_call]
        puts = [o for o in options if not o.is_call]

        if not calls or not puts:
            return None

        if trade_type == "risk_reversal":
            # Find matching delta put and call
            target_put = min(puts, key=lambda o: abs(abs(o.delta) - target_delta))
            target_call = min(calls, key=lambda o: abs(o.delta - target_delta))

            net_premium = target_call.mid_price - target_put.mid_price

            return {
                "trade_type": "risk_reversal",
                "buy_leg": {
                    "type": "call",
                    "strike": target_call.strike,
                    "delta": target_call.delta,
                    "price": target_call.mid_price,
                    "iv": target_call.implied_vol,
                },
                "sell_leg": {
                    "type": "put",
                    "strike": target_put.strike,
                    "delta": target_put.delta,
                    "price": target_put.mid_price,
                    "iv": target_put.implied_vol,
                },
                "net_premium": net_premium,
                "implied_skew": target_put.implied_vol - target_call.implied_vol,
                "max_loss": "unlimited on downside",
                "max_profit": "unlimited on upside",
            }

        elif trade_type == "butterfly":
            # Find ATM and wings
            atm_opts = [o for o in options if abs(o.delta) > 0.45 and abs(o.delta) < 0.55]
            if not atm_opts:
                return None

            # Use calls for simplicity
            atm_call = min([o for o in atm_opts if o.is_call], key=lambda o: abs(o.delta - 0.5), default=None)
            if not atm_call:
                return None

            # Find wings at target delta
            lower_wing = min(calls, key=lambda o: abs(o.delta - (1 - target_delta)))
            upper_wing = min(calls, key=lambda o: abs(o.delta - target_delta))

            # Long butterfly: buy lower, sell 2x ATM, buy upper
            net_premium = lower_wing.mid_price - 2 * atm_call.mid_price + upper_wing.mid_price

            return {
                "trade_type": "butterfly",
                "lower_wing": {
                    "strike": lower_wing.strike,
                    "action": "buy",
                    "price": lower_wing.mid_price,
                },
                "body": {
                    "strike": atm_call.strike,
                    "action": "sell",
                    "quantity": 2,
                    "price": atm_call.mid_price,
                },
                "upper_wing": {
                    "strike": upper_wing.strike,
                    "action": "buy",
                    "price": upper_wing.mid_price,
                },
                "net_premium": net_premium,
                "max_profit": atm_call.strike - lower_wing.strike - abs(net_premium),
                "max_loss": abs(net_premium),
            }

        return None

    # =========================================================================
    # OPTION PORTFOLIO HEDGING SUGGESTIONS (#O12)
    # =========================================================================

    def suggest_portfolio_hedges(
        self,
        portfolio_greeks: dict,
        available_options: list[OptionData],
        spot_price: float,
        hedge_targets: dict | None = None
    ) -> list[dict]:
        """
        Suggest hedging trades for portfolio Greeks (#O12).

        Args:
            portfolio_greeks: Current portfolio Greeks {delta, gamma, theta, vega}
            available_options: Options available for hedging
            spot_price: Current underlying price
            hedge_targets: Target Greeks (default: neutralize all)

        Returns:
            List of suggested hedge trades
        """
        if hedge_targets is None:
            hedge_targets = {"delta": 0, "gamma": 0, "vega": 0}

        suggestions = []

        # Delta hedge (with stock)
        current_delta = portfolio_greeks.get("delta", 0)
        target_delta = hedge_targets.get("delta", 0)
        delta_gap = target_delta - current_delta

        if abs(delta_gap) > 10:  # More than 10 delta
            suggestions.append({
                "type": "delta_hedge",
                "instrument": "stock",
                "action": "buy" if delta_gap > 0 else "sell",
                "quantity": abs(int(delta_gap)),
                "rationale": f"Neutralize delta from {current_delta:.0f} to {target_delta:.0f}",
            })

        # Gamma hedge (with options)
        current_gamma = portfolio_greeks.get("gamma", 0)
        target_gamma = hedge_targets.get("gamma", 0)
        gamma_gap = target_gamma - current_gamma

        if abs(gamma_gap) > 1:  # Significant gamma exposure
            # Find ATM options with highest gamma
            atm_opts = [o for o in available_options if abs(o.strike - spot_price) / spot_price < 0.05]
            if atm_opts:
                best_gamma_opt = max(atm_opts, key=lambda o: o.gamma)
                contracts_needed = int(abs(gamma_gap) / (best_gamma_opt.gamma * 100))

                if contracts_needed > 0:
                    suggestions.append({
                        "type": "gamma_hedge",
                        "instrument": best_gamma_opt.symbol,
                        "action": "buy" if gamma_gap > 0 else "sell",
                        "quantity": contracts_needed,
                        "option_gamma": best_gamma_opt.gamma,
                        "rationale": f"Adjust gamma from {current_gamma:.2f} toward {target_gamma:.2f}",
                    })

        # Vega hedge (with options)
        current_vega = portfolio_greeks.get("vega", 0)
        target_vega = hedge_targets.get("vega", 0)
        vega_gap = target_vega - current_vega

        if abs(vega_gap) > 100:  # Significant vega exposure
            # Find options with most vega
            high_vega_opts = sorted(available_options, key=lambda o: o.vega, reverse=True)[:5]
            if high_vega_opts:
                best_vega_opt = high_vega_opts[0]
                contracts_needed = int(abs(vega_gap) / (best_vega_opt.vega * 100))

                if contracts_needed > 0:
                    suggestions.append({
                        "type": "vega_hedge",
                        "instrument": best_vega_opt.symbol,
                        "action": "buy" if vega_gap > 0 else "sell",
                        "quantity": contracts_needed,
                        "option_vega": best_vega_opt.vega,
                        "rationale": f"Adjust vega from {current_vega:.0f} toward {target_vega:.0f}",
                    })

        return suggestions

    def calculate_hedge_cost(
        self,
        hedge_suggestions: list[dict],
        options: list[OptionData],
        spot_price: float
    ) -> dict:
        """
        Calculate cost of implementing hedge suggestions (#O12).

        Args:
            hedge_suggestions: List of suggested hedges
            options: Option universe
            spot_price: Current underlying price

        Returns:
            Cost breakdown
        """
        total_cost = 0
        cost_breakdown = []

        for hedge in hedge_suggestions:
            if hedge["type"] == "delta_hedge":
                # Stock hedge
                cost = hedge["quantity"] * spot_price
                cost_breakdown.append({
                    "hedge": hedge,
                    "cost": cost,
                    "type": "stock",
                })
                total_cost += cost

            else:
                # Option hedge
                opt_symbol = hedge.get("instrument")
                matching = [o for o in options if o.symbol == opt_symbol]
                if matching:
                    opt = matching[0]
                    # Use ask for buys, bid for sells
                    price = opt.ask if hedge["action"] == "buy" else opt.bid
                    cost = hedge["quantity"] * price * 100  # 100 shares per contract
                    cost_breakdown.append({
                        "hedge": hedge,
                        "cost": cost,
                        "option_price": price,
                        "type": "option",
                    })
                    total_cost += cost if hedge["action"] == "buy" else -cost

        return {
            "total_cost": total_cost,
            "breakdown": cost_breakdown,
            "n_hedges": len(hedge_suggestions),
        }

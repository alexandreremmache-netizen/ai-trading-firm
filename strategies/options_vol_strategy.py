"""
Options Volatility Strategy
===========================

Implements options and volatility-based trading logic.

TODO: This is a placeholder - implement actual vol models.
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
        Calculate Black-Scholes option price with dividend yield.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry in years
            r: Risk-free interest rate (annualized)
            sigma: Volatility (annualized)
            is_call: True for call, False for put
            q: Continuous dividend yield (annualized, e.g., 0.02 for 2%)

        Returns:
            Option price

        Note: For stocks with discrete dividends, convert to continuous yield:
            q = -ln(1 - PV(dividends)/S) / T
        """
        if T <= 0 or sigma <= 0:
            # Intrinsic value at expiry
            return max(0, S - K) if is_call else max(0, K - S)

        sqrt_T = math.sqrt(T)

        # Modified d1 and d2 with dividend yield
        d1 = (math.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Discount factors
        discount_factor = math.exp(-r * T)
        dividend_discount = math.exp(-q * T)

        if is_call:
            price = S * dividend_discount * norm.cdf(d1) - K * discount_factor * norm.cdf(d2)
        else:
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
        Calculate option Greeks with dividend yield.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry in years
            r: Risk-free rate
            sigma: Volatility
            is_call: True for call, False for put
            q: Continuous dividend yield

        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        if T <= 0 or sigma <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Discount factors
        discount_factor = math.exp(-r * T)
        dividend_discount = math.exp(-q * T)

        # Delta (adjusted for dividend yield)
        if is_call:
            delta = dividend_discount * norm.cdf(d1)
        else:
            delta = -dividend_discount * norm.cdf(-d1)

        # Gamma (same formula, but d1 uses dividend-adjusted drift)
        gamma = dividend_discount * norm.pdf(d1) / (S * sigma * sqrt_T)

        # Theta (per day) - more complex with dividends
        theta_term1 = -S * dividend_discount * norm.pdf(d1) * sigma / (2 * sqrt_T)
        if is_call:
            theta_term2 = -r * K * discount_factor * norm.cdf(d2)
            theta_term3 = q * S * dividend_discount * norm.cdf(d1)
        else:
            theta_term2 = r * K * discount_factor * norm.cdf(-d2)
            theta_term3 = -q * S * dividend_discount * norm.cdf(-d1)
        theta = (theta_term1 + theta_term2 + theta_term3) / 365

        # Vega (per 1% vol change)
        vega = S * dividend_discount * sqrt_T * norm.pdf(d1) / 100

        # Rho (per 1% rate change)
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
        Calculate implied volatility using Newton-Raphson.

        Args:
            price: Market price of the option
            S: Spot price
            K: Strike price
            T: Time to expiry in years
            r: Risk-free rate
            is_call: True for call, False for put
            q: Continuous dividend yield
            precision: Price precision for convergence
            max_iterations: Maximum iterations

        Returns:
            Implied volatility

        TODO: Use more robust method (Brent's, bisection with bounds).
        """
        sigma = 0.2  # Initial guess

        for _ in range(max_iterations):
            bs_price = self.black_scholes_price(S, K, T, r, sigma, is_call, q)
            vega = self.calculate_greeks(S, K, T, r, sigma, is_call, q)["vega"] * 100

            if abs(vega) < 1e-8:
                break

            sigma = sigma - (bs_price - price) / vega

            if sigma <= 0:
                sigma = 0.01

            if abs(bs_price - price) < precision:
                break

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

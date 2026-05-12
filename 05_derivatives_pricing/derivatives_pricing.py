"""
Derivatives Pricing Module

This module provides analytical and numerical pricing tools for common derivatives:
- Black-Scholes pricing and Greeks for European options
- Binomial tree pricing for European and American options
- Monte Carlo pricing for European options
- Black-76 pricing for options on forwards/futures
- Term structure models (Vasicek and CIR) for zero-coupon bonds

Author: Kevin J.D. Metzler
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


@dataclass
class OptionParams:
    """Parameters for equity-style options."""
    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0
    option_type: str = "call"

    def __post_init__(self) -> None:
        if self.spot <= 0 or self.strike <= 0:
            raise ValueError("Spot and strike must be positive.")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive.")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive.")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'.")


def forward_price(spot: float, rate: float, dividend_yield: float, maturity: float) -> float:
    """Forward price under continuous dividend yield."""
    return spot * np.exp((rate - dividend_yield) * maturity)


def put_call_parity(params: OptionParams, call_price: float, put_price: float) -> float:
    """Compute parity residual: C - P - (S e^{-qT} - K e^{-rT})."""
    parity = params.spot * np.exp(-params.dividend_yield * params.maturity)
    parity -= params.strike * np.exp(-params.rate * params.maturity)
    return call_price - put_price - parity


class BlackScholesPricer:
    """Black-Scholes pricing for European options with continuous dividend yield."""

    def __init__(self, params: OptionParams):
        self.params = params

    def _d1_d2(self) -> Tuple[float, float]:
        p = self.params
        d1 = (
            np.log(p.spot / p.strike)
            + (p.rate - p.dividend_yield + 0.5 * p.volatility ** 2) * p.maturity
        ) / (p.volatility * np.sqrt(p.maturity))
        d2 = d1 - p.volatility * np.sqrt(p.maturity)
        return d1, d2

    def price(self) -> float:
        p = self.params
        d1, d2 = self._d1_d2()
        discount_r = np.exp(-p.rate * p.maturity)
        discount_q = np.exp(-p.dividend_yield * p.maturity)

        if p.option_type == "call":
            return p.spot * discount_q * norm.cdf(d1) - p.strike * discount_r * norm.cdf(d2)
        return p.strike * discount_r * norm.cdf(-d2) - p.spot * discount_q * norm.cdf(-d1)

    def greeks(self) -> Dict[str, float]:
        p = self.params
        d1, d2 = self._d1_d2()
        discount_r = np.exp(-p.rate * p.maturity)
        discount_q = np.exp(-p.dividend_yield * p.maturity)
        pdf_d1 = norm.pdf(d1)

        delta = discount_q * norm.cdf(d1) if p.option_type == "call" else discount_q * (norm.cdf(d1) - 1)
        gamma = discount_q * pdf_d1 / (p.spot * p.volatility * np.sqrt(p.maturity))
        vega = p.spot * discount_q * pdf_d1 * np.sqrt(p.maturity)

        if p.option_type == "call":
            theta = (
                -p.spot * discount_q * pdf_d1 * p.volatility / (2 * np.sqrt(p.maturity))
                - p.rate * p.strike * discount_r * norm.cdf(d2)
                + p.dividend_yield * p.spot * discount_q * norm.cdf(d1)
            )
            rho = p.strike * p.maturity * discount_r * norm.cdf(d2)
        else:
            theta = (
                -p.spot * discount_q * pdf_d1 * p.volatility / (2 * np.sqrt(p.maturity))
                + p.rate * p.strike * discount_r * norm.cdf(-d2)
                - p.dividend_yield * p.spot * discount_q * norm.cdf(-d1)
            )
            rho = -p.strike * p.maturity * discount_r * norm.cdf(-d2)

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }

    @staticmethod
    def implied_volatility(
        market_price: float,
        params: OptionParams,
        vol_lower: float = 1e-6,
        vol_upper: float = 5.0,
    ) -> float:
        """Solve for implied volatility using a bracketing root-finder."""
        if market_price <= 0:
            raise ValueError("Market price must be positive.")

        def objective(vol: float) -> float:
            trial = OptionParams(
                spot=params.spot,
                strike=params.strike,
                maturity=params.maturity,
                rate=params.rate,
                volatility=vol,
                dividend_yield=params.dividend_yield,
                option_type=params.option_type,
            )
            return BlackScholesPricer(trial).price() - market_price

        lower_val = objective(vol_lower)
        upper_val = objective(vol_upper)
        if lower_val * upper_val > 0:
            raise ValueError("Implied volatility not bracketed. Adjust bounds.")

        return brentq(objective, vol_lower, vol_upper, maxiter=200)


def black_76_price(forward: float, strike: float, maturity: float, rate: float,
                   volatility: float, option_type: str = "call") -> float:
    """Black-76 price for options on forwards/futures."""
    if forward <= 0 or strike <= 0 or maturity <= 0 or volatility <= 0:
        raise ValueError("Inputs must be positive.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    d1 = (np.log(forward / strike) + 0.5 * volatility ** 2 * maturity) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    discount = np.exp(-rate * maturity)

    if option_type == "call":
        return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


class BinomialTreePricer:
    """Cox-Ross-Rubinstein binomial tree for European and American options."""

    def __init__(self, params: OptionParams, steps: int = 200, is_american: bool = False):
        if steps <= 0:
            raise ValueError("steps must be positive.")
        self.params = params
        self.steps = steps
        self.is_american = is_american

    def price(self) -> float:
        p = self.params
        dt = p.maturity / self.steps
        u = np.exp(p.volatility * np.sqrt(dt))
        d = 1 / u
        disc = np.exp(-p.rate * dt)
        p_up = (np.exp((p.rate - p.dividend_yield) * dt) - d) / (u - d)
        if p_up <= 0 or p_up >= 1:
            raise ValueError("Arbitrage detected in tree parameters.")

        # Terminal payoffs
        spots = p.spot * (u ** np.arange(self.steps, -1, -1)) * (d ** np.arange(0, self.steps + 1))
        if p.option_type == "call":
            values = np.maximum(spots - p.strike, 0.0)
        else:
            values = np.maximum(p.strike - spots, 0.0)

        # Backward induction
        for step in range(self.steps - 1, -1, -1):
            values = disc * (p_up * values[:-1] + (1 - p_up) * values[1:])
            if self.is_american:
                spots = p.spot * (u ** np.arange(step, -1, -1)) * (d ** np.arange(0, step + 1))
                if p.option_type == "call":
                    exercise = np.maximum(spots - p.strike, 0.0)
                else:
                    exercise = np.maximum(p.strike - spots, 0.0)
                values = np.maximum(values, exercise)

        return float(values[0])


class MonteCarloPricer:
    """Monte Carlo pricing for European options under GBM."""

    def __init__(self, params: OptionParams, n_paths: int = 10000, n_steps: int = 1,
                 seed: Optional[int] = 42, antithetic: bool = True):
        if n_paths <= 0 or n_steps <= 0:
            raise ValueError("n_paths and n_steps must be positive.")
        self.params = params
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.antithetic = antithetic

    def _simulate_terminal(self) -> np.ndarray:
        p = self.params
        dt = p.maturity / self.n_steps
        drift = (p.rate - p.dividend_yield - 0.5 * p.volatility ** 2) * dt
        diffusion = p.volatility * np.sqrt(dt)

        rng = np.random.default_rng(self.seed)
        n = self.n_paths
        if self.antithetic:
            n_half = n // 2
            z = rng.standard_normal((n_half, self.n_steps))
            z = np.vstack([z, -z])
        else:
            z = rng.standard_normal((n, self.n_steps))

        increments = drift + diffusion * z
        log_paths = np.cumsum(increments, axis=1)
        terminal = p.spot * np.exp(log_paths[:, -1])
        return terminal

    def price(self) -> Tuple[float, float]:
        p = self.params
        terminal = self._simulate_terminal()
        if p.option_type == "call":
            payoffs = np.maximum(terminal - p.strike, 0.0)
        else:
            payoffs = np.maximum(p.strike - terminal, 0.0)

        discount = np.exp(-p.rate * p.maturity)
        price = discount * np.mean(payoffs)
        stderr = discount * np.std(payoffs, ddof=1) / np.sqrt(len(payoffs))
        return float(price), float(stderr)


class VasicekModel:
    """Vasicek short-rate model for zero-coupon bond pricing."""

    def __init__(self, a: float, b: float, sigma: float, r0: float):
        if a <= 0 or sigma <= 0:
            raise ValueError("a and sigma must be positive.")
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0

    def zero_coupon_bond_price(self, maturity: float) -> float:
        a, b, sigma, r0 = self.a, self.b, self.sigma, self.r0
        if maturity <= 0:
            raise ValueError("maturity must be positive.")
        B = (1 - np.exp(-a * maturity)) / a
        A = np.exp(
            (b - sigma ** 2 / (2 * a ** 2)) * (B - maturity)
            - (sigma ** 2) * (B ** 2) / (4 * a)
        )
        return float(A * np.exp(-B * r0))


class CIRModel:
    """Cox-Ingersoll-Ross (CIR) short-rate model for zero-coupon bonds."""

    def __init__(self, a: float, b: float, sigma: float, r0: float):
        if a <= 0 or sigma <= 0 or b <= 0:
            raise ValueError("a, b, and sigma must be positive.")
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0

    def zero_coupon_bond_price(self, maturity: float) -> float:
        a, b, sigma, r0 = self.a, self.b, self.sigma, self.r0
        if maturity <= 0:
            raise ValueError("maturity must be positive.")
        h = np.sqrt(a ** 2 + 2 * sigma ** 2)
        exp_hT = np.exp(h * maturity)
        numerator = 2 * h * np.exp((a + h) * maturity / 2)
        denominator = 2 * h + (a + h) * (exp_hT - 1)
        A = (numerator / denominator) ** (2 * a * b / sigma ** 2)
        B = 2 * (exp_hT - 1) / denominator
        return float(A * np.exp(-B * r0))

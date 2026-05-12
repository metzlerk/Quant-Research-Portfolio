#!/usr/bin/env python3
"""
Derivatives Pricing Module Example

This script demonstrates basic usage of the derivatives pricing module.
"""

import numpy as np
from derivatives_pricing import (
    OptionParams,
    BlackScholesPricer,
    BinomialTreePricer,
    MonteCarloPricer,
    VasicekModel,
    CIRModel,
    black_76_price,
)


def run_examples() -> None:
    print("Derivatives Pricing Module Demonstration")
    print("=" * 60)

    # Black-Scholes
    params = OptionParams(
        spot=100,
        strike=100,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        dividend_yield=0.01,
        option_type="call",
    )
    bs = BlackScholesPricer(params)
    price = bs.price()
    greeks = bs.greeks()
    print("\nBlack-Scholes European Call:")
    print(f"  Price: {price:.4f}")
    print(f"  Delta: {greeks['delta']:.4f}")

    # Binomial tree (American put)
    params_put = OptionParams(
        spot=50,
        strike=55,
        maturity=0.5,
        rate=0.03,
        volatility=0.25,
        dividend_yield=0.0,
        option_type="put",
    )
    tree = BinomialTreePricer(params_put, steps=300, is_american=True)
    tree_price = tree.price()
    print("\nBinomial Tree American Put:")
    print(f"  Price: {tree_price:.4f}")

    # Monte Carlo
    mc = MonteCarloPricer(params, n_paths=20000, n_steps=1, seed=42)
    mc_price, stderr = mc.price()
    print("\nMonte Carlo European Call:")
    print(f"  Price: {mc_price:.4f} (SE: {stderr:.4f})")

    # Black-76
    fwd_price = 100
    black76 = black_76_price(forward=fwd_price, strike=100, maturity=1.0, rate=0.05, volatility=0.2)
    print("\nBlack-76 Call on Forward:")
    print(f"  Price: {black76:.4f}")

    # Term structure models
    vasicek = VasicekModel(a=0.5, b=0.03, sigma=0.01, r0=0.02)
    cir = CIRModel(a=0.6, b=0.04, sigma=0.08, r0=0.03)
    print("\nZero-Coupon Bond Prices (5y):")
    print(f"  Vasicek: {vasicek.zero_coupon_bond_price(5.0):.4f}")
    print(f"  CIR: {cir.zero_coupon_bond_price(5.0):.4f}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    run_examples()

# Derivatives Pricing

## Overview

This module provides analytical and numerical methods for pricing derivatives. It focuses on:

1. **European Options**: Black-Scholes pricing and Greeks
2. **American Options**: Binomial tree pricing
3. **Monte Carlo Simulation**: GBM-based pricing for European options
4. **Term Structure Models**: Vasicek and CIR zero-coupon bond pricing

## Mathematical Foundations

### Black-Scholes Model

For a European call option with continuous dividend yield \(q\):

$$C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

with:

$$d_1 = \frac{\ln(S_0 / K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

Put prices follow by symmetry.

### Binomial Tree

The Cox-Ross-Rubinstein tree uses:

$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u$$

and risk-neutral probability:

$$p = \frac{e^{(r-q)\Delta t} - d}{u - d}$$

### Monte Carlo Pricing

Under GBM:

$$S_T = S_0 \exp\left((r-q-\tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}Z\right)$$

Monte Carlo estimates use discounted average payoffs.

### Term Structure Models

Short-rate models provide closed-form zero-coupon bond prices:

- **Vasicek**: Mean-reverting Gaussian rates
- **CIR**: Mean-reverting square-root diffusion rates

## Implementation Details

### Key Classes and Functions

1. **`OptionParams`**: Parameter container for option contracts
2. **`BlackScholesPricer`**: Price, Greeks, and implied volatility
3. **`BinomialTreePricer`**: European and American pricing
4. **`MonteCarloPricer`**: Simulation-based European pricing
5. **`VasicekModel`** and **`CIRModel`**: Zero-coupon bond pricing

## Usage Examples

## Notebook

For a full walkthrough, see the Jupyter notebook:

- `05_derivatives_pricing/derivatives_pricing_analysis.ipynb`

### Black-Scholes Pricing

```python
from derivatives_pricing import OptionParams, BlackScholesPricer

params = OptionParams(
    spot=100,
    strike=100,
    maturity=1.0,
    rate=0.05,
    volatility=0.2,
    dividend_yield=0.0,
    option_type="call"
)

pricer = BlackScholesPricer(params)
price = pricer.price()
greeks = pricer.greeks()
```

### Binomial Tree (American Put)

```python
from derivatives_pricing import OptionParams, BinomialTreePricer

params = OptionParams(
    spot=50,
    strike=55,
    maturity=0.5,
    rate=0.03,
    volatility=0.25,
    dividend_yield=0.0,
    option_type="put"
)

tree = BinomialTreePricer(params, steps=300, is_american=True)
price = tree.price()
```

### Monte Carlo Pricing

```python
from derivatives_pricing import OptionParams, MonteCarloPricer

params = OptionParams(
    spot=100,
    strike=95,
    maturity=1.0,
    rate=0.04,
    volatility=0.2,
    dividend_yield=0.01,
    option_type="call"
)

mc = MonteCarloPricer(params, n_paths=20000, n_steps=1)
price, stderr = mc.price()
```

### Zero-Coupon Bonds

```python
from derivatives_pricing import VasicekModel, CIRModel

vasicek = VasicekModel(a=0.5, b=0.03, sigma=0.01, r0=0.02)
price_v = vasicek.zero_coupon_bond_price(maturity=5.0)

cir = CIRModel(a=0.6, b=0.04, sigma=0.08, r0=0.03)
price_c = cir.zero_coupon_bond_price(maturity=5.0)
```

## References

1. Hull, J. (2018). Options, Futures, and Other Derivatives. Pearson.
2. Björk, T. (2009). Arbitrage Theory in Continuous Time. Oxford University Press.
3. Shreve, S. (2004). Stochastic Calculus for Finance II. Springer.

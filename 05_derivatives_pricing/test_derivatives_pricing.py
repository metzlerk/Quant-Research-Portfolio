"""
Unit tests for derivatives pricing module.
"""

import unittest
import numpy as np
import sys
import os

# Add module directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from derivatives_pricing import (
    OptionParams,
    BlackScholesPricer,
    BinomialTreePricer,
    MonteCarloPricer,
    VasicekModel,
    CIRModel,
    black_76_price,
    put_call_parity,
)


class TestBlackScholes(unittest.TestCase):
    def test_black_scholes_call_price(self):
        params = OptionParams(
            spot=100, strike=100, maturity=1.0, rate=0.05, volatility=0.2, option_type="call"
        )
        price = BlackScholesPricer(params).price()
        self.assertAlmostEqual(price, 10.4506, places=2)

    def test_put_call_parity(self):
        params = OptionParams(
            spot=100, strike=100, maturity=1.0, rate=0.05, volatility=0.2, option_type="call"
        )
        call = BlackScholesPricer(params).price()
        put_params = OptionParams(
            spot=100, strike=100, maturity=1.0, rate=0.05, volatility=0.2, option_type="put"
        )
        put = BlackScholesPricer(put_params).price()
        residual = put_call_parity(params, call, put)
        self.assertAlmostEqual(residual, 0.0, places=6)

    def test_implied_volatility(self):
        params = OptionParams(
            spot=100, strike=105, maturity=0.75, rate=0.03, volatility=0.25, option_type="call"
        )
        market_price = BlackScholesPricer(params).price()
        implied = BlackScholesPricer.implied_volatility(market_price, params)
        self.assertAlmostEqual(implied, 0.25, places=3)


class TestBinomialTree(unittest.TestCase):
    def test_binomial_tree_convergence(self):
        params = OptionParams(
            spot=100, strike=100, maturity=1.0, rate=0.05, volatility=0.2, option_type="call"
        )
        bs_price = BlackScholesPricer(params).price()
        tree_price = BinomialTreePricer(params, steps=300, is_american=False).price()
        self.assertLess(abs(tree_price - bs_price), 0.5)


class TestMonteCarlo(unittest.TestCase):
    def test_monte_carlo_price(self):
        params = OptionParams(
            spot=100, strike=100, maturity=1.0, rate=0.05, volatility=0.2, option_type="call"
        )
        bs_price = BlackScholesPricer(params).price()
        mc_price, stderr = MonteCarloPricer(params, n_paths=20000, seed=7).price()
        self.assertLess(abs(mc_price - bs_price), 1.0)
        self.assertGreater(stderr, 0)


class TestBlack76(unittest.TestCase):
    def test_black_76_parity(self):
        forward = 100
        strike = 100
        maturity = 1.0
        rate = 0.05
        sigma = 0.2
        call = black_76_price(forward, strike, maturity, rate, sigma, option_type="call")
        put = black_76_price(forward, strike, maturity, rate, sigma, option_type="put")
        parity = (call - put) - np.exp(-rate * maturity) * (forward - strike)
        self.assertAlmostEqual(parity, 0.0, places=6)


class TestTermStructure(unittest.TestCase):
    def test_vasicek_cir_bond_prices(self):
        vasicek = VasicekModel(a=0.5, b=0.03, sigma=0.01, r0=0.02)
        cir = CIRModel(a=0.6, b=0.04, sigma=0.08, r0=0.03)

        v_short = vasicek.zero_coupon_bond_price(1.0)
        v_long = vasicek.zero_coupon_bond_price(5.0)
        c_short = cir.zero_coupon_bond_price(1.0)
        c_long = cir.zero_coupon_bond_price(5.0)

        self.assertTrue(0 < v_short <= 1)
        self.assertTrue(0 < v_long <= 1)
        self.assertTrue(0 < c_short <= 1)
        self.assertTrue(0 < c_long <= 1)
        self.assertGreater(v_short, v_long)
        self.assertGreater(c_short, c_long)


if __name__ == "__main__":
    unittest.main(verbosity=2)

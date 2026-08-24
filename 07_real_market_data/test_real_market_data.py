"""
Unit tests for the real market data validation module.

Tests are split into two groups:
- Offline tests using synthetic data for the statistical/backtesting logic
  (Kupiec test, rolling VaR backtest, rolling GARCH backtest). These never
  touch the network and are the ones that matter for CI correctness.
- A small number of live-network integration tests against Yahoo Finance,
  which are skipped automatically if the network is unavailable rather than
  failing the suite.
"""

import unittest
import sys
import os

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_market_data import (
    kupiec_pof_test,
    rolling_var_backtest,
    rolling_garch_vol_backtest,
    fetch_close_prices,
    fetch_option_chain,
    implied_vol_smile,
)


def _network_available() -> bool:
    import socket
    try:
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=3).close()
        return True
    except OSError:
        return False


NETWORK_AVAILABLE = _network_available()


class TestKupiecPofTest(unittest.TestCase):
    def test_well_calibrated_model_is_not_rejected(self):
        # Exactly the expected 5% violation rate at 95% confidence.
        result = kupiec_pof_test(n_obs=1000, n_violations=50, confidence=0.95)
        self.assertAlmostEqual(result["observed_violation_rate"], 0.05, places=3)
        self.assertFalse(result["reject_null_at_5pct"])
        self.assertGreater(result["p_value"], 0.05)

    def test_badly_miscalibrated_model_is_rejected(self):
        # Way too many violations for a 95% VaR (20% instead of 5%).
        result = kupiec_pof_test(n_obs=1000, n_violations=200, confidence=0.95)
        self.assertTrue(result["reject_null_at_5pct"])
        self.assertLess(result["p_value"], 0.05)

    def test_zero_violations_does_not_crash(self):
        result = kupiec_pof_test(n_obs=200, n_violations=0, confidence=0.99)
        self.assertEqual(result["n_violations"], 0)
        self.assertTrue(np.isfinite(result["lr_statistic"]))


class TestRollingVarBacktest(unittest.TestCase):
    def test_violation_rate_matches_confidence_for_iid_normal_returns(self):
        rng = np.random.default_rng(0)
        n = 1500
        dates = pd.date_range('2018-01-01', periods=n, freq='B')
        returns = pd.Series(rng.normal(0, 0.01, n), index=dates)

        backtest = rolling_var_backtest(returns, window=250, confidence=0.95, method='historical')
        self.assertFalse(backtest.empty)
        self.assertIn('violation', backtest.columns)

        violation_rate = backtest['violation'].mean()
        # For truly IID normal returns and a historical VaR, the realized
        # violation rate should land in the right ballpark of 5%.
        self.assertLess(abs(violation_rate - 0.05), 0.03)

    def test_var_estimates_are_negative_for_typical_returns(self):
        rng = np.random.default_rng(1)
        n = 400
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        returns = pd.Series(rng.normal(0.0005, 0.015, n), index=dates)

        backtest = rolling_var_backtest(returns, window=250, confidence=0.95)
        self.assertTrue((backtest['var_estimate'] < 0).all())


class TestRollingGarchVolBacktest(unittest.TestCase):
    def test_produces_forecast_and_realized_columns(self):
        # Simulate a short GARCH(1,1)-like series; use a small window/horizon
        # to keep the walk-forward refits fast for a unit test.
        rng = np.random.default_rng(42)
        n = 700
        returns = np.zeros(n)
        sigma2 = 0.0001
        for t in range(n):
            if t > 0:
                sigma2 = 0.00001 + 0.08 * returns[t-1]**2 + 0.85 * sigma2
            returns[t] = rng.normal(0, np.sqrt(sigma2))
        dates = pd.date_range('2019-01-01', periods=n, freq='B')
        returns = pd.Series(returns, index=dates)

        backtest = rolling_garch_vol_backtest(
            returns, window=300, refit_every=50, forecast_horizon=10
        )
        self.assertFalse(backtest.empty)
        for col in ('forecast_vol_annualized', 'realized_vol_annualized', 'forecast_error'):
            self.assertIn(col, backtest.columns)
        # Volatility forecasts should be positive, finite, and in a sane
        # annualized range for a daily-vol process (not exploding to inf).
        self.assertTrue((backtest['forecast_vol_annualized'] > 0).all())
        self.assertTrue(np.isfinite(backtest['forecast_vol_annualized']).all())
        self.assertTrue((backtest['forecast_vol_annualized'] < 5.0).all())


@unittest.skipUnless(NETWORK_AVAILABLE, "No network access to Yahoo Finance")
class TestLiveDataIntegration(unittest.TestCase):
    def test_fetch_close_prices_returns_real_data(self):
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        prices = fetch_close_prices(
            ['SPY'], '2024-01-01', '2024-02-01', data_dir=tmp_dir, use_cache=True
        )
        self.assertIn('SPY', prices.columns)
        self.assertGreater(len(prices), 10)
        self.assertTrue((prices['SPY'] > 0).all())

    def test_fetch_option_chain_and_implied_vol_smile(self):
        expiry, spot, calls, puts = fetch_option_chain('SPY', min_days_out=25)
        self.assertGreater(spot, 0)
        self.assertFalse(calls.empty)
        days_out = (pd.Timestamp(expiry) - pd.Timestamp.now().normalize()).days
        self.assertGreaterEqual(days_out, 25)

        as_of = pd.Timestamp.now(tz=None).normalize()
        smile = implied_vol_smile(
            calls, spot=spot, expiry=expiry, as_of=as_of,
            rate=0.04, option_type='call', min_volume=1
        )
        if not smile.empty:
            self.assertTrue((smile['model_implied_vol'] > 0).all())
            self.assertTrue((smile['model_implied_vol'] < 5.0).all())


if __name__ == '__main__':
    unittest.main()

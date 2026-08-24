"""
Unit tests for trading strategies module.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add module directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_strategies import (
    StrategyConfig,
    MeanReversionStrategy,
    MomentumStrategy,
    StatisticalArbitrageStrategy,
)


def make_price_data(n_assets=4, n_periods=400, seed=42):
    """Generate correlated GBM price series for backtests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-01', periods=n_periods, freq='B')
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    returns = rng.normal(0.0003, 0.015, size=(n_periods, n_assets))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def make_ou_series(n=500, theta=0.05, mu=np.log(100), sigma=0.02, seed=7):
    """Simulate a discretized Ornstein-Uhlenbeck process in log-price space."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = x[t-1] + theta * (mu - x[t-1]) + rng.normal(0, sigma)
    dates = pd.date_range('2022-01-01', periods=n, freq='B')
    return pd.Series(np.exp(x), index=dates)


class TestStrategyConfig(unittest.TestCase):
    def test_defaults(self):
        config = StrategyConfig(name='test')
        self.assertEqual(config.lookback_period, 252)
        self.assertEqual(config.holding_period, 22)
        self.assertEqual(config.transaction_cost, 0.001)
        self.assertIsNone(config.stop_loss)


class TestMeanReversionStrategy(unittest.TestCase):
    def setUp(self):
        self.config = StrategyConfig(
            name='mean_reversion', lookback_period=100,
            transaction_cost=0.001, max_position_size=0.2
        )
        self.strategy = MeanReversionStrategy(self.config, entry_threshold=1.5, exit_threshold=0.5)

    def test_fit_ou_process_recovers_mean_reverting_parameters(self):
        series = make_ou_series(n=500, theta=0.08, mu=np.log(100), sigma=0.015)
        params = self.strategy.fit_ou_process(series)

        self.assertIn('theta', params)
        self.assertIn('mu', params)
        self.assertIn('half_life', params)
        # Strong mean reversion should be detected (theta > 0)
        self.assertGreater(params['theta'], 0)
        self.assertGreater(params['half_life'], 0)
        # Recovered long-run mean should be close to the true log-price mean
        self.assertAlmostEqual(params['mu'], np.log(100), delta=0.1)

    def test_generate_signals_shape_and_range(self):
        prices = make_price_data(n_assets=2, n_periods=200)
        signals = self.strategy.generate_signals(prices)

        self.assertEqual(signals.shape, prices.shape)
        # Signals should be bounded in {-1, 0, 1}
        unique_vals = pd.unique(signals.values.ravel())
        self.assertTrue(set(unique_vals).issubset({-1, 0, 1}))

    def test_calculate_positions_respects_max_size(self):
        prices = make_price_data(n_assets=2, n_periods=200)
        signals = self.strategy.generate_signals(prices)
        positions = self.strategy.calculate_positions(signals)

        self.assertTrue((positions.abs() <= self.config.max_position_size + 1e-9).all().all())

    def test_backtest_returns_expected_keys(self):
        prices = make_price_data(n_assets=2, n_periods=200)
        result = self.strategy.backtest(prices)

        for key in ('signals', 'positions', 'returns', 'performance'):
            self.assertIn(key, result)
        perf = result['performance']
        for key in ('sharpe_ratio', 'max_drawdown', 'annualized_return', 'volatility'):
            self.assertIn(key, perf)
        self.assertLessEqual(perf['max_drawdown'], 0)


class TestMomentumStrategy(unittest.TestCase):
    def setUp(self):
        self.config = StrategyConfig(
            name='momentum', lookback_period=60, holding_period=22,
            max_position_size=0.3
        )

    def test_rsi_bounded_between_0_and_100(self):
        strategy = MomentumStrategy(self.config, use_ml_enhancement=False)
        prices = make_price_data(n_assets=1, n_periods=200)
        returns = prices.iloc[:, 0].pct_change().dropna()

        rsi = strategy._calculate_rsi(returns, window=14).dropna()
        self.assertTrue((rsi >= 0).all())
        self.assertTrue((rsi <= 100).all())

    def test_macd_returns_series_same_length(self):
        strategy = MomentumStrategy(self.config, use_ml_enhancement=False)
        prices = make_price_data(n_assets=1, n_periods=200)
        returns = prices.iloc[:, 0].pct_change().dropna()

        macd = strategy._calculate_macd(returns)
        self.assertEqual(len(macd), len(returns))

    def test_calculate_momentum_features_no_lookahead_columns(self):
        strategy = MomentumStrategy(self.config, momentum_periods=[1, 3], use_ml_enhancement=False)
        prices = make_price_data(n_assets=2, n_periods=250)
        returns = prices.pct_change().dropna()

        features = strategy.calculate_momentum_features(returns)
        self.assertFalse(features.empty)
        self.assertTrue(any('mom_1m' in c for c in features.columns))
        self.assertTrue(any('rsi' in c for c in features.columns))

    def test_generate_signals_without_ml_enhancement(self):
        strategy = MomentumStrategy(self.config, momentum_periods=[1, 3], use_ml_enhancement=False)
        prices = make_price_data(n_assets=4, n_periods=250)

        signals = strategy.generate_signals(prices)
        self.assertEqual(signals.shape, prices.shape)
        unique_vals = pd.unique(signals.values.ravel())
        self.assertTrue(set(unique_vals).issubset({-1, 0, 1}))


class TestStatisticalArbitrageStrategy(unittest.TestCase):
    def setUp(self):
        self.config = StrategyConfig(
            name='stat_arb', lookback_period=100, holding_period=5,
            max_position_size=0.4
        )
        self.strategy = StatisticalArbitrageStrategy(self.config, cointegration_lookback=100)

    def test_cointegration_detected_for_constructed_pair(self):
        rng = np.random.default_rng(1)
        n = 300
        dates = pd.date_range('2022-01-01', periods=n, freq='B')
        common_walk = np.cumsum(rng.normal(0, 1, n))
        price1 = pd.Series(100 + common_walk + rng.normal(0, 0.5, n), index=dates)
        price2 = pd.Series(50 + 0.5 * common_walk + rng.normal(0, 0.5, n), index=dates)

        result = self.strategy.test_cointegration(price1, price2)
        self.assertIn('cointegrated', result)
        self.assertIn('p_value', result)
        self.assertTrue(result['cointegrated'])

    def test_cointegration_rejected_for_independent_series(self):
        rng = np.random.default_rng(2)
        n = 300
        dates = pd.date_range('2022-01-01', periods=n, freq='B')
        price1 = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=dates)
        price2 = pd.Series(50 + np.cumsum(rng.normal(0, 1, n)), index=dates)

        result = self.strategy.test_cointegration(price1, price2)
        self.assertFalse(result['cointegrated'])

    def test_insufficient_data_returns_default(self):
        short_series = pd.Series(np.arange(10, dtype=float))
        result = self.strategy.test_cointegration(short_series, short_series)
        self.assertFalse(result['cointegrated'])
        self.assertEqual(result['p_value'], 1.0)

    def test_calculate_positions_within_bounds(self):
        n = 5
        dates = pd.date_range('2022-01-01', periods=n, freq='B')
        signals = pd.DataFrame(
            {'A': [0.9, -0.9, 0.2, 0.0, 0.5], 'B': [0.9, 0.9, -0.2, 0.0, -0.5]},
            index=dates
        )
        positions = self.strategy.calculate_positions(signals)
        self.assertTrue((positions.abs() <= self.config.max_position_size + 1e-9).all().all())


if __name__ == '__main__':
    unittest.main()

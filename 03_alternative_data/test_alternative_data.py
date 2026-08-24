"""
Unit tests for alternative data module.
"""

import unittest
import shutil
import tempfile
import numpy as np
import pandas as pd
import sys
import os

# Add module directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alternative_data import AlternativeDataConfig, BaseAlternativeData, NewsSentimentAnalyzer
from alternative_utils import (
    calculate_lead_lag_correlation,
    calculate_mutual_information,
    calculate_information_coefficient,
    backtest_signal_accuracy,
)


class TestAlternativeDataConfig(unittest.TestCase):
    def test_defaults(self):
        config = AlternativeDataConfig(data_dir='some_dir')
        self.assertTrue(config.cache_data)
        self.assertEqual(config.sentiment_window, 30)


class TestBaseAlternativeData(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = AlternativeDataConfig(data_dir=self.tmp_dir, cache_data=True)
        self.base = BaseAlternativeData(self.config)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cache_round_trip(self):
        self.base._save_to_cache('test_key', {'a': 1, 'b': [1, 2, 3]})
        loaded = self.base._load_from_cache('test_key')
        self.assertEqual(loaded, {'a': 1, 'b': [1, 2, 3]})

    def test_load_missing_cache_returns_none(self):
        self.assertIsNone(self.base._load_from_cache('does_not_exist'))

    def test_align_with_market_data_forward_fills(self):
        market_dates = pd.date_range('2022-01-01', periods=10, freq='D')
        market_data = pd.DataFrame({'Close': np.arange(10.0)}, index=market_dates)

        alt_dates = pd.date_range('2022-01-01', periods=3, freq='3D')
        alt_data = pd.DataFrame({'sentiment': [0.1, 0.5, -0.2]}, index=alt_dates)

        aligned = self.base.align_with_market_data(alt_data, market_data)
        self.assertEqual(len(aligned), len(market_data))
        self.assertFalse(aligned['sentiment'].isna().any())


class TestNewsSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        config = AlternativeDataConfig(data_dir=self.tmp_dir, cache_data=False, sentiment_window=3)
        self.analyzer = NewsSentimentAnalyzer(config)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_analyze_sentiment_raw_text_positive(self):
        result = self.analyzer.analyze_sentiment("This is fantastic, wonderful, amazing news for the company!")
        self.assertIn('compound', result)
        self.assertGreater(result['compound'], 0)

    def test_analyze_sentiment_raw_text_negative(self):
        result = self.analyzer.analyze_sentiment("This is terrible, awful, disastrous news for the company.")
        self.assertIn('compound', result)
        self.assertLess(result['compound'], 0)

    def test_analyze_sentiment_empty_articles(self):
        result = self.analyzer.analyze_sentiment([])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_analyze_sentiment_articles_aggregates_by_day(self):
        articles = [
            {'title': 'Great earnings beat', 'description': 'Company profits soar', 'content': '',
             'publishedAt': '2022-01-01T10:00:00Z'},
            {'title': 'Terrible losses reported', 'description': 'Stock crashes on bad news', 'content': '',
             'publishedAt': '2022-01-01T15:00:00Z'},
            {'title': 'Steady performance', 'description': 'In line with expectations', 'content': '',
             'publishedAt': '2022-01-02T09:00:00Z'},
        ]
        daily = self.analyzer.analyze_sentiment(articles)
        self.assertIsInstance(daily, pd.DataFrame)
        self.assertEqual(len(daily), 2)
        self.assertIn('compound', daily.columns)
        self.assertIn('sentiment_momentum', daily.columns)

    def test_analyze_sentiment_skips_invalid_dates(self):
        articles = [
            {'title': 'Some news', 'description': '', 'content': '', 'publishedAt': 'not-a-date'},
            {'title': 'Good outlook', 'description': 'positive growth', 'content': '',
             'publishedAt': '2022-01-01T10:00:00Z'},
        ]
        daily = self.analyzer.analyze_sentiment(articles)
        self.assertEqual(len(daily), 1)


class TestAlternativeUtils(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        dates = pd.date_range('2022-01-01', periods=200, freq='D')
        self.price = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)), index=dates)

    def test_lead_lag_correlation_finds_true_lead(self):
        # Feature leads price by 3 periods: feature[t] correlates with price[t+3]
        shifted = self.price.shift(-3)
        corr_df = calculate_lead_lag_correlation(self.price, shifted, max_lag=5, max_lead=5)

        self.assertIn('lag_lead', corr_df.columns)
        self.assertIn('correlation', corr_df.columns)
        best_row = corr_df.loc[corr_df['correlation'].abs().idxmax()]
        self.assertEqual(int(best_row['lag_lead']), -3)

    def test_mutual_information_ranks_informative_feature_higher(self):
        rng = np.random.default_rng(1)
        n = 300
        y = pd.Series(rng.normal(0, 1, n))
        informative = y * 2 + rng.normal(0, 0.1, n)
        noise = pd.Series(rng.normal(0, 1, n))
        X = pd.DataFrame({'informative': informative, 'noise': noise})

        mi = calculate_mutual_information(X, y)
        self.assertEqual(list(mi.index)[0], 'informative')

    def test_information_coefficient_perfect_correlation(self):
        predicted = pd.Series(np.arange(50, dtype=float))
        actual = pd.Series(np.arange(50, dtype=float) * 2)
        ic = calculate_information_coefficient(predicted, actual)
        self.assertAlmostEqual(ic, 1.0, places=6)

    def test_information_coefficient_insufficient_data(self):
        ic = calculate_information_coefficient(pd.Series([1.0]), pd.Series([1.0]))
        self.assertEqual(ic, 0.0)

    def test_backtest_signal_accuracy_perfect_signal(self):
        n = 100
        dates = pd.date_range('2022-01-01', periods=n, freq='D')
        returns = pd.Series(np.concatenate([np.full(50, 0.01), np.full(50, -0.01)]), index=dates)
        # Signal perfectly predicts the next-period return's sign
        signal = returns.shift(-1)

        result = backtest_signal_accuracy(signal, returns, lookahead_periods=1)
        self.assertIn('overall_accuracy', result)
        self.assertGreaterEqual(result['overall_accuracy'], 0.9)


if __name__ == '__main__':
    unittest.main()

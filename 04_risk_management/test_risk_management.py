"""
Unit tests for Risk Management module

Author: Kevin J.D. Metzler
Date: August 7, 2025
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_management import TailRiskModeler, BlackLittermanOptimizer, DynamicHedger, FactorModelBuilder
from risk_utils import calculate_var_es, decompose_risk, optimize_risk_parity


class TestTailRiskModeler(unittest.TestCase):
    """Test cases for TailRiskModeler class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        
        # Generate synthetic return data with some extreme values
        returns = np.random.normal(0, 0.02, 1000)
        returns[::100] = np.random.normal(-0.1, 0.05, 10)  # Add some extreme negative returns
        
        self.test_data = pd.DataFrame({'returns': returns}, index=dates)
        self.risk_model = TailRiskModeler(self.test_data)
    
    def test_pot_model_fitting(self):
        """Test POT model fitting."""
        params = self.risk_model.fit_evt_model('returns', method='pot')
        
        self.assertIsInstance(params, dict)
        self.assertIn('scale', params)
        self.assertIn('shape', params)
        self.assertIn('threshold', params)
        self.assertGreater(params['scale'], 0)
        self.assertGreater(params['n_exceedances'], 0)
    
    def test_block_maxima_fitting(self):
        """Test Block Maxima model fitting."""
        params = self.risk_model.fit_evt_model('returns', method='bm', block_size=22)
        
        self.assertIsInstance(params, dict)
        self.assertIn('location', params)
        self.assertIn('scale', params)
        self.assertIn('shape', params)
        self.assertGreater(params['scale'], 0)
        self.assertGreater(params['n_blocks'], 0)
    
    def test_var_es_calculation(self):
        """Test VaR and ES calculation."""
        # First fit a model
        self.risk_model.fit_evt_model('returns', method='pot')
        
        var, es = self.risk_model.calculate_var_es(confidence_level=0.95, method='pot')
        
        self.assertIsInstance(var, float)
        self.assertIsInstance(es, float)
        self.assertLess(var, 0)  # VaR should be negative (loss)
        self.assertLess(es, var)  # ES should be more extreme than VaR
    
    def test_tail_scenario_generation(self):
        """Test tail scenario generation."""
        # First fit a model
        self.risk_model.fit_evt_model('returns', method='pot')
        
        scenarios = self.risk_model.generate_tail_scenarios(100, method='pot')
        
        self.assertEqual(len(scenarios), 100)
        self.assertTrue(all(scenarios < 0))  # All scenarios should be negative (losses)


class TestBlackLittermanOptimizer(unittest.TestCase):
    """Test cases for BlackLittermanOptimizer class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        # Generate correlated returns for 5 assets
        n_assets = 5
        cov_matrix = np.random.rand(n_assets, n_assets)
        cov_matrix = cov_matrix @ cov_matrix.T  # Make positive definite
        
        returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=cov_matrix * 0.0001,  # Scale down for daily returns
            size=500
        )
        
        asset_names = [f'Asset_{i}' for i in range(n_assets)]
        self.test_returns = pd.DataFrame(returns, index=dates, columns=asset_names)
        self.market_weights = pd.Series(1/n_assets, index=asset_names)
        
        self.bl_optimizer = BlackLittermanOptimizer(self.test_returns, self.market_weights)
    
    def test_market_parameter_estimation(self):
        """Test market parameter estimation."""
        expected_returns, cov_matrix = self.bl_optimizer.estimate_market_parameters()
        
        self.assertIsInstance(expected_returns, pd.Series)
        self.assertIsInstance(cov_matrix, pd.DataFrame)
        self.assertEqual(len(expected_returns), len(self.test_returns.columns))
        self.assertEqual(cov_matrix.shape, (len(self.test_returns.columns), len(self.test_returns.columns)))
    
    def test_portfolio_optimization(self):
        """Test basic portfolio optimization."""
        weights = self.bl_optimizer.optimize(risk_aversion=3.0)
        
        self.assertIsInstance(weights, pd.Series)
        self.assertEqual(len(weights), len(self.test_returns.columns))
        self.assertAlmostEqual(weights.sum(), 1.0, places=6)
    
    def test_portfolio_analysis(self):
        """Test portfolio analysis."""
        weights = self.bl_optimizer.optimize(risk_aversion=3.0)
        analytics = self.bl_optimizer.analyze_portfolio(weights)
        
        self.assertIsInstance(analytics, dict)
        self.assertIn('expected_return', analytics)
        self.assertIn('volatility', analytics)
        self.assertIn('sharpe_ratio', analytics)
        self.assertGreater(analytics['volatility'], 0)


class TestDynamicHedger(unittest.TestCase):
    """Test cases for DynamicHedger class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        
        # Generate portfolio and hedging instrument returns
        portfolio_returns = np.random.normal(0.001, 0.02, 300)
        hedge1_returns = np.random.normal(0, 0.015, 300) - 0.5 * portfolio_returns + np.random.normal(0, 0.01, 300)
        hedge2_returns = np.random.normal(0, 0.02, 300) - 0.3 * portfolio_returns + np.random.normal(0, 0.015, 300)
        
        self.test_data = pd.DataFrame({
            'Portfolio': portfolio_returns,
            'Hedge1': hedge1_returns,
            'Hedge2': hedge2_returns
        }, index=dates)
        
        self.hedger = DynamicHedger(self.test_data)
    
    def test_minimum_variance_hedge(self):
        """Test minimum variance hedge calculation."""
        hedge_ratios = self.hedger.calculate_optimal_hedge(
            hedging_instruments=['Hedge1', 'Hedge2'],
            portfolio_col='Portfolio',
            method='minimum_variance',
            lookback_period=100,
            rebalance_frequency=20
        )
        
        self.assertIsInstance(hedge_ratios, pd.DataFrame)
        self.assertEqual(list(hedge_ratios.columns), ['Hedge1', 'Hedge2'])
        self.assertFalse(hedge_ratios.isna().all().all())
    
    def test_hedge_application(self):
        """Test hedge application and performance evaluation."""
        hedge_ratios = self.hedger.calculate_optimal_hedge(
            hedging_instruments=['Hedge1', 'Hedge2'],
            portfolio_col='Portfolio',
            method='minimum_variance',
            lookback_period=100,
            rebalance_frequency=20
        )
        
        hedged_performance = self.hedger.apply_hedge(hedge_ratios, 'Portfolio')
        
        self.assertIsInstance(hedged_performance, pd.DataFrame)
        self.assertIn('original_portfolio', hedged_performance.columns)
        self.assertIn('hedged_portfolio', hedged_performance.columns)
        
        # Evaluate hedge effectiveness
        effectiveness = self.hedger.evaluate_hedge_effectiveness(hedged_performance)
        self.assertIsInstance(effectiveness, dict)
        self.assertIn('hedge_effectiveness', effectiveness)
        self.assertIn('volatility_reduction', effectiveness)


class TestFactorModelBuilder(unittest.TestCase):
    """Test cases for FactorModelBuilder class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=400, freq='D')
        
        # Generate factor-based returns
        n_assets = 8
        n_factors = 3
        
        # Generate factor returns
        factor_returns = np.random.multivariate_normal(
            mean=np.zeros(n_factors),
            cov=np.eye(n_factors) * 0.0004,
            size=400
        )
        
        # Generate factor loadings
        factor_loadings = np.random.normal(0, 0.5, (n_assets, n_factors))
        
        # Generate asset returns from factor model
        asset_returns = factor_returns @ factor_loadings.T + np.random.normal(0, 0.01, (400, n_assets))
        
        asset_names = [f'Asset_{i}' for i in range(n_assets)]
        self.test_returns = pd.DataFrame(asset_returns, index=dates, columns=asset_names)
        self.factor_builder = FactorModelBuilder(self.test_returns)
    
    def test_statistical_factor_model(self):
        """Test statistical factor model (PCA)."""
        model = self.factor_builder.build_statistical_factor_model(n_factors=3)
        
        self.assertIsInstance(model, dict)
        self.assertIn('factor_loadings', model)
        self.assertIn('factor_returns', model)
        self.assertIn('explained_variance', model)
        self.assertEqual(model['factor_loadings'].shape[1], 3)
        self.assertEqual(len(model['explained_variance']), 3)
    
    def test_fundamental_factor_model(self):
        """Test fundamental factor model."""
        # Create dummy factor exposures
        exposures = pd.DataFrame(
            np.random.normal(0, 1, (len(self.test_returns.columns), 2)),
            index=self.test_returns.columns,
            columns=['Factor1', 'Factor2']
        )
        
        model = self.factor_builder.build_fundamental_factor_model(exposures)
        
        self.assertIsInstance(model, dict)
        self.assertIn('factor_loadings', model)
        self.assertIn('factor_returns', model)
        self.assertIn('specific_risks', model)


class TestRiskUtils(unittest.TestCase):
    """Test cases for risk utility functions."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        # Add some extreme values
        self.returns.iloc[::100] = np.random.normal(-0.05, 0.01, 10)
    
    def test_var_es_calculation(self):
        """Test VaR and ES calculation utilities."""
        results = calculate_var_es(self.returns, confidence_levels=[0.95, 0.99])
        
        self.assertIsInstance(results, dict)
        self.assertIn('CL_95%', results)
        self.assertIn('CL_99%', results)
        
        for cl_key in results:
            self.assertIn('VaR', results[cl_key])
            self.assertIn('ES', results[cl_key])
            self.assertLess(results[cl_key]['VaR'], 0)  # Should be negative
            self.assertLess(results[cl_key]['ES'], results[cl_key]['VaR'])  # ES more extreme
    
    def test_risk_parity_optimization(self):
        """Test risk parity optimization."""
        # Create a covariance matrix
        np.random.seed(42)
        n_assets = 5
        cov_matrix = np.random.rand(n_assets, n_assets)
        cov_matrix = cov_matrix @ cov_matrix.T
        cov_df = pd.DataFrame(cov_matrix, 
                             index=[f'Asset_{i}' for i in range(n_assets)],
                             columns=[f'Asset_{i}' for i in range(n_assets)])
        
        weights = optimize_risk_parity(cov_df, method='equal_risk_contribution')
        
        self.assertIsInstance(weights, pd.Series)
        self.assertEqual(len(weights), n_assets)
        self.assertAlmostEqual(weights.sum(), 1.0, places=6)
        self.assertTrue(all(weights >= 0))  # Should be long-only
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        # Create cumulative returns with some drawdowns
        cumulative_returns = (1 + self.returns).cumprod()
        
        from risk_utils import calculate_drawdowns
        drawdown_results = calculate_drawdowns(cumulative_returns)
        
        self.assertIsInstance(drawdown_results, dict)
        self.assertIn('max_drawdown', drawdown_results)
        self.assertIn('drawdown_series', drawdown_results)
        self.assertLessEqual(drawdown_results['max_drawdown'], 0)  # Should be negative


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)

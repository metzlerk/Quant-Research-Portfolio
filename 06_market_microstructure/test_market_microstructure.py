"""
Unit tests for market microstructure module.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add module directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from market_microstructure import (
    OrderBookSnapshot,
    LimitOrderBook,
    LinearImpactModel,
    PowerLawImpactModel,
    OptimalExecutionAC,
    HighFrequencyMetrics,
    microstructure_summary,
)


class TestOrderBookSnapshot(unittest.TestCase):
    def setUp(self):
        """Create sample order book snapshots."""
        self.bid_prices = np.array([100.00, 99.99, 99.98])
        self.bid_volumes = np.array([1000, 500, 300])
        self.ask_prices = np.array([100.01, 100.02, 100.03])
        self.ask_volumes = np.array([800, 400, 200])
        
        self.snapshot = OrderBookSnapshot(
            timestamp=0.0,
            bid_prices=self.bid_prices,
            bid_volumes=self.bid_volumes,
            ask_prices=self.ask_prices,
            ask_volumes=self.ask_volumes
        )
    
    def test_spread(self):
        """Test bid-ask spread calculation."""
        expected_spread = 100.01 - 100.00
        self.assertAlmostEqual(self.snapshot.spread, expected_spread, places=6)
    
    def test_spread_bps(self):
        """Test spread in basis points."""
        spread_bps = self.snapshot.spread_bps
        self.assertGreater(spread_bps, 0)
        self.assertLess(spread_bps, 100)  # less than 100 bps is reasonable
    
    def test_depth(self):
        """Test bid and ask depth calculations."""
        expected_bid_depth = 1000 + 500 + 300
        expected_ask_depth = 800 + 400 + 200
        self.assertEqual(self.snapshot.bid_depth, expected_bid_depth)
        self.assertEqual(self.snapshot.ask_depth, expected_ask_depth)
    
    def test_imbalance(self):
        """Test order imbalance calculation."""
        imbalance = self.snapshot.imbalance
        # bid_depth > ask_depth, so imbalance should be positive
        self.assertGreater(imbalance, 0)
        self.assertLess(imbalance, 1)
    
    def test_mid_price(self):
        """Test mid-price calculation."""
        expected_mid = (100.00 + 100.01) / 2.0
        self.assertAlmostEqual(self.snapshot.mid_price, expected_mid, places=6)


class TestLimitOrderBook(unittest.TestCase):
    def setUp(self):
        """Create a limit order book with multiple snapshots."""
        self.lob = LimitOrderBook(num_levels=5, tick_size=0.01)
        
        # Add several snapshots
        for i in range(10):
            bid_prices = np.array([100.00 - 0.01*j for j in range(5)])
            bid_volumes = np.array([1000 - 100*j for j in range(5)])
            ask_prices = np.array([100.01 + 0.01*j for j in range(5)])
            ask_volumes = np.array([800 - 80*j for j in range(5)])
            
            snapshot = OrderBookSnapshot(
                timestamp=float(i),
                bid_prices=bid_prices,
                bid_volumes=bid_volumes,
                ask_prices=ask_prices,
                ask_volumes=ask_volumes
            )
            self.lob.add_snapshot(snapshot)
    
    def test_snapshot_count(self):
        """Test that snapshots are added correctly."""
        self.assertEqual(len(self.lob.snapshots), 10)
    
    def test_microstructure_metrics(self):
        """Test computation of microstructure metrics."""
        metrics = self.lob.compute_microstructure_metrics(window=5)
        self.assertEqual(len(metrics), 5)  # 10 snapshots, window=5, so 10-5=5 metrics
        self.assertIn('spread_mean', metrics.columns)
        self.assertIn('imbalance_mean', metrics.columns)
        self.assertIn('depth_ratio', metrics.columns)
    
    def test_spread_components(self):
        """Test spread component estimation."""
        components = self.lob.estimate_spread_components()
        self.assertIn('total_spread', components)
        self.assertIn('adverse_selection', components)
        self.assertIn('inventory_cost', components)
        self.assertGreater(components['total_spread'], 0)


class TestMarketImpactModels(unittest.TestCase):
    def setUp(self):
        """Create impact models."""
        self.linear_model = LinearImpactModel(alpha=0.001, beta_temp=0.0005, beta_perm=0.0002)
        self.power_model = PowerLawImpactModel(lambda_temp=0.5, lambda_perm=0.3)
    
    def test_linear_impact(self):
        """Test linear impact model."""
        volume = 10000
        market_volume = 100000
        temp_impact, perm_impact = self.linear_model.compute_impact(volume, market_volume)
        
        # Linear model should give positive impacts
        self.assertGreater(temp_impact, 0)
        self.assertGreater(perm_impact, 0)
        # Temporary impact should exceed permanent
        self.assertGreater(temp_impact, perm_impact)
    
    def test_power_law_impact(self):
        """Test power-law impact model."""
        volume = 10000
        market_volume = 100000
        temp_impact, perm_impact = self.power_model.compute_impact(volume, market_volume)
        
        self.assertGreater(temp_impact, 0)
        self.assertGreater(perm_impact, 0)
        self.assertGreater(temp_impact, perm_impact)
    
    def test_volume_scaling(self):
        """Test that impact increases with volume."""
        small_volume = 1000
        large_volume = 100000
        market_volume = 500000
        
        temp_small, _ = self.linear_model.compute_impact(small_volume, market_volume)
        temp_large, _ = self.linear_model.compute_impact(large_volume, market_volume)
        
        self.assertGreater(temp_large, temp_small)


class TestOptimalExecution(unittest.TestCase):
    def setUp(self):
        """Create optimal execution problem."""
        self.impact_model = LinearImpactModel()
        self.executor = OptimalExecutionAC(
            initial_quantity=100000,
            time_horizon=1.0,
            num_periods=20,
            impact_model=self.impact_model,
            daily_volume=1e6
        )
    
    def test_optimal_schedule(self):
        """Test optimal execution schedule."""
        schedule = self.executor.optimal_execution(lambda_risk=1e-6)
        
        # Schedule should sum to total quantity
        self.assertAlmostEqual(np.sum(schedule), 100000, places=2)
        
        # All execution quantities should be non-negative
        self.assertTrue(np.all(schedule >= 0))
        
        # Number of periods should match
        self.assertEqual(len(schedule), 20)
    
    def test_schedule_feasibility(self):
        """Test that schedule is feasible."""
        schedule = self.executor.optimal_execution()
        
        # Cumulative sum should be non-decreasing
        cumsum = np.cumsum(schedule)
        self.assertTrue(np.all(np.diff(cumsum) >= 0))
        
        # Final cumsum should equal total quantity
        self.assertAlmostEqual(cumsum[-1], 100000, places=1)


class TestHighFrequencyMetrics(unittest.TestCase):
    def setUp(self):
        """Create sample trade data."""
        np.random.seed(42)
        n_trades = 500
        self.trade_data = pd.DataFrame({
            'timestamp': np.arange(n_trades) * 0.1,  # trades every 0.1 seconds
            'price': 100 + np.cumsum(np.random.normal(0, 0.05, n_trades)),
            'quantity': np.random.uniform(100, 1000, n_trades),
        })
        self.metrics = HighFrequencyMetrics(self.trade_data)
    
    def test_realized_volatility(self):
        """Test realized volatility computation."""
        rv = self.metrics.compute_realized_volatility(window=20)
        self.assertEqual(len(rv), len(self.trade_data) - 1)
        self.assertTrue(np.all(rv.dropna() > 0))
    
    def test_order_clustering(self):
        """Test order clustering analysis."""
        clustering = self.metrics.compute_order_clustering()
        self.assertIn('fano_factor', clustering)
        self.assertIn('clustering_detected', clustering)
        self.assertIn('hurst_exponent', clustering)
    
    def test_order_duration(self):
        """Test order duration statistics."""
        duration = self.metrics.compute_order_duration()
        self.assertIn('mean_duration', duration)
        self.assertIn('std_duration', duration)
        self.assertGreater(duration['mean_duration'], 0)
    
    def test_amihud_illiquidity(self):
        """Test Amihud illiquidity calculation."""
        illiq = self.metrics.compute_amihud_illiquidity()
        self.assertGreater(illiq, 0)
    
    def test_roll_spread(self):
        """Test Roll's spread estimate."""
        roll_spread = self.metrics.compute_roll_spread()
        self.assertGreaterEqual(roll_spread, 0)
    
    def test_data_validation(self):
        """Test data validation."""
        bad_data = pd.DataFrame({'price': [100, 101, 102]})  # missing required columns
        with self.assertRaises(ValueError):
            HighFrequencyMetrics(bad_data)


class TestMicrostructureSummary(unittest.TestCase):
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.trade_data = pd.DataFrame({
            'timestamp': np.arange(100) * 0.1,
            'price': 100 + np.cumsum(np.random.normal(0, 0.05, 100)),
            'quantity': np.random.uniform(100, 1000, 100),
        })
    
    def test_summary_generation(self):
        """Test microstructure summary generation."""
        summary = microstructure_summary(self.trade_data)
        self.assertEqual(len(summary), 1)
        self.assertIn('realized_volatility', summary.columns)
        self.assertIn('amihud_illiquidity', summary.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)

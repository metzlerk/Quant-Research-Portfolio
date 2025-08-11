#!/usr/bin/env python3
"""
Risk Management Module Example

This script demonstrates the basic usage of the risk management module
for quantitative portfolio analysis.

Author: Kevin J. Metzler
Date: August 7, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from risk_management import TailRiskModeler, BlackLittermanOptimizer, DynamicHedger, FactorModelBuilder
from risk_utils import calculate_var_es, optimize_risk_parity, perform_stress_test


def generate_sample_data():
    """Generate sample financial data for demonstration."""
    print("Generating sample financial data...")
    
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Create factor-based returns for realistic correlations
    n_assets = 5
    asset_names = ['TECH', 'FINANCE', 'HEALTHCARE', 'ENERGY', 'UTILITIES']
    
    # Market factor
    market_returns = np.random.normal(0.0008, 0.015, 1000)
    
    # Asset-specific returns with different factor loadings
    returns_data = {}
    factor_loadings = [1.2, 0.8, 0.6, 1.5, 0.4]  # Beta values
    
    for i, asset in enumerate(asset_names):
        specific_return = np.random.normal(0, 0.01, 1000)
        asset_return = factor_loadings[i] * market_returns + specific_return
        
        # Add some tail events
        tail_events = np.random.choice(range(1000), size=20, replace=False)
        asset_return[tail_events] += np.random.normal(-0.05, 0.02, 20)
        
        returns_data[asset] = asset_return
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    print(f"Generated returns for {len(asset_names)} assets over {len(dates)} days")
    print("\nSample statistics:")
    print(returns_df.describe())
    
    return returns_df


def demonstrate_tail_risk_modeling(returns_df):
    """Demonstrate Extreme Value Theory tail risk modeling."""
    print("\n" + "="*60)
    print("TAIL RISK MODELING WITH EXTREME VALUE THEORY")
    print("="*60)
    
    # Focus on the most volatile asset for demonstration
    asset_name = 'ENERGY'
    
    print(f"\nAnalyzing tail risk for {asset_name}...")
    
    # Initialize tail risk modeler
    risk_modeler = TailRiskModeler(returns_df[[asset_name]])
    
    # Fit Peaks Over Threshold model
    print("\n1. Fitting Peaks Over Threshold (POT) model...")
    pot_params = risk_modeler.fit_evt_model(asset_name, method='pot', threshold_percentile=0.1)
    print(f"POT Parameters: {pot_params}")
    
    # Calculate VaR and ES
    print("\n2. Calculating Value at Risk (VaR) and Expected Shortfall (ES)...")
    var_95, es_95 = risk_modeler.calculate_var_es(confidence_level=0.95, method='pot')
    var_99, es_99 = risk_modeler.calculate_var_es(confidence_level=0.99, method='pot')
    
    print(f"95% VaR: {var_95:.4f}, 95% ES: {es_95:.4f}")
    print(f"99% VaR: {var_99:.4f}, 99% ES: {es_99:.4f}")
    
    # Generate tail scenarios
    print("\n3. Generating tail risk scenarios...")
    scenarios = risk_modeler.generate_tail_scenarios(100, method='pot')
    print(f"Generated {len(scenarios)} tail scenarios")
    print(f"Mean tail scenario: {np.mean(scenarios):.4f}")
    print(f"Worst tail scenario: {np.min(scenarios):.4f}")


def demonstrate_black_litterman_optimization(returns_df):
    """Demonstrate Black-Litterman portfolio optimization."""
    print("\n" + "="*60)
    print("BLACK-LITTERMAN PORTFOLIO OPTIMIZATION")
    print("="*60)
    
    # Assume equal market cap weights initially
    market_weights = pd.Series(1/len(returns_df.columns), index=returns_df.columns)
    
    print(f"\nOptimizing portfolio for {len(returns_df.columns)} assets...")
    print(f"Market cap weights: {market_weights.to_dict()}")
    
    # Initialize Black-Litterman optimizer
    bl_optimizer = BlackLittermanOptimizer(returns_df, market_weights)
    
    # Estimate market-implied parameters
    print("\n1. Estimating market-implied expected returns...")
    mu_market, cov_market = bl_optimizer.estimate_market_parameters()
    print("Market-implied expected returns:")
    print(mu_market)
    
    # Optimize without views (pure market-implied)
    print("\n2. Optimizing without investor views...")
    weights_market = bl_optimizer.optimize(risk_aversion=3.0)
    analytics_market = bl_optimizer.analyze_portfolio(weights_market)
    
    print("Market-implied optimal weights:")
    print(weights_market)
    print(f"Expected return: {analytics_market['expected_return']:.4f}")
    print(f"Volatility: {analytics_market['volatility']:.4f}")
    print(f"Sharpe ratio: {analytics_market['sharpe_ratio']:.4f}")
    
    # Add some investor views
    print("\n3. Adding investor views and re-optimizing...")
    
    # View 1: TECH will outperform UTILITIES by 2% annually
    # View 2: FINANCE will have absolute return of 8% annually
    P = np.array([[1, 0, 0, 0, -1],  # TECH - UTILITIES
                  [0, 1, 0, 0, 0]])   # FINANCE
    Q = np.array([0.02/252, 0.08/252])  # Convert to daily returns
    Omega = np.diag([0.001, 0.002])  # View uncertainty
    
    weights_views = bl_optimizer.optimize(risk_aversion=3.0, views_P=P, views_Q=Q, views_Omega=Omega)
    analytics_views = bl_optimizer.analyze_portfolio(weights_views)
    
    print("Optimal weights with views:")
    print(weights_views)
    print(f"Expected return: {analytics_views['expected_return']:.4f}")
    print(f"Volatility: {analytics_views['volatility']:.4f}")
    print(f"Sharpe ratio: {analytics_views['sharpe_ratio']:.4f}")


def demonstrate_dynamic_hedging(returns_df):
    """Demonstrate dynamic hedging strategies."""
    print("\n" + "="*60)
    print("DYNAMIC HEDGING STRATEGIES")
    print("="*60)
    
    # Create a concentrated portfolio (focus on TECH and FINANCE)
    portfolio_weights = pd.Series([0.6, 0.4, 0.0, 0.0, 0.0], index=returns_df.columns)
    portfolio_returns = (returns_df * portfolio_weights).sum(axis=1)
    
    # Add portfolio returns to dataframe
    hedging_data = returns_df.copy()
    hedging_data['Portfolio'] = portfolio_returns
    
    print(f"\nCreated concentrated portfolio with weights: {portfolio_weights.to_dict()}")
    
    # Initialize dynamic hedger
    hedger = DynamicHedger(hedging_data[['Portfolio']])
    
    print("\n1. Calculating optimal hedge ratios...")
    # Use other assets as hedging instruments
    hedging_instruments = ['HEALTHCARE', 'ENERGY', 'UTILITIES']
    
    hedge_ratios = hedger.calculate_optimal_hedge(
        hedging_instruments=hedging_instruments,
        portfolio_col='Portfolio',
        method='minimum_variance',
        lookback_period=120,
        rebalance_frequency=30
    )
    
    print("Average hedge ratios:")
    print(hedge_ratios.mean())
    
    print("\n2. Applying hedge and evaluating effectiveness...")
    hedged_performance = hedger.apply_hedge(hedge_ratios, 'Portfolio')
    effectiveness = hedger.evaluate_hedge_effectiveness(hedged_performance)
    
    print(f"Hedge effectiveness: {effectiveness['hedge_effectiveness']:.4f}")
    print(f"Volatility reduction: {effectiveness['volatility_reduction']:.4f}")
    print(f"Original portfolio volatility: {effectiveness['original_volatility']:.4f}")
    print(f"Hedged portfolio volatility: {effectiveness['hedged_volatility']:.4f}")


def demonstrate_factor_modeling(returns_df):
    """Demonstrate factor model construction."""
    print("\n" + "="*60)
    print("FACTOR MODEL CONSTRUCTION")
    print("="*60)
    
    print(f"\nBuilding factor models for {len(returns_df.columns)} assets...")
    
    # Initialize factor model builder
    factor_builder = FactorModelBuilder(returns_df)
    
    print("\n1. Building statistical factor model (PCA)...")
    statistical_model = factor_builder.build_statistical_factor_model(n_factors=3)
    
    print("Factor loadings:")
    print(statistical_model['factor_loadings'])
    print(f"\nExplained variance by factors: {statistical_model['explained_variance']}")
    print(f"Total explained variance: {sum(statistical_model['explained_variance']):.4f}")
    
    print("\n2. Creating fundamental factor exposures...")
    # Create dummy fundamental exposures
    exposures = pd.DataFrame({
        'Market_Beta': [1.2, 0.8, 0.6, 1.5, 0.4],
        'Size_Factor': [0.5, -0.3, 0.2, 0.8, -0.7],
        'Value_Factor': [-0.2, 0.6, 0.1, -0.4, 0.3]
    }, index=returns_df.columns)
    
    print("Fundamental factor exposures:")
    print(exposures)
    
    fundamental_model = factor_builder.build_fundamental_factor_model(exposures)
    
    print("\nFundamental model factor returns (first 5 days):")
    print(fundamental_model['factor_returns'].head())
    
    print("\nSpecific risks:")
    print(fundamental_model['specific_risks'])


def demonstrate_risk_utilities(returns_df):
    """Demonstrate risk utility functions."""
    print("\n" + "="*60)
    print("RISK UTILITY FUNCTIONS")
    print("="*60)
    
    print("\n1. Portfolio-level VaR and ES calculation...")
    portfolio_returns = returns_df.mean(axis=1)  # Equal-weight portfolio
    var_es_results = calculate_var_es(portfolio_returns, confidence_levels=[0.95, 0.99])
    
    for cl, metrics in var_es_results.items():
        print(f"{cl}: VaR = {metrics['VaR']:.4f}, ES = {metrics['ES']:.4f}")
    
    print("\n2. Risk parity optimization...")
    cov_matrix = returns_df.cov()
    risk_parity_weights = optimize_risk_parity(cov_matrix, method='equal_risk_contribution')
    
    print("Risk parity weights:")
    print(risk_parity_weights)
    
    print("\n3. Stress testing...")
    # Define stress scenarios
    stress_scenarios = {
        'Market_Crash': pd.Series([-0.1, -0.08, -0.06, -0.12, -0.04], index=returns_df.columns),
        'Tech_Bubble': pd.Series([-0.2, -0.02, -0.03, -0.05, -0.01], index=returns_df.columns),
        'Energy_Crisis': pd.Series([-0.05, -0.03, -0.02, -0.15, 0.02], index=returns_df.columns)
    }
    
    equal_weights = pd.Series(1/len(returns_df.columns), index=returns_df.columns)
    stress_results = perform_stress_test(equal_weights, stress_scenarios)
    
    print("Stress test results:")
    for scenario, loss in stress_results.items():
        print(f"{scenario}: {loss:.4f}")


def main():
    """Main function to run all demonstrations."""
    print("Risk Management Module Demonstration")
    print("=" * 60)
    
    # Generate sample data
    returns_df = generate_sample_data()
    
    # Run demonstrations
    demonstrate_tail_risk_modeling(returns_df)
    demonstrate_black_litterman_optimization(returns_df)
    demonstrate_dynamic_hedging(returns_df)
    demonstrate_factor_modeling(returns_df)
    demonstrate_risk_utilities(returns_df)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nThis example showcased the key features of the risk management module:")
    print("• Extreme Value Theory for tail risk modeling")
    print("• Black-Litterman optimization with investor views")
    print("• Dynamic hedging with minimum variance approach")
    print("• Statistical and fundamental factor model construction")
    print("• Risk utility functions for portfolio analysis")
    print("\nFor more detailed examples, see the Jupyter notebook: risk_analysis.ipynb")


if __name__ == "__main__":
    main()

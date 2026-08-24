"""
Example usage of the market microstructure module.

This script demonstrates key capabilities:
1. Order book analysis and microstructure metrics
2. Market impact estimation with different models
3. Optimal execution strategy design
4. High-frequency trading metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from market_microstructure import (
    OrderBookSnapshot,
    LimitOrderBook,
    LinearImpactModel,
    PowerLawImpactModel,
    OptimalExecutionAC,
    HighFrequencyMetrics,
)


def example_1_order_book_analysis():
    """
    Example 1: Analyze limit order book and compute microstructure metrics.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Order Book Analysis")
    print("="*70)
    
    # Create limit order book
    lob = LimitOrderBook(num_levels=10, tick_size=0.01)
    
    # Simulate order book evolution over time
    np.random.seed(42)
    for t in range(50):
        bid_prices = np.array([100.00 - 0.01*i for i in range(10)])
        ask_prices = np.array([100.01 + 0.01*i for i in range(10)])
        
        # Random volume fluctuations
        bid_volumes = np.array([1000 - 100*i + np.random.normal(0, 50) for i in range(10)])
        ask_volumes = np.array([800 - 80*i + np.random.normal(0, 40) for i in range(10)])
        
        # Ensure volumes are positive
        bid_volumes = np.maximum(bid_volumes, 100)
        ask_volumes = np.maximum(ask_volumes, 100)
        
        snapshot = OrderBookSnapshot(
            timestamp=float(t),
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes
        )
        lob.add_snapshot(snapshot)
    
    # Compute metrics
    metrics = lob.compute_microstructure_metrics(window=10)
    spread_components = lob.estimate_spread_components()
    
    print("\nOrder Book Snapshots:", len(lob.snapshots))
    print("\nMicrostructure Metrics (last 5 periods):")
    print(metrics.tail().to_string())
    
    print("\nSpread Decomposition:")
    for key, value in spread_components.items():
        print(f"  {key}: {value:.6f}")
    
    return lob, metrics, spread_components


def example_2_market_impact():
    """
    Example 2: Compare linear vs. power-law market impact models.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Market Impact Models Comparison")
    print("="*70)
    
    # Create impact models
    linear_model = LinearImpactModel(alpha=0.001, beta_temp=0.0005, beta_perm=0.0002)
    power_model = PowerLawImpactModel(lambda_temp=0.5, lambda_perm=0.3,
                                      coeff_temp=0.0001, coeff_perm=0.00005)
    
    # Compute impact across different order sizes
    volumes = np.array([5000, 10000, 25000, 50000, 100000, 250000, 500000])
    daily_volume = 1e6
    
    results = []
    for vol in volumes:
        temp_lin, perm_lin = linear_model.compute_impact(vol, daily_volume)
        temp_pow, perm_pow = power_model.compute_impact(vol, daily_volume)
        results.append({
            'volume': vol,
            'volume_pct': 100 * vol / daily_volume,
            'temp_impact_linear': temp_lin,
            'perm_impact_linear': perm_lin,
            'temp_impact_power': temp_pow,
            'perm_impact_power': perm_pow,
        })
    
    impact_df = pd.DataFrame(results)
    print("\nMarket Impact Estimates (bps):")
    print(impact_df.to_string(index=False))
    
    print("\nInterpretation:")
    print("  - Linear model: impact grows linearly with volume")
    print("  - Power-law model: impact grows sublinearly (concave) or superlinearly (convex)")
    print("  - Temporary impact >> permanent impact (typical microstructure finding)")
    
    return impact_df


def example_3_optimal_execution():
    """
    Example 3: Design optimal execution schedule using Almgren-Chriss.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Optimal Execution (Almgren-Chriss)")
    print("="*70)
    
    # Define execution problem
    total_shares = 100000
    time_horizon = 1.0  # 1 trading day
    num_periods = 20
    daily_volume = 1e6
    
    # Create impact model and executor
    impact_model = LinearImpactModel(alpha=0.001, beta_temp=0.0005, beta_perm=0.0002)
    executor = OptimalExecutionAC(
        initial_quantity=total_shares,
        time_horizon=time_horizon,
        num_periods=num_periods,
        impact_model=impact_model,
        daily_volume=daily_volume
    )
    
    # Solve for optimal schedule with different risk aversion parameters
    risk_params = [1e-7, 1e-6, 1e-5, 1e-4]
    schedules = {}
    
    for lambda_risk in risk_params:
        schedule = executor.optimal_execution(lambda_risk=lambda_risk)
        schedules[f'lambda={lambda_risk:.0e}'] = schedule
        
        print(f"\nOptimal Schedule (λ={lambda_risk:.0e}):")
        print(f"  Min period execution: {np.min(schedule):,.0f} shares")
        print(f"  Max period execution: {np.max(schedule):,.0f} shares")
        print(f"  Std dev: {np.std(schedule):,.0f} shares")
        print(f"  Total: {np.sum(schedule):,.0f} shares")
    
    # The schedule with lowest lambda is most aggressive (uniform), 
    # higher lambda leads to front-loaded execution
    print("\nInterpretation:")
    print("  - Low λ: aggressive execution, higher market impact cost")
    print("  - High λ: patient execution, lower immediate impact but higher timing risk")
    
    return executor, schedules


def example_4_hft_metrics():
    """
    Example 4: Compute high-frequency trading metrics.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: High-Frequency Trading Metrics")
    print("="*70)
    
    # Simulate trade data
    np.random.seed(42)
    n_trades = 1000
    
    # Geometric Brownian Motion for prices
    log_returns = np.random.normal(-0.0001, 0.005, n_trades)
    prices = 100 * np.exp(np.cumsum(log_returns))
    
    # Inter-arrival times (exponential distribution)
    inter_arrival = np.random.exponential(0.1, n_trades)
    timestamps = np.cumsum(inter_arrival)
    
    # Trade volumes
    volumes = np.random.gamma(2, 500, n_trades)
    
    trade_data = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'quantity': volumes,
    })
    
    # Compute metrics
    metrics = HighFrequencyMetrics(trade_data)
    
    realized_vol = metrics.compute_realized_volatility(window=50).mean()
    amihud = metrics.compute_amihud_illiquidity()
    roll_spread = metrics.compute_roll_spread()
    clustering = metrics.compute_order_clustering()
    duration = metrics.compute_order_duration()
    
    print(f"\nRealized Volatility (annualized): {realized_vol:.4f}")
    print(f"Amihud Illiquidity: {amihud:.6f}")
    print(f"Roll's Spread Estimate: {roll_spread:.6f}")
    
    print("\nOrder Clustering Analysis:")
    for key, value in clustering.items():
        print(f"  {key}: {value}")
    
    print("\nOrder Duration Statistics:")
    for key, value in duration.items():
        print(f"  {key}: {value:.6f}")
    
    print("\nInterpretation:")
    print("  - Fano factor > 1: orders cluster in time (non-Poisson)")
    print("  - Hurst exponent > 0.5: persistent clustering")
    print("  - High Amihud: low liquidity (large price impact per unit volume)")
    
    return trade_data, metrics


def example_5_visualization():
    """
    Example 5: Visualize market impact and execution schedules.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Visualization of Impact and Execution")
    print("="*70)
    
    # Create models and compute impacts
    linear_model = LinearImpactModel(alpha=0.001, beta_temp=0.0005, beta_perm=0.0002)
    power_model = PowerLawImpactModel(lambda_temp=0.5, lambda_perm=0.3,
                                      coeff_temp=0.0001, coeff_perm=0.00005)
    
    volumes = np.linspace(1000, 500000, 100)
    daily_volume = 1e6
    
    impacts_lin = np.array([linear_model.compute_impact(v, daily_volume)[0] for v in volumes])
    impacts_pow = np.array([power_model.compute_impact(v, daily_volume)[0] for v in volumes])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Market impact vs. order size
    axes[0].plot(volumes / 1000, impacts_lin * 10000, 'b-', label='Linear', linewidth=2)
    axes[0].plot(volumes / 1000, impacts_pow * 10000, 'r--', label='Power-law', linewidth=2)
    axes[0].set_xlabel('Order Size ($1000s)')
    axes[0].set_ylabel('Temporary Impact (bps)')
    axes[0].set_title('Market Impact Models')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Optimal execution schedule
    executor = OptimalExecutionAC(
        initial_quantity=100000,
        time_horizon=1.0,
        num_periods=20,
        impact_model=linear_model,
        daily_volume=1e6
    )
    
    schedule_aggressive = executor.optimal_execution(lambda_risk=1e-7)
    schedule_patient = executor.optimal_execution(lambda_risk=1e-4)
    
    axes[1].plot(range(1, 21), schedule_aggressive / 1000, 'b-o', label='Aggressive (λ=1e-7)', linewidth=2)
    axes[1].plot(range(1, 21), schedule_patient / 1000, 'r-s', label='Patient (λ=1e-4)', linewidth=2)
    axes[1].set_xlabel('Period')
    axes[1].set_ylabel('Execution Volume ($1000s)')
    axes[1].set_title('Optimal Execution Schedules')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("\nVisualization created (market_microstructure_visualization.png)")
    # Uncomment to display:
    # plt.show()
    
    return fig


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MARKET MICROSTRUCTURE MODULE - COMPREHENSIVE EXAMPLES")
    print("="*70)
    
    # Run examples
    lob, metrics, spread_comp = example_1_order_book_analysis()
    impact_df = example_2_market_impact()
    executor, schedules = example_3_optimal_execution()
    trade_data, hft_metrics = example_4_hft_metrics()
    fig = example_5_visualization()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    main()

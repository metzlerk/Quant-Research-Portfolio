"""
Risk Management Utility Functions

This module provides various helper functions for risk management and portfolio analysis.

Author: Kevin J.D. Metzler
Date: August 7, 2025
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings


def calculate_var_es(returns: pd.Series, 
                    confidence_levels: List[float] = [0.95, 0.99],
                    method: str = 'historical') -> Dict[str, Dict[str, float]]:
    """
    Calculate Value-at-Risk (VaR) and Expected Shortfall (ES) using various methods.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    confidence_levels : List[float]
        Confidence levels for VaR/ES calculation
    method : str
        Method to use ('historical', 'parametric', 'cornish_fisher')
        
    Returns:
    --------
    Dict[str, Dict[str, float]]
        VaR and ES values for each confidence level
    """
    results = {}
    
    for cl in confidence_levels:
        alpha = 1 - cl
        
        if method == 'historical':
            var = np.percentile(returns.dropna(), alpha * 100)
            es = returns[returns <= var].mean()
            
        elif method == 'parametric':
            # Assume normal distribution
            mu = returns.mean()
            sigma = returns.std()
            var = stats.norm.ppf(alpha, mu, sigma)
            # ES for normal distribution
            es = mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
            
        elif method == 'cornish_fisher':
            # Cornish-Fisher expansion for non-normal distributions
            mu = returns.mean()
            sigma = returns.std()
            skew = stats.skew(returns.dropna())
            kurt = stats.kurtosis(returns.dropna())
            
            # Standard normal quantile
            z = stats.norm.ppf(alpha)
            
            # Cornish-Fisher adjustment
            z_cf = z + (z**2 - 1) * skew / 6 + \
                   (z**3 - 3*z) * kurt / 24 - \
                   (2*z**3 - 5*z) * skew**2 / 36
            
            var = mu + sigma * z_cf
            
            # Approximate ES using modified Cornish-Fisher
            es_adjustment = (1 + skew * z + kurt * (z**2 - 1) / 4) / alpha
            es = mu + sigma * z * es_adjustment
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results[f'CL_{cl:.0%}'] = {
            'VaR': var,
            'ES': es
        }
    
    return results


def decompose_risk(portfolio_weights: pd.Series, 
                  covariance_matrix: pd.DataFrame,
                  factor_loadings: Optional[pd.DataFrame] = None) -> Dict[str, Union[pd.Series, float]]:
    """
    Decompose portfolio risk into systematic and idiosyncratic components.
    
    Parameters:
    -----------
    portfolio_weights : pd.Series
        Portfolio weights
    covariance_matrix : pd.DataFrame
        Asset covariance matrix
    factor_loadings : pd.DataFrame, optional
        Factor loadings for factor-based decomposition
        
    Returns:
    --------
    Dict[str, Union[pd.Series, float]]
        Risk decomposition results
    """
    # Total portfolio variance
    portfolio_variance = portfolio_weights.T @ covariance_matrix @ portfolio_weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Marginal contribution to risk (MCR)
    mcr = (covariance_matrix @ portfolio_weights) / portfolio_volatility
    
    # Component contribution to risk (CCR)
    ccr = portfolio_weights * mcr
    
    # Percentage contribution to risk (PCR)
    pcr = ccr / portfolio_variance
    
    results = {
        'total_volatility': portfolio_volatility,
        'marginal_contribution': mcr,
        'component_contribution': ccr,
        'percentage_contribution': pcr
    }
    
    # Factor-based decomposition if factor loadings provided
    if factor_loadings is not None:
        # Systematic risk (factor-based)
        factor_variance = factor_loadings.T @ covariance_matrix @ factor_loadings
        systematic_variance = portfolio_weights.T @ factor_loadings @ factor_variance @ factor_loadings.T @ portfolio_weights
        
        # Idiosyncratic risk
        idiosyncratic_variance = portfolio_variance - systematic_variance
        
        results.update({
            'systematic_volatility': np.sqrt(max(0, systematic_variance)),
            'idiosyncratic_volatility': np.sqrt(max(0, idiosyncratic_variance)),
            'systematic_percentage': systematic_variance / portfolio_variance,
            'idiosyncratic_percentage': idiosyncratic_variance / portfolio_variance
        })
    
    return results


def perform_stress_test(portfolio_weights: pd.Series,
                       asset_returns: pd.DataFrame,
                       stress_scenarios: Dict[str, Dict[str, float]],
                       monte_carlo_scenarios: int = 1000) -> Dict[str, Dict[str, float]]:
    """
    Perform stress testing on portfolio using historical and hypothetical scenarios.
    
    Parameters:
    -----------
    portfolio_weights : pd.Series
        Portfolio weights
    asset_returns : pd.DataFrame
        Historical asset returns
    stress_scenarios : Dict[str, Dict[str, float]]
        Predefined stress scenarios
    monte_carlo_scenarios : int
        Number of Monte Carlo scenarios to generate
        
    Returns:
    --------
    Dict[str, Dict[str, float]]
        Stress test results
    """
    results = {}
    
    # Historical stress test - worst periods
    portfolio_returns = (asset_returns * portfolio_weights).sum(axis=1)
    worst_day = portfolio_returns.min()
    worst_week = portfolio_returns.rolling(5).sum().min()
    worst_month = portfolio_returns.rolling(22).sum().min()
    
    results['historical_stress'] = {
        'worst_day': worst_day,
        'worst_week': worst_week,
        'worst_month': worst_month
    }
    
    # Predefined scenarios
    for scenario_name, shocks in stress_scenarios.items():
        scenario_return = 0
        for asset, shock in shocks.items():
            if asset in portfolio_weights.index:
                scenario_return += portfolio_weights[asset] * shock
        
        results[scenario_name] = {
            'portfolio_return': scenario_return
        }
    
    # Monte Carlo stress test
    if monte_carlo_scenarios > 0:
        # Estimate asset return distributions
        means = asset_returns.mean()
        cov_matrix = asset_returns.cov()
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        random_returns = np.random.multivariate_normal(
            means.values, 
            cov_matrix.values, 
            monte_carlo_scenarios
        )
        
        # Calculate portfolio returns for each scenario
        mc_portfolio_returns = random_returns @ portfolio_weights.values
        
        results['monte_carlo_stress'] = {
            'var_95': np.percentile(mc_portfolio_returns, 5),
            'var_99': np.percentile(mc_portfolio_returns, 1),
            'expected_shortfall_95': mc_portfolio_returns[mc_portfolio_returns <= np.percentile(mc_portfolio_returns, 5)].mean(),
            'expected_shortfall_99': mc_portfolio_returns[mc_portfolio_returns <= np.percentile(mc_portfolio_returns, 1)].mean(),
            'worst_case': mc_portfolio_returns.min(),
            'best_case': mc_portfolio_returns.max()
        }
    
    return results


def optimize_risk_parity(covariance_matrix: pd.DataFrame,
                        method: str = 'equal_risk_contribution',
                        constraints: Optional[Dict] = None) -> pd.Series:
    """
    Optimize portfolio for risk parity allocation.
    
    Parameters:
    -----------
    covariance_matrix : pd.DataFrame
        Asset covariance matrix
    method : str
        Risk parity method ('equal_risk_contribution', 'inverse_volatility')
    constraints : Dict, optional
        Additional constraints
        
    Returns:
    --------
    pd.Series
        Risk parity weights
    """
    n_assets = len(covariance_matrix)
    
    if method == 'inverse_volatility':
        # Simple inverse volatility weighting
        volatilities = np.sqrt(np.diag(covariance_matrix))
        weights = 1 / volatilities
        weights = weights / weights.sum()
        
        return pd.Series(weights, index=covariance_matrix.index)
    
    elif method == 'equal_risk_contribution':
        # Equal risk contribution optimization
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(weights.T @ covariance_matrix.values @ weights)
            marginal_contrib = (covariance_matrix.values @ weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Target: equal risk contribution (1/n for each asset)
            target_contrib = portfolio_vol / n_assets
            
            # Sum of squared deviations from target
            return np.sum((contrib - target_contrib)**2)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Budget constraint
            {'type': 'ineq', 'fun': lambda w: w}  # Long-only
        ]
        
        # Additional constraints
        if constraints:
            if 'max_weight' in constraints:
                constraints_list.append({
                    'type': 'ineq', 
                    'fun': lambda w: constraints['max_weight'] - w
                })
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn("Risk parity optimization did not converge")
            return pd.Series(x0, index=covariance_matrix.index)
        
        return pd.Series(result.x, index=covariance_matrix.index)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_drawdowns(cumulative_returns: pd.Series,
                       min_periods: int = 1) -> Dict[str, Union[pd.Series, float]]:
    """
    Calculate drawdown statistics for a return series.
    
    Parameters:
    -----------
    cumulative_returns : pd.Series
        Cumulative return series
    min_periods : int
        Minimum periods for rolling calculations
        
    Returns:
    --------
    Dict[str, Union[pd.Series, float]]
        Drawdown statistics
    """
    # Calculate running maximum
    running_max = cumulative_returns.expanding(min_periods=min_periods).max()
    
    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Maximum drawdown
    max_drawdown = drawdown.min()
    
    # Find drawdown periods
    is_drawdown = drawdown < 0
    drawdown_starts = (~is_drawdown) & is_drawdown.shift(-1, fill_value=False)
    drawdown_ends = is_drawdown & (~is_drawdown.shift(-1, fill_value=False))
    
    # Calculate duration of each drawdown period
    drawdown_periods = []
    start_indices = drawdown.index[drawdown_starts]
    end_indices = drawdown.index[drawdown_ends]
    
    for start, end in zip(start_indices, end_indices):
        period_drawdown = drawdown.loc[start:end]
        duration = len(period_drawdown)
        depth = period_drawdown.min()
        
        drawdown_periods.append({
            'start': start,
            'end': end,
            'duration': duration,
            'depth': depth
        })
    
    # Sort by depth (worst drawdowns first)
    drawdown_periods.sort(key=lambda x: x['depth'])
    
    # Average drawdown duration
    avg_duration = np.mean([dd['duration'] for dd in drawdown_periods]) if drawdown_periods else 0
    
    # Time to recovery (time from max drawdown to recovery)
    max_dd_date = drawdown.idxmin()
    recovery_dates = cumulative_returns[cumulative_returns.index > max_dd_date]
    max_dd_value = cumulative_returns.loc[max_dd_date]
    
    recovery_date = None
    for date in recovery_dates.index:
        if recovery_dates.loc[date] >= cumulative_returns.loc[max_dd_date:date].max():
            recovery_date = date
            break
    
    time_to_recovery = None
    if recovery_date:
        delta = recovery_date - max_dd_date
        time_to_recovery = delta.days if hasattr(delta, 'days') else int(delta)
    
    results = {
        'drawdown_series': drawdown,
        'max_drawdown': max_drawdown,
        'max_drawdown_date': drawdown.idxmin(),
        'avg_drawdown_duration': avg_duration,
        'time_to_recovery': time_to_recovery,
        'drawdown_periods': drawdown_periods[:5],  # Top 5 worst drawdowns
        'current_drawdown': drawdown.iloc[-1],
        'recovery_factor': abs(max_drawdown) / cumulative_returns.std() if cumulative_returns.std() > 0 else 0
    }
    
    return results


def calculate_rolling_risk_metrics(returns: pd.Series,
                                  window: int = 252,
                                  confidence_level: float = 0.95) -> pd.DataFrame:
    """
    Calculate rolling risk metrics over time.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    window : int
        Rolling window size
    confidence_level : float
        Confidence level for risk metrics
        
    Returns:
    --------
    pd.DataFrame
        Rolling risk metrics
    """
    rolling_metrics = pd.DataFrame(index=returns.index)
    
    # Rolling volatility
    rolling_metrics['volatility'] = returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling VaR
    rolling_metrics['var'] = returns.rolling(window).quantile(1 - confidence_level)
    
    # Rolling Expected Shortfall
    def rolling_es(x):
        threshold = x.quantile(1 - confidence_level)
        return x[x <= threshold].mean()
    
    rolling_metrics['expected_shortfall'] = returns.rolling(window).apply(rolling_es)
    
    # Rolling Sharpe ratio
    rolling_metrics['sharpe_ratio'] = (returns.rolling(window).mean() / 
                                     returns.rolling(window).std()) * np.sqrt(252)
    
    # Rolling skewness and kurtosis
    rolling_metrics['skewness'] = returns.rolling(window).skew()
    rolling_metrics['kurtosis'] = returns.rolling(window).kurt()
    
    # Rolling maximum drawdown
    def rolling_max_dd(x):
        cumulative = (1 + x).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    rolling_metrics['max_drawdown'] = returns.rolling(window).apply(rolling_max_dd)
    
    return rolling_metrics.dropna()


def create_risk_dashboard(portfolio_returns: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         risk_free_rate: float = 0.02) -> None:
    """
    Create comprehensive risk dashboard with visualizations.
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio return series
    benchmark_returns : pd.Series, optional
        Benchmark return series for comparison
    risk_free_rate : float
        Risk-free rate for Sharpe ratio calculation
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Risk Management Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, 
                   label='Portfolio', linewidth=2)
    
    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                       label='Benchmark', linewidth=2, alpha=0.7)
    
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Drawdown analysis
    drawdown_results = calculate_drawdowns(cumulative_returns)
    drawdown_series = drawdown_results['drawdown_series']
    
    axes[0, 1].fill_between(drawdown_series.index, drawdown_series.values, 0, 
                           alpha=0.3, color='red')
    axes[0, 1].plot(drawdown_series.index, drawdown_series.values, color='red')
    axes[0, 1].set_title(f'Drawdown (Max: {drawdown_results["max_drawdown"]:.2%})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Return distribution
    axes[0, 2].hist(portfolio_returns.dropna(), bins=50, alpha=0.7, density=True)
    axes[0, 2].axvline(portfolio_returns.mean(), color='red', linestyle='--', 
                      label=f'Mean: {portfolio_returns.mean():.3f}')
    axes[0, 2].axvline(np.percentile(portfolio_returns.dropna(), 5), color='orange', 
                      linestyle='--', label=f'5% VaR: {np.percentile(portfolio_returns.dropna(), 5):.3f}')
    axes[0, 2].set_title('Return Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Rolling volatility
    rolling_vol = portfolio_returns.rolling(252).std() * np.sqrt(252)
    axes[1, 0].plot(rolling_vol.index, rolling_vol.values)
    axes[1, 0].axhline(rolling_vol.mean(), color='red', linestyle='--', 
                      label=f'Average: {rolling_vol.mean():.2%}')
    axes[1, 0].set_title('Rolling 1-Year Volatility')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Rolling Sharpe ratio
    excess_returns = portfolio_returns - risk_free_rate/252
    rolling_sharpe = (excess_returns.rolling(252).mean() / 
                     portfolio_returns.rolling(252).std()) * np.sqrt(252)
    
    axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
    axes[1, 1].axhline(rolling_sharpe.mean(), color='red', linestyle='--', 
                      label=f'Average: {rolling_sharpe.mean():.2f}')
    axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_title('Rolling Sharpe Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Risk metrics summary
    risk_metrics = calculate_var_es(portfolio_returns)
    metrics_text = f"""Risk Metrics Summary:
    
Volatility: {portfolio_returns.std() * np.sqrt(252):.2%}
Sharpe Ratio: {(portfolio_returns.mean() - risk_free_rate/252) / portfolio_returns.std() * np.sqrt(252):.2f}
Skewness: {stats.skew(portfolio_returns.dropna()):.2f}
Kurtosis: {stats.kurtosis(portfolio_returns.dropna()):.2f}

VaR (95%): {risk_metrics['CL_95%']['VaR']:.3f}
ES (95%): {risk_metrics['CL_95%']['ES']:.3f}
VaR (99%): {risk_metrics['CL_99%']['VaR']:.3f}
ES (99%): {risk_metrics['CL_99%']['ES']:.3f}

Max Drawdown: {drawdown_results['max_drawdown']:.2%}
Recovery Time: {drawdown_results['time_to_recovery']} days"""
    
    axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_title('Risk Metrics Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_tail_dependencies(returns_matrix: pd.DataFrame,
                            method: str = 'kendall_tau') -> pd.DataFrame:
    """
    Analyze tail dependencies between assets using copula-based measures.
    
    Parameters:
    -----------
    returns_matrix : pd.DataFrame
        Matrix of asset returns
    method : str
        Method for measuring dependence ('kendall_tau', 'spearman')
        
    Returns:
    --------
    pd.DataFrame
        Tail dependence matrix
    """
    n_assets = len(returns_matrix.columns)
    tail_dependence = pd.DataFrame(
        index=returns_matrix.columns,
        columns=returns_matrix.columns,
        dtype=float
    )
    
    for i, asset1 in enumerate(returns_matrix.columns):
        for j, asset2 in enumerate(returns_matrix.columns):
            if i == j:
                tail_dependence.loc[asset1, asset2] = 1.0
            elif i < j:
                returns1 = returns_matrix[asset1].dropna()
                returns2 = returns_matrix[asset2].dropna()
                
                # Align series
                common_index = returns1.index.intersection(returns2.index)
                aligned_returns1 = returns1.loc[common_index]
                aligned_returns2 = returns2.loc[common_index]
                
                if method == 'kendall_tau':
                    tau, _ = stats.kendalltau(aligned_returns1, aligned_returns2)
                    tail_dependence.loc[asset1, asset2] = tau
                    tail_dependence.loc[asset2, asset1] = tau
                elif method == 'spearman':
                    rho, _ = stats.spearmanr(aligned_returns1, aligned_returns2)
                    tail_dependence.loc[asset1, asset2] = rho
                    tail_dependence.loc[asset2, asset1] = rho
    
    return tail_dependence


def calculate_regime_dependent_risk(returns: pd.Series,
                                  regime_indicator: pd.Series,
                                  regimes: List[str] = ['low_vol', 'high_vol']) -> Dict[str, Dict[str, float]]:
    """
    Calculate risk metrics conditional on market regimes.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    regime_indicator : pd.Series
        Regime indicator (categorical)
    regimes : List[str]
        List of regime names
        
    Returns:
    --------
    Dict[str, Dict[str, float]]
        Risk metrics by regime
    """
    regime_risk = {}
    
    for regime in regimes:
        regime_returns = returns[regime_indicator == regime]
        
        if len(regime_returns) > 0:
            regime_risk[regime] = {
                'volatility': regime_returns.std() * np.sqrt(252),
                'skewness': stats.skew(regime_returns.dropna()),
                'kurtosis': stats.kurtosis(regime_returns.dropna()),
                'var_95': np.percentile(regime_returns.dropna(), 5),
                'var_99': np.percentile(regime_returns.dropna(), 1),
                'max_loss': regime_returns.min(),
                'mean_return': regime_returns.mean() * 252,
                'num_observations': len(regime_returns)
            }
        else:
            regime_risk[regime] = {}
    
    return regime_risk

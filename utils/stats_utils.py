"""
Statistical and econometric utilities for quantitative research.
Includes time series analysis, hypothesis testing, and model diagnostics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, List, Optional, Union
import warnings
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    """
    Comprehensive time series analysis toolkit for financial data.
    """
    
    @staticmethod
    def test_stationarity(
        series: pd.Series, 
        method: str = 'adf',
        significance_level: float = 0.05
    ) -> Dict[str, any]:
        """
        Test for stationarity using multiple methods.
        
        Parameters:
        -----------
        series : pd.Series
            Time series data to test
        method : str
            Testing method: 'adf', 'kpss', or 'both'
        significance_level : float
            Significance level for tests
            
        Returns:
        --------
        Dict
            Test results and interpretation
        """
        series_clean = series.dropna()
        results = {}
        
        if method in ['adf', 'both']:
            # Augmented Dickey-Fuller Test
            adf_result = adfuller(series_clean, autolag='AIC')
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < significance_level,
                'interpretation': 'Stationary' if adf_result[1] < significance_level else 'Non-stationary'
            }
        
        if method in ['kpss', 'both']:
            # KPSS Test
            kpss_result = kpss(series_clean, regression='c')
            results['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > significance_level,
                'interpretation': 'Stationary' if kpss_result[1] > significance_level else 'Non-stationary'
            }
        
        return results
    
    @staticmethod
    def test_cointegration(
        series1: pd.Series, 
        series2: pd.Series, 
        method: str = 'engle_granger'
    ) -> Dict[str, any]:
        """
        Test for cointegration between two time series.
        
        Parameters:
        -----------
        series1, series2 : pd.Series
            Time series to test for cointegration
        method : str
            Method to use: 'engle_granger' or 'johansen'
            
        Returns:
        --------
        Dict
            Cointegration test results
        """
        # Align series
        aligned_data = pd.concat([series1, series2], axis=1).dropna()
        s1, s2 = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
        
        if method == 'engle_granger':
            # Engle-Granger two-step method
            coint_result = coint(s1, s2)
            return {
                'method': 'Engle-Granger',
                'statistic': coint_result[0],
                'p_value': coint_result[1],
                'critical_values': coint_result[2],
                'is_cointegrated': coint_result[1] < 0.05,
                'interpretation': 'Cointegrated' if coint_result[1] < 0.05 else 'Not cointegrated'
            }
        
        elif method == 'johansen':
            # Johansen test
            johansen_result = coint_johansen(aligned_data.values, det_order=0, k_ar_diff=1)
            return {
                'method': 'Johansen',
                'trace_statistic': johansen_result.lr1,
                'max_eigenvalue_statistic': johansen_result.lr2,
                'critical_values_trace': johansen_result.cvt,
                'critical_values_max_eigen': johansen_result.cvm,
                'eigenvalues': johansen_result.eig
            }
    
    @staticmethod
    def test_autocorrelation(
        series: pd.Series, 
        lags: int = 10,
        test_type: str = 'ljungbox'
    ) -> Dict[str, any]:
        """
        Test for autocorrelation in time series.
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        lags : int
            Number of lags to test
        test_type : str
            Type of test: 'ljungbox'
            
        Returns:
        --------
        Dict
            Autocorrelation test results
        """
        series_clean = series.dropna()
        
        if test_type == 'ljungbox':
            ljung_result = acorr_ljungbox(series_clean, lags=lags, return_df=True)
            return {
                'method': 'Ljung-Box',
                'statistics': ljung_result['lb_stat'].values,
                'p_values': ljung_result['lb_pvalue'].values,
                'has_autocorrelation': (ljung_result['lb_pvalue'] < 0.05).any(),
                'significant_lags': ljung_result[ljung_result['lb_pvalue'] < 0.05].index.tolist()
            }


class RiskMetrics:
    """
    Comprehensive risk measurement toolkit.
    """
    
    @staticmethod
    def calculate_var(
        returns: pd.Series, 
        confidence_level: float = 0.05,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        confidence_level : float
            Confidence level (e.g., 0.05 for 95% VaR)
        method : str
            Method: 'historical', 'parametric', or 'cornish_fisher'
            
        Returns:
        --------
        float
            VaR estimate
        """
        returns_clean = returns.dropna()
        
        if method == 'historical':
            return np.percentile(returns_clean, confidence_level * 100)
        
        elif method == 'parametric':
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            return stats.norm.ppf(confidence_level, mu, sigma)
        
        elif method == 'cornish_fisher':
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            skewness = stats.skew(returns_clean)
            kurtosis = stats.kurtosis(returns_clean, fisher=False)
            
            # Cornish-Fisher adjustment
            z = stats.norm.ppf(confidence_level)
            cf_adjustment = (z + (z**2 - 1) * skewness / 6 + 
                           (z**3 - 3*z) * (kurtosis - 3) / 24 - 
                           (2*z**3 - 5*z) * skewness**2 / 36)
            
            return mu + sigma * cf_adjustment
    
    @staticmethod
    def calculate_cvar(
        returns: pd.Series, 
        confidence_level: float = 0.05,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        confidence_level : float
            Confidence level
        method : str
            Method: 'historical' or 'parametric'
            
        Returns:
        --------
        float
            CVaR estimate
        """
        returns_clean = returns.dropna()
        
        if method == 'historical':
            var = RiskMetrics.calculate_var(returns_clean, confidence_level, 'historical')
            return returns_clean[returns_clean <= var].mean()
        
        elif method == 'parametric':
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            z = stats.norm.ppf(confidence_level)
            return mu - sigma * stats.norm.pdf(z) / confidence_level
    
    @staticmethod
    def calculate_maximum_drawdown(price_series: pd.Series) -> Dict[str, any]:
        """
        Calculate maximum drawdown and related metrics.
        
        Parameters:
        -----------
        price_series : pd.Series
            Price series
            
        Returns:
        --------
        Dict
            Drawdown metrics
        """
        # Calculate running maximum
        running_max = price_series.expanding().max()
        
        # Calculate drawdown
        drawdown = (price_series - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Calculate drawdown duration
        in_drawdown = drawdown < -0.001  # More than 0.1% drawdown
        drawdown_periods = []
        start_dd = None
        
        for date, is_dd in in_drawdown.items():
            if is_dd and start_dd is None:
                start_dd = date
            elif not is_dd and start_dd is not None:
                drawdown_periods.append((start_dd, date))
                start_dd = None
        
        # Calculate average and maximum drawdown duration
        if drawdown_periods:
            durations = [(end - start).days for start, end in drawdown_periods]
            avg_duration = np.mean(durations)
            max_duration = max(durations)
        else:
            avg_duration = max_duration = 0
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_date,
            'avg_drawdown_duration': avg_duration,
            'max_drawdown_duration': max_duration,
            'drawdown_series': drawdown
        }
    
    @staticmethod
    def calculate_risk_adjusted_returns(
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk-adjusted return metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Strategy returns
        benchmark_returns : pd.Series, optional
            Benchmark returns for comparison
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        Dict
            Risk-adjusted performance metrics
        """
        returns_clean = returns.dropna()
        annual_factor = 252  # Trading days per year
        
        # Basic metrics
        total_return = (1 + returns_clean).prod() - 1
        annualized_return = (1 + total_return) ** (annual_factor / len(returns_clean)) - 1
        annualized_volatility = returns_clean.std() * np.sqrt(annual_factor)
        
        # Sharpe ratio
        excess_returns = returns_clean - risk_free_rate / annual_factor
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(annual_factor)
        
        # Sortino ratio
        downside_returns = returns_clean[returns_clean < 0]
        downside_volatility = downside_returns.std() * np.sqrt(annual_factor)
        sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else np.inf
        
        # Calmar ratio (using price series would be better, but using returns as approximation)
        max_dd = returns_clean.min()  # Simplified maximum drawdown
        calmar_ratio = annualized_return / abs(max_dd) if max_dd < 0 else np.inf
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'skewness': stats.skew(returns_clean),
            'kurtosis': stats.kurtosis(returns_clean, fisher=False)
        }
        
        # Add benchmark comparison if provided
        if benchmark_returns is not None:
            benchmark_clean = benchmark_returns.dropna()
            aligned_returns = pd.concat([returns_clean, benchmark_clean], axis=1).dropna()
            
            if len(aligned_returns) > 1:
                strategy_rets = aligned_returns.iloc[:, 0]
                benchmark_rets = aligned_returns.iloc[:, 1]
                
                # Information ratio
                active_returns = strategy_rets - benchmark_rets
                tracking_error = active_returns.std() * np.sqrt(annual_factor)
                information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(annual_factor) if active_returns.std() > 0 else 0
                
                # Beta and alpha
                covariance = np.cov(strategy_rets, benchmark_rets)[0, 1]
                benchmark_variance = benchmark_rets.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                benchmark_annual_return = (1 + benchmark_rets).prod() ** (annual_factor / len(benchmark_rets)) - 1
                alpha = annualized_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
                
                metrics.update({
                    'information_ratio': information_ratio,
                    'tracking_error': tracking_error,
                    'beta': beta,
                    'alpha': alpha
                })
        
        return metrics


class ModelDiagnostics:
    """
    Model validation and diagnostic tools.
    """
    
    @staticmethod
    def backtest_model(
        model,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str] = None
    ) -> Dict[str, any]:
        """
        Comprehensive model backtesting framework.
        
        Parameters:
        -----------
        model : sklearn-compatible model
            Model to backtest
        train_data : pd.DataFrame
            Training data
        test_data : pd.DataFrame
            Test data
        target_column : str
            Target variable column name
        feature_columns : List[str], optional
            Feature column names
            
        Returns:
        --------
        Dict
            Backtest results and metrics
        """
        if feature_columns is None:
            feature_columns = [col for col in train_data.columns if col != target_column]
        
        # Prepare data
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Generate predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        results = {
            'train_metrics': {
                'mse': mean_squared_error(y_train, y_pred_train),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': model.score(X_train, y_train)
            },
            'test_metrics': {
                'mse': mean_squared_error(y_test, y_pred_test),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': model.score(X_test, y_test)
            },
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            },
            'residuals': {
                'train': y_train - y_pred_train,
                'test': y_test - y_pred_test
            }
        }
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = dict(zip(feature_columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            results['coefficients'] = dict(zip(feature_columns, model.coef_))
        
        return results
    
    @staticmethod
    def plot_residual_analysis(residuals: pd.Series, title: str = "Residual Analysis"):
        """
        Plot comprehensive residual analysis.
        
        Parameters:
        -----------
        residuals : pd.Series
            Model residuals
        title : str
            Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        # Residuals vs fitted
        axes[0, 0].scatter(range(len(residuals)), residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        
        # QQ plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, density=True)
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        
        # ACF of residuals
        if len(residuals) > 10:
            from statsmodels.tsa.stattools import acf
            lags = min(20, len(residuals) // 4)
            acf_values = acf(residuals, nlags=lags)
            axes[1, 1].plot(range(len(acf_values)), acf_values, 'bo-')
            axes[1, 1].axhline(y=0, color='r', linestyle='-')
            axes[1, 1].axhline(y=1.96/np.sqrt(len(residuals)), color='r', linestyle='--')
            axes[1, 1].axhline(y=-1.96/np.sqrt(len(residuals)), color='r', linestyle='--')
            axes[1, 1].set_title('ACF of Residuals')
            axes[1, 1].set_xlabel('Lag')
            axes[1, 1].set_ylabel('ACF')
        
        plt.tight_layout()
        return fig

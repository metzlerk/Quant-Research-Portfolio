"""
Risk Management & Portfolio Optimization Module

This module implements advanced risk management and portfolio optimization techniques
including Extreme Value Theory for tail risk modeling, Black-Litterman optimization
with machine learning views, dynamic hedging strategies, and factor model construction.

Author: Kevin J.D. Metzler
Date: August 7, 2025
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import optimize, special
from scipy.linalg import inv, pinv
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import cvxpy as cp


class RiskManager(ABC):
    """
    Abstract base class for risk management functionality.
    Provides the framework for implementing various risk management techniques.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the risk manager with market data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with returns or prices
        """
        self.data = data
        if self._is_return_data(data):
            self.returns = data.select_dtypes(include=[np.number]).copy()
        else:
            self.returns = self._calculate_returns()
    
    @staticmethod
    def _is_return_data(data: pd.DataFrame) -> bool:
        """Heuristic check for return-like data."""
        column_names = [str(col).lower() for col in data.columns]
        if any('return' in col for col in column_names):
            return True
        
        numeric = data.select_dtypes(include=[np.number])
        if numeric.empty:
            return True
        
        if (numeric < 0).any().any():
            return True
        
        median_abs = numeric.abs().median().median()
        max_abs = numeric.abs().max().max()
        return median_abs <= 1 and max_abs <= 5
        
    def _calculate_returns(self) -> pd.DataFrame:
        """Calculate returns from price data."""
        numeric = self.data.select_dtypes(include=[np.number])
        returns_data = pd.DataFrame(index=self.data.index)
        for col in numeric.columns:
            returns_data[col] = numeric[col].pct_change()
        return returns_data.dropna()
    
    @abstractmethod
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate basic risk metrics."""
        pass


class TailRiskModeler(RiskManager):
    """
    Implements Extreme Value Theory for tail risk assessment.
    Provides both Block Maxima (GEV) and Peaks-Over-Threshold (GPD) approaches.
    """
    
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.evt_params = {}
        self.fitted_models = {}
        
    def fit_evt_model(self, returns_col: Union[str, pd.Series, pd.DataFrame], method: str = 'pot', 
                      threshold: Optional[float] = None,
                      threshold_percentile: Optional[float] = None,
                      block_size: int = 22) -> Dict[str, float]:
        """
        Fit Extreme Value Theory model to the data.
        
        Parameters:
        -----------
        returns_col : Union[str, pd.Series, pd.DataFrame]
            Column name for returns data or a returns series/dataframe
        method : str
            'pot' for Peaks-Over-Threshold or 'bm' for Block Maxima
        threshold : float, optional
            Threshold for POT method (if None, uses percentile threshold)
        threshold_percentile : float, optional
            Percentile for POT threshold (e.g., 0.05 or 5 for 5th percentile)
        block_size : int
            Block size for Block Maxima method (default: 22 for monthly blocks)
            
        Returns:
        --------
        Dict[str, float]
            Fitted parameters of the EVT model
        """
        if isinstance(returns_col, pd.Series):
            returns = returns_col.dropna()
            returns_label = returns_col.name or 'returns'
        elif isinstance(returns_col, pd.DataFrame):
            if returns_col.shape[1] != 1:
                raise ValueError("returns_col DataFrame must have exactly one column")
            returns = returns_col.iloc[:, 0].dropna()
            returns_label = returns_col.columns[0]
        else:
            returns = self.returns[returns_col].dropna()
            returns_label = returns_col
        
        if method == 'pot':
            return self._fit_pot_model(returns, threshold, threshold_percentile, returns_label)
        elif method == 'bm':
            return self._fit_block_maxima_model(returns, block_size)
        else:
            raise ValueError("Method must be 'pot' or 'bm'")
    
    def _fit_pot_model(self, returns: pd.Series, threshold: Optional[float] = None,
                       threshold_percentile: Optional[float] = None,
                       returns_label: Optional[str] = None) -> Dict[str, float]:
        """
        Fit Generalized Pareto Distribution using Peaks-Over-Threshold method.
        
        Mathematical Foundation:
        The GPD is used to model exceedances over a high threshold u:
        G(x; σ, ξ) = 1 - (1 + ξx/σ)^(-1/ξ) for ξ ≠ 0
        
        The likelihood function for GPD parameters is:
        L(σ, ξ) = σ^(-n) * ∏(1 + ξx_i/σ)^(-(1+1/ξ))
        """
        if threshold is None:
            if threshold_percentile is not None:
                percentile = threshold_percentile * 100 if 0 < threshold_percentile < 1 else threshold_percentile
                if percentile <= 0 or percentile >= 100:
                    raise ValueError("threshold_percentile must be between 0 and 100")
                threshold = abs(np.percentile(returns, percentile))
            else:
                threshold = np.percentile(returns, 95)
        
        threshold = abs(threshold)
        
        # Extract exceedances (negative returns beyond threshold)
        exceedances = -(returns[returns < -threshold] + threshold)
        
        if len(exceedances) < 10:
            warnings.warn("Insufficient exceedances for reliable estimation")
            
        # Maximum Likelihood Estimation for GPD
        def gpd_neg_log_likelihood(params):
            scale, shape = params
            if scale <= 0:
                return np.inf
            
            if shape != 0:
                if np.any(1 + shape * exceedances / scale <= 0):
                    return np.inf
                log_likelihood = -len(exceedances) * np.log(scale) - \
                               (1 + 1/shape) * np.sum(np.log(1 + shape * exceedances / scale))
            else:
                log_likelihood = -len(exceedances) * np.log(scale) - \
                               np.sum(exceedances) / scale
            
            return -log_likelihood
        
        # Initial parameter estimates
        initial_params = [np.std(exceedances), 0.1]
        
        # Optimize
        result = optimize.minimize(gpd_neg_log_likelihood, initial_params, 
                                 method='L-BFGS-B', 
                                 bounds=[(1e-6, None), (-0.5, 0.5)])
        
        if not result.success:
            warnings.warn("Optimization did not converge")
        
        scale, shape = result.x
        
        params = {
            'scale': scale,
            'shape': shape,
            'threshold': threshold,
            'n_exceedances': len(exceedances),
            'returns_col': returns_label,
            'method': 'pot'
        }
        
        self.evt_params[f'pot_{threshold:.4f}'] = params
        return params
    
    def _fit_block_maxima_model(self, returns: pd.Series, block_size: int) -> Dict[str, float]:
        """
        Fit Generalized Extreme Value Distribution using Block Maxima method.
        
        Mathematical Foundation:
        The GEV distribution has the form:
        F(x; μ, σ, ξ) = exp{-[1 + ξ(x-μ)/σ]^(-1/ξ)} for ξ ≠ 0
        """
        # Create blocks and extract minima (for negative returns)
        n_blocks = len(returns) // block_size
        block_minima = []
        
        for i in range(n_blocks):
            block_data = returns.iloc[i*block_size:(i+1)*block_size]
            block_minima.append(block_data.min())
        
        block_minima = -np.array(block_minima)  # Convert to positive exceedances
        
        # Fit GEV distribution using MLE
        def gev_neg_log_likelihood(params):
            location, scale, shape = params
            if scale <= 0:
                return np.inf
                
            z = (block_minima - location) / scale
            
            if shape != 0:
                if np.any(1 + shape * z <= 0):
                    return np.inf
                log_likelihood = -len(block_minima) * np.log(scale) - \
                               (1 + 1/shape) * np.sum(np.log(1 + shape * z)) - \
                               np.sum((1 + shape * z)**(-1/shape))
            else:
                log_likelihood = -len(block_minima) * np.log(scale) - \
                               np.sum(z) - np.sum(np.exp(-z))
            
            return -log_likelihood
        
        # Initial parameter estimates
        initial_params = [np.mean(block_minima), np.std(block_minima), 0.1]
        
        # Optimize
        result = optimize.minimize(gev_neg_log_likelihood, initial_params,
                                 method='L-BFGS-B',
                                 bounds=[(-np.inf, np.inf), (1e-6, None), (-0.5, 0.5)])
        
        if not result.success:
            warnings.warn("GEV optimization did not converge")
        
        location, scale, shape = result.x
        
        params = {
            'location': location,
            'scale': scale,
            'shape': shape,
            'block_size': block_size,
            'n_blocks': n_blocks,
            'method': 'block_maxima'
        }
        
        self.evt_params[f'bm_{block_size}'] = params
        return params
    
    def calculate_var_es(self, confidence_level: float = 0.99, 
                        method: str = 'pot', 
                        model_key: Optional[str] = None) -> Tuple[float, float]:
        """
        Calculate Value-at-Risk and Expected Shortfall using EVT.
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level for VaR/ES calculation
        method : str
            'pot' or 'bm' for the EVT method to use
        model_key : str, optional
            Specific model key to use
            
        Returns:
        --------
        Tuple[float, float]
            VaR and Expected Shortfall values
        """
        if model_key is None:
            model_key = next(k for k in self.evt_params.keys() if method in k)
        
        params = self.evt_params[model_key]
        
        if params['method'] == 'pot':
            return self._calculate_pot_var_es(params, confidence_level)
        else:
            return self._calculate_bm_var_es(params, confidence_level)
    
    def _calculate_pot_var_es(self, params: Dict, confidence_level: float) -> Tuple[float, float]:
        """Calculate VaR and ES using POT method."""
        scale = params['scale']
        shape = params['shape']
        threshold = params['threshold']
        n_exceedances = params['n_exceedances']
        n_total = len(self.returns)
        
        # Probability of exceedance
        p_exceedance = n_exceedances / n_total
        
        # Quantile level for GPD
        q = (confidence_level - (1 - p_exceedance)) / p_exceedance
        
        if q <= 0:
            returns_col = params.get('returns_col')
            if returns_col and returns_col in self.returns.columns:
                returns_series = self.returns[returns_col].dropna()
            else:
                returns_series = self.returns.iloc[:, 0].dropna()
            
            exceedances = -(returns_series[returns_series < -threshold] + threshold)
            var = -threshold
            es = -(threshold + exceedances.mean()) if len(exceedances) > 0 else var
            return var, es
        
        if q >= 1:
            warnings.warn("Confidence level too extreme for available data")
            returns_col = params.get('returns_col')
            if returns_col and returns_col in self.returns.columns:
                returns_series = self.returns[returns_col].dropna()
            else:
                returns_series = self.returns.iloc[:, 0].dropna()
            
            var = returns_series.min()
            es = returns_series[returns_series <= var].mean()
            return var, es
        
        # VaR calculation
        if shape != 0:
            gpd_quantile = (scale / shape) * ((1 - q)**(-shape) - 1)
        else:
            gpd_quantile = -scale * np.log(1 - q)
        
        var = -(threshold + gpd_quantile)
        
        # Expected Shortfall calculation
        if shape != 0 and shape < 1:
            es_gpd = (scale - shape * gpd_quantile) / (1 - shape)
            es = -(threshold + es_gpd)
        else:
            # For shape >= 1, ES is infinite; use numerical approximation
            es = var * 1.5  # Conservative approximation
            warnings.warn("Expected Shortfall may be infinite for shape >= 1")
        
        return var, es
    
    def _calculate_bm_var_es(self, params: Dict, confidence_level: float) -> Tuple[float, float]:
        """Calculate VaR and ES using Block Maxima method."""
        location = params['location']
        scale = params['scale']
        shape = params['shape']
        
        # GEV quantile function
        if shape != 0:
            gev_quantile = location - (scale / shape) * (1 - (-np.log(confidence_level))**(-shape))
        else:
            gev_quantile = location - scale * np.log(-np.log(confidence_level))
        
        var = -gev_quantile
        
        # Expected Shortfall for GEV
        if shape != 0 and shape < 1:
            # Analytical formula for ES when it exists
            gamma_term = special.gamma(1 - shape)
            es_gev = location - (scale / shape) * (1 - gamma_term * (-np.log(confidence_level))**(-shape))
            es = -es_gev
        else:
            # Numerical approximation
            es = var * 1.3
            warnings.warn("Using approximation for Expected Shortfall")
        
        return var, es
    
    def generate_tail_scenarios(self, num_scenarios: int = 1000, 
                               method: str = 'pot',
                               model_key: Optional[str] = None) -> np.ndarray:
        """
        Generate tail risk scenarios using fitted EVT model.
        
        Parameters:
        -----------
        num_scenarios : int
            Number of scenarios to generate
        method : str
            EVT method to use
        model_key : str, optional
            Specific model key to use
            
        Returns:
        --------
        np.ndarray
            Array of simulated tail scenarios
        """
        if model_key is None:
            model_key = next(k for k in self.evt_params.keys() if method in k)
        
        params = self.evt_params[model_key]
        
        if params['method'] == 'pot':
            scale = params['scale']
            shape = params['shape']
            threshold = params['threshold']
            
            # Generate from GPD
            u = np.random.uniform(0, 1, num_scenarios)
            if shape != 0:
                scenarios = (scale / shape) * ((1 - u)**(-shape) - 1)
            else:
                scenarios = -scale * np.log(1 - u)
            
            # Convert back to return space
            scenarios = -(threshold + scenarios)
            
        else:  # block_maxima
            location = params['location']
            scale = params['scale']
            shape = params['shape']
            
            # Generate from GEV
            u = np.random.uniform(0, 1, num_scenarios)
            if shape != 0:
                scenarios = location - (scale / shape) * (1 - (-np.log(u))**(-shape))
            else:
                scenarios = location - scale * np.log(-np.log(u))
            
            scenarios = -scenarios
        
        return scenarios
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics including EVT-based measures."""
        metrics = {}
        
        # Standard risk metrics
        returns_series = self.returns.iloc[:, 0] if isinstance(self.returns, pd.DataFrame) else self.returns
        
        metrics['volatility'] = returns_series.std() * np.sqrt(252)
        metrics['skewness'] = stats.skew(returns_series.dropna())
        metrics['kurtosis'] = stats.kurtosis(returns_series.dropna())
        metrics['sharpe_ratio'] = returns_series.mean() / returns_series.std() * np.sqrt(252)
        
        # Historical VaR and ES
        metrics['var_95_hist'] = np.percentile(returns_series.dropna(), 5)
        metrics['var_99_hist'] = np.percentile(returns_series.dropna(), 1)
        
        # ES (Conditional VaR)
        var_95 = metrics['var_95_hist']
        var_99 = metrics['var_99_hist']
        metrics['es_95_hist'] = returns_series[returns_series <= var_95].mean()
        metrics['es_99_hist'] = returns_series[returns_series <= var_99].mean()
        
        # EVT-based metrics if models are fitted
        if self.evt_params:
            try:
                var_evt, es_evt = self.calculate_var_es(confidence_level=0.99)
                metrics['var_99_evt'] = var_evt
                metrics['es_99_evt'] = es_evt
            except (StopIteration, KeyError, ValueError) as exc:
                warnings.warn(f"EVT risk metric calculation failed: {exc}")
        
        return metrics


class BlackLittermanOptimizer(RiskManager):
    """
    Portfolio optimization using the Black-Litterman approach enhanced with machine learning views.
    
    The Black-Litterman model provides a Bayesian framework for portfolio optimization
    that combines market equilibrium with investor views.
    """
    
    def __init__(self, market_data: pd.DataFrame, market_cap_weights: Optional[pd.Series] = None):
        """
        Initialize Black-Litterman optimizer.
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Market price/return data for assets
        market_cap_weights : pd.Series, optional
            Market capitalization weights for equilibrium portfolio
        """
        super().__init__(market_data)
        self.market_cap_weights = market_cap_weights
        self.covariance_matrix = None
        self.expected_returns = None
        self.tau = 0.025  # Default uncertainty parameter
        self.risk_aversion = 3.0  # Default risk aversion coefficient
        
    def estimate_market_parameters(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Estimate market equilibrium expected returns and covariance matrix.
        
        Returns:
        --------
        Tuple[pd.Series, pd.DataFrame]
            Expected returns and covariance matrix
        """
        # Calculate sample covariance matrix
        returns_df = self.returns.select_dtypes(include=[np.number])
        self.covariance_matrix = returns_df.cov() * 252  # Annualized
        
        # If market cap weights not provided, use equal weights
        if self.market_cap_weights is None:
            self.market_cap_weights = pd.Series(
                1/len(returns_df.columns), 
                index=returns_df.columns
            )
        else:
            missing = returns_df.columns.difference(self.market_cap_weights.index)
            if len(missing) > 0:
                warnings.warn(f"Market cap weights missing assets: {list(missing)}; setting to 0")
                self.market_cap_weights = self.market_cap_weights.reindex(returns_df.columns).fillna(0)
            else:
                self.market_cap_weights = self.market_cap_weights.reindex(returns_df.columns)
        
        # Market equilibrium expected returns using reverse optimization
        # π = δ * Σ * w_market
        self.expected_returns = self.risk_aversion * self.covariance_matrix.dot(self.market_cap_weights)
        
        return self.expected_returns, self.covariance_matrix
    
    def generate_ml_views(self, features: pd.DataFrame, 
                         model_type: str = 'random_forest',
                         prediction_horizon: int = 22,
                         confidence_scaling: float = 1.0) -> Dict[str, Any]:
        """
        Generate views using machine learning models.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature data for ML models (e.g., alternative data, technical indicators)
        model_type : str
            Type of ML model ('random_forest', 'linear', 'svm')
        prediction_horizon : int
            Prediction horizon in days
        confidence_scaling : float
            Scaling factor for view confidence
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing views matrix P, views vector Q, and uncertainty matrix Ω
        """
        if self.covariance_matrix is None:
            self.estimate_market_parameters()
        
        returns_df = self.returns.select_dtypes(include=[np.number])
        
        # Prepare target variables (forward returns)
        targets = {}
        for asset in returns_df.columns:
            target = returns_df[asset].shift(-prediction_horizon).dropna()
            targets[asset] = target
        
        # Align features with target data
        min_length = min(len(targets[asset]) for asset in targets.keys())
        aligned_features = features.iloc[:min_length]
        
        # Train ML models for each asset
        models = {}
        predictions = {}
        feature_importances = {}
        
        for asset in returns_df.columns:
            if model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Model type {model_type} not implemented")
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Fit model
            target_aligned = targets[asset].iloc[:len(aligned_features)]
            feature_data = aligned_features.select_dtypes(include=[np.number]).fillna(0)
            
            model.fit(feature_data, target_aligned)
            models[asset] = model
            
            # Generate predictions
            latest_features = feature_data.iloc[-1:].values
            pred = model.predict(latest_features)[0]
            predictions[asset] = pred
            
            # Store feature importances
            if hasattr(model, 'feature_importances_'):
                feature_importances[asset] = model.feature_importances_
        
        # Convert predictions to views
        P = np.eye(len(returns_df.columns))  # Identity matrix for absolute views
        Q = np.array([predictions[asset] for asset in returns_df.columns])
        
        # Calculate view uncertainty based on prediction confidence
        # Use cross-validation to estimate prediction errors
        view_uncertainties = []
        
        for asset in returns_df.columns:
            model = models[asset]
            target_aligned = targets[asset].iloc[:len(aligned_features)]
            feature_data = aligned_features.select_dtypes(include=[np.number]).fillna(0)
            
            # Cross-validation to estimate prediction error
            cv_scores = []
            tscv = TimeSeriesSplit(n_splits=5)
            
            for train_idx, test_idx in tscv.split(feature_data):
                X_train, X_test = feature_data.iloc[train_idx], feature_data.iloc[test_idx]
                y_train, y_test = target_aligned.iloc[train_idx], target_aligned.iloc[test_idx]
                
                model_cv = RandomForestRegressor(n_estimators=50, random_state=42)
                model_cv.fit(X_train, y_train)
                pred_cv = model_cv.predict(X_test)
                
                mse = np.mean((y_test - pred_cv)**2)
                cv_scores.append(mse)
            
            avg_mse = np.mean(cv_scores)
            view_uncertainties.append(avg_mse * confidence_scaling)
        
        # Uncertainty matrix Ω
        Omega = np.diag(view_uncertainties)
        
        return {
            'P': P,
            'Q': Q,
            'Omega': Omega,
            'models': models,
            'predictions': predictions,
            'feature_importances': feature_importances
        }
    
    def optimize(self, views: Optional[Dict[str, Any]] = None,
                risk_aversion: float = 3.0,
                constraints: Optional[Dict[str, Any]] = None) -> pd.Series:
        """
        Optimize portfolio using Black-Litterman model.
        
        Parameters:
        -----------
        views : Dict[str, Any], optional
            Views dictionary from generate_ml_views()
        risk_aversion : float
            Risk aversion parameter δ
        constraints : Dict[str, Any], optional
            Portfolio constraints (leverage, sector, etc.)
            
        Returns:
        --------
        pd.Series
            Optimal portfolio weights
        """
        if self.covariance_matrix is None:
            self.estimate_market_parameters()
        
        self.risk_aversion = risk_aversion
        
        # If no views provided, use market equilibrium
        if views is None:
            optimal_weights = self.market_cap_weights.copy()
        else:
            # Black-Litterman formula
            P = views['P']
            Q = views['Q']
            Omega = views['Omega']
            
            # Prior precision matrix
            tau_sigma_inv = inv(self.tau * self.covariance_matrix.values)
            
            # Views precision matrix
            P_omega_inv_P = P.T @ inv(Omega) @ P
            
            # Posterior precision matrix
            posterior_precision = tau_sigma_inv + P_omega_inv_P
            
            # Posterior expected returns
            prior_term = tau_sigma_inv @ self.expected_returns.values
            views_term = P.T @ inv(Omega) @ Q
            
            posterior_returns = inv(posterior_precision) @ (prior_term + views_term)
            
            # Optimal weights using mean-variance optimization
            # w* = (δΣ)^(-1) * μ
            optimal_weights_values = inv(self.risk_aversion * self.covariance_matrix.values) @ posterior_returns
            
            optimal_weights = pd.Series(
                optimal_weights_values, 
                index=self.covariance_matrix.index
            )
        
        # Apply constraints if provided
        if constraints:
            optimal_weights = self._apply_constraints(optimal_weights, constraints)
        
        # Normalize weights to sum to 1
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        return optimal_weights
    
    def _apply_constraints(self, weights: pd.Series, constraints: Dict[str, Any]) -> pd.Series:
        """
        Apply portfolio constraints using convex optimization.
        
        Parameters:
        -----------
        weights : pd.Series
            Unconstrained optimal weights
        constraints : Dict[str, Any]
            Portfolio constraints
            
        Returns:
        --------
        pd.Series
            Constrained optimal weights
        """
        n_assets = len(weights)
        w = cp.Variable(n_assets)
        
        # Objective function: minimize tracking error from unconstrained optimum
        objective = cp.Minimize(cp.quad_form(w - weights.values, np.eye(n_assets)))
        
        # Constraints
        constraints_list = []
        
        # Budget constraint (sum to 1)
        constraints_list.append(cp.sum(w) == 1)
        
        # Long-only constraint (if specified)
        if constraints.get('long_only', False):
            constraints_list.append(w >= 0)
        
        # Maximum leverage constraint
        if 'max_leverage' in constraints:
            constraints_list.append(cp.norm(w, 1) <= constraints['max_leverage'])
        
        # Individual weight constraints
        if 'max_weight' in constraints:
            constraints_list.append(w <= constraints['max_weight'])
        
        if 'min_weight' in constraints:
            constraints_list.append(w >= constraints['min_weight'])
        
        # Solve optimization problem
        prob = cp.Problem(objective, constraints_list)
        prob.solve()
        
        if prob.status not in ["infeasible", "unbounded"]:
            constrained_weights = pd.Series(w.value, index=weights.index)
        else:
            warnings.warn("Constraint optimization failed, using unconstrained weights")
            constrained_weights = weights
        
        return constrained_weights
    
    def analyze_portfolio(self, weights: pd.Series) -> Dict[str, float]:
        """
        Analyze portfolio characteristics.
        
        Parameters:
        -----------
        weights : pd.Series
            Portfolio weights
            
        Returns:
        --------
        Dict[str, float]
            Portfolio analytics
        """
        if self.covariance_matrix is None:
            self.estimate_market_parameters()
        
        # Portfolio expected return
        portfolio_return = weights.dot(self.expected_returns)
        
        # Portfolio variance and volatility
        portfolio_variance = weights.dot(self.covariance_matrix).dot(weights)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = portfolio_return / portfolio_volatility
        
        # Diversification metrics
        concentration = np.sum(weights**2)  # Herfindahl index
        effective_assets = 1 / concentration
        
        # Maximum weight
        max_weight = weights.abs().max()
        
        analytics = {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'concentration': concentration,
            'effective_assets': effective_assets,
            'max_weight': max_weight,
            'num_assets': len(weights),
            'long_positions': (weights > 0).sum(),
            'short_positions': (weights < 0).sum()
        }
        
        return analytics
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics for the optimized portfolio."""
        if self.market_cap_weights is not None:
            portfolio_analytics = self.analyze_portfolio(self.market_cap_weights)
            return portfolio_analytics
        else:
            return super().calculate_risk_metrics()


class DynamicHedger(RiskManager):
    """
    Implementation of dynamic hedging strategies for portfolio risk management.
    
    Uses stochastic control theory and rolling optimization to maintain
    optimal hedge ratios for risk reduction.
    """
    
    def __init__(self, portfolio_data: pd.DataFrame):
        """
        Initialize dynamic hedger.
        
        Parameters:
        -----------
        portfolio_data : pd.DataFrame
            Portfolio return or value data
        """
        super().__init__(portfolio_data)
        self.hedge_instruments = {}
        self.hedge_ratios = {}
        self.hedge_performance = {}
        
    def calculate_optimal_hedge(self, hedging_instruments: List[str],
                               portfolio_col: str,
                               method: str = 'minimum_variance',
                               lookback_period: int = 252,
                               rebalance_frequency: int = 22) -> pd.DataFrame:
        """
        Calculate optimal hedge ratios using various methods.
        
        Parameters:
        -----------
        hedging_instruments : List[str]
            List of hedging instrument column names
        portfolio_col : str
            Portfolio return column name
        method : str
            Hedging method ('minimum_variance', 'beta_hedge', 'correlation_weighted')
        lookback_period : int
            Lookback period for parameter estimation
        rebalance_frequency : int
            Rebalancing frequency in days
            
        Returns:
        --------
        pd.DataFrame
            Time series of optimal hedge ratios
        """
        returns_df = self.returns[[portfolio_col] + hedging_instruments].dropna()
        
        hedge_ratios = pd.DataFrame(index=returns_df.index, columns=hedging_instruments)
        
        for i in range(lookback_period, len(returns_df), rebalance_frequency):
            # Get rolling window data
            window_data = returns_df.iloc[i-lookback_period:i]
            
            if method == 'minimum_variance':
                ratios = self._calculate_minimum_variance_hedge(
                    window_data[portfolio_col], 
                    window_data[hedging_instruments]
                )
            elif method == 'beta_hedge':
                ratios = self._calculate_beta_hedge(
                    window_data[portfolio_col], 
                    window_data[hedging_instruments]
                )
            elif method == 'correlation_weighted':
                ratios = self._calculate_correlation_weighted_hedge(
                    window_data[portfolio_col], 
                    window_data[hedging_instruments]
                )
            else:
                raise ValueError(f"Unknown hedging method: {method}")
            
            # Fill forward until next rebalance
            end_idx = min(i + rebalance_frequency, len(returns_df))
            hedge_ratios.iloc[i:end_idx] = ratios
        
        # Forward fill any remaining NaN values
        hedge_ratios = hedge_ratios.fillna(method='ffill')
        
        self.hedge_ratios[method] = hedge_ratios
        return hedge_ratios
    
    def _calculate_minimum_variance_hedge(self, portfolio_returns: pd.Series,
                                        hedge_returns: pd.DataFrame) -> pd.Series:
        """
        Calculate minimum variance hedge ratios.
        
        Mathematical Foundation:
        The minimum variance hedge ratio minimizes:
        Var(R_p - h'R_h) where h is the vector of hedge ratios
        
        Solution: h* = Σ_hh^(-1) * Σ_hp
        """
        aligned_data = hedge_returns.join(portfolio_returns.rename('portfolio')).dropna()
        aligned_hedge = aligned_data[hedge_returns.columns]
        aligned_portfolio = aligned_data['portfolio']
        
        # Covariance matrix of hedge instruments
        cov_hedge = aligned_hedge.cov().values
        
        # Covariance between portfolio and hedge instruments
        cov_portfolio_hedge = aligned_hedge.apply(lambda x: x.cov(aligned_portfolio)).values
        
        # Handle singular covariance matrices
        try:
            hedge_ratios = np.linalg.solve(cov_hedge, cov_portfolio_hedge)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for singular matrices
            hedge_ratios = pinv(cov_hedge) @ cov_portfolio_hedge
        
        return pd.Series(hedge_ratios, index=hedge_returns.columns)
    
    def _calculate_beta_hedge(self, portfolio_returns: pd.Series,
                            hedge_returns: pd.DataFrame) -> pd.Series:
        """
        Calculate beta-based hedge ratios.
        
        Uses individual beta of each hedge instrument with the portfolio.
        """
        hedge_ratios = []
        
        for instrument in hedge_returns.columns:
            # Calculate beta: β = Cov(R_p, R_h) / Var(R_h)
            covariance = portfolio_returns.cov(hedge_returns[instrument])
            variance = hedge_returns[instrument].var()
            
            beta = covariance / variance if variance > 0 else 0
            hedge_ratios.append(beta)
        
        return pd.Series(hedge_ratios, index=hedge_returns.columns)
    
    def _calculate_correlation_weighted_hedge(self, portfolio_returns: pd.Series,
                                           hedge_returns: pd.DataFrame) -> pd.Series:
        """
        Calculate correlation-weighted hedge ratios.
        
        Weights hedge instruments by their correlation with the portfolio.
        """
        correlations = hedge_returns.corrwith(portfolio_returns).abs()
        
        # Normalize correlations to sum to 1
        if correlations.sum() > 0:
            hedge_ratios = correlations / correlations.sum()
        else:
            hedge_ratios = pd.Series(0, index=hedge_returns.columns)
        
        return hedge_ratios
    
    def apply_hedge(self, hedge_ratios: pd.DataFrame,
                   portfolio_col: str,
                   transaction_costs: float = 0.001) -> pd.DataFrame:
        """
        Apply hedge and calculate hedged portfolio returns.
        
        Parameters:
        -----------
        hedge_ratios : pd.DataFrame
            Time series of hedge ratios
        portfolio_col : str
            Portfolio column name
        transaction_costs : float
            Transaction costs as percentage of trade value
            
        Returns:
        --------
        pd.DataFrame
            Hedged portfolio performance
        """
        returns_df = self.returns[[portfolio_col] + list(hedge_ratios.columns)].dropna()
        
        # Align hedge ratios with returns
        aligned_ratios = hedge_ratios.reindex(returns_df.index).fillna(method='ffill')
        
        # Calculate hedged returns
        hedged_returns = returns_df[portfolio_col].copy()
        
        for instrument in hedge_ratios.columns:
            hedge_return = aligned_ratios[instrument] * returns_df[instrument]
            hedged_returns = hedged_returns - hedge_return
        
        # Apply transaction costs on hedge rebalancing
        ratio_changes = aligned_ratios.diff().abs().sum(axis=1)
        transaction_cost_drag = ratio_changes * transaction_costs
        
        hedged_returns = hedged_returns - transaction_cost_drag
        
        # Create performance DataFrame
        performance = pd.DataFrame({
            'original_portfolio': returns_df[portfolio_col],
            'hedged_portfolio': hedged_returns,
            'hedge_return': hedged_returns - returns_df[portfolio_col],
            'transaction_costs': transaction_cost_drag
        })
        
        return performance
    
    def evaluate_hedge_effectiveness(self, hedged_performance: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the effectiveness of the hedging strategy.
        
        Parameters:
        -----------
        hedged_performance : pd.DataFrame
            Output from apply_hedge method
            
        Returns:
        --------
        Dict[str, float]
            Hedge effectiveness metrics
        """
        original = hedged_performance['original_portfolio']
        hedged = hedged_performance['hedged_portfolio']
        
        # Risk reduction metrics
        original_vol = original.std() * np.sqrt(252)
        hedged_vol = hedged.std() * np.sqrt(252)
        vol_reduction = (original_vol - hedged_vol) / original_vol
        
        # Correlation-based hedge effectiveness
        hedge_effectiveness = 1 - (hedged.var() / original.var())
        
        # Maximum drawdown comparison
        original_cumret = (1 + original).cumprod()
        hedged_cumret = (1 + hedged).cumprod()
        
        original_dd = (original_cumret / original_cumret.cummax() - 1).min()
        hedged_dd = (hedged_cumret / hedged_cumret.cummax() - 1).min()
        
        # Sharpe ratio comparison
        original_sharpe = original.mean() / original.std() * np.sqrt(252)
        hedged_sharpe = hedged.mean() / hedged.std() * np.sqrt(252)
        
        metrics = {
            'volatility_reduction': vol_reduction,
            'hedge_effectiveness': hedge_effectiveness,
            'original_volatility': original_vol,
            'hedged_volatility': hedged_vol,
            'original_max_drawdown': original_dd,
            'hedged_max_drawdown': hedged_dd,
            'original_sharpe': original_sharpe,
            'hedged_sharpe': hedged_sharpe,
            'avg_transaction_costs': hedged_performance['transaction_costs'].mean()
        }
        
        return metrics
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics for hedged portfolios."""
        base_metrics = super().calculate_risk_metrics()
        
        # Add hedging-specific metrics if available
        if self.hedge_performance:
            for method, performance in self.hedge_performance.items():
                effectiveness = self.evaluate_hedge_effectiveness(performance)
                base_metrics.update({f'{method}_{k}': v for k, v in effectiveness.items()})
        
        return base_metrics


class FactorModelBuilder(RiskManager):
    """
    Construction and validation of multi-factor risk models.
    
    Implements various factor model approaches including:
    - Fundamental factor models
    - Statistical factor models (PCA)
    - Macroeconomic factor models
    """
    
    def __init__(self, asset_returns: pd.DataFrame, factor_data: Optional[pd.DataFrame] = None):
        """
        Initialize factor model builder.
        
        Parameters:
        -----------
        asset_returns : pd.DataFrame
            Asset return data
        factor_data : pd.DataFrame, optional
            External factor data (for fundamental/macro models)
        """
        super().__init__(asset_returns)
        self.factor_data = factor_data
        self.factor_loadings = {}
        self.factor_returns = {}
        self.specific_risks = {}
        self.factor_models = {}
        
    def build_statistical_factor_model(self, n_factors: int = 5,
                                     method: str = 'pca') -> Dict[str, Any]:
        """
        Build statistical factor model using PCA or other dimensionality reduction.
        
        Parameters:
        -----------
        n_factors : int
            Number of factors to extract
        method : str
            Method for factor extraction ('pca', 'ica')
            
        Returns:
        --------
        Dict[str, Any]
            Factor model results
        """
        returns_df = self.returns.select_dtypes(include=[np.number]).dropna()

        # Defensive checks: remove non-finite or zero-variance columns
        finite_mask = returns_df.columns[returns_df.replace([np.inf, -np.inf], np.nan).notna().all()]
        returns_df = returns_df.loc[:, finite_mask]
        var_series = returns_df.var()
        nonconstant_cols = var_series[var_series > 0].index.tolist()
        returns_df = returns_df[nonconstant_cols]

        if returns_df.empty:
            raise ValueError("No valid numeric return columns available for PCA")

        if method == 'pca':
            # Ensure n_factors does not exceed available dimensions
            n_factors = min(n_factors, returns_df.shape[1])

            # Standardize returns
            scaler = StandardScaler()
            standardized_returns = scaler.fit_transform(returns_df)
            
            # Apply PCA
            pca = PCA(n_components=n_factors)
            factor_scores = pca.fit_transform(standardized_returns)
            
            # Factor loadings (eigenvectors)
            factor_loadings = pd.DataFrame(
                pca.components_.T,
                index=returns_df.columns,
                columns=[f'Factor_{i+1}' for i in range(n_factors)]
            )
            
            # Factor returns (principal components)
            factor_returns = pd.DataFrame(
                factor_scores,
                index=returns_df.index,
                columns=[f'Factor_{i+1}' for i in range(n_factors)]
            )
            
            # Explained variance
            explained_variance = pca.explained_variance_ratio_
            
            # Calculate specific (idiosyncratic) risk
            reconstructed_returns = factor_scores @ pca.components_
            residuals = standardized_returns - reconstructed_returns
            specific_risks = pd.Series(
                np.var(residuals, axis=0),
                index=returns_df.columns
            )
            
            model_results = {
                'method': 'pca',
                'factor_loadings': factor_loadings,
                'factor_returns': factor_returns,
                'specific_risks': specific_risks,
                'explained_variance': explained_variance,
                'cumulative_variance': np.cumsum(explained_variance),
                'pca_model': pca,
                'scaler': scaler
            }
            
        else:
            raise ValueError(f"Method {method} not implemented")
        
        self.factor_models['statistical'] = model_results
        return model_results
    
    def build_fundamental_factor_model(self, factor_exposures: pd.DataFrame) -> Dict[str, Any]:
        """
        Build fundamental factor model using pre-specified factor exposures.
        
        Parameters:
        -----------
        factor_exposures : pd.DataFrame
            Asset exposures to fundamental factors
            
        Returns:
        --------
        Dict[str, Any]
            Factor model results
        """
        returns_df = self.returns.select_dtypes(include=[np.number]).dropna()

        # Align factor exposures with returns and remove assets with missing exposures
        common_assets = returns_df.columns.intersection(factor_exposures.index)
        aligned_exposures = factor_exposures.loc[common_assets].dropna(how='any')
        common_assets = aligned_exposures.index.intersection(returns_df.columns)
        aligned_returns = returns_df[common_assets]
        aligned_exposures = aligned_exposures.loc[common_assets]
        
        # Estimate factor returns using cross-sectional regression
        factor_returns = pd.DataFrame(
            index=aligned_returns.index,
            columns=aligned_exposures.columns
        )
        
        specific_returns = pd.DataFrame(
            index=aligned_returns.index,
            columns=common_assets
        )
        
        # Use pseudo-inverse to handle collinearity / singular matrices robustly
        X = aligned_exposures.values
        pinvX = pinv(X)

        for date in aligned_returns.index:
            # Cross-sectional regression: R_i,t = X_i * f_t + ε_i,t
            y = aligned_returns.loc[date].values
            try:
                factor_ret = pinvX @ y
                factor_returns.loc[date] = factor_ret

                # Calculate specific returns
                predicted_returns = X @ factor_ret
                specific_returns.loc[date] = y - predicted_returns

            except Exception:
                # Handle unexpected errors
                factor_returns.loc[date] = np.nan
                specific_returns.loc[date] = np.nan
        
        # Calculate specific risks
        specific_risks = specific_returns.var()
        
        # Factor loadings are the exposures themselves
        factor_loadings = aligned_exposures
        
        model_results = {
            'method': 'fundamental',
            'factor_loadings': factor_loadings,
            'factor_returns': factor_returns.dropna(),
            'specific_returns': specific_returns.dropna(),
            'specific_risks': specific_risks,
            'factor_exposures': aligned_exposures
        }
        
        self.factor_models['fundamental'] = model_results
        return model_results
    
    def build_macroeconomic_factor_model(self, macro_factors: pd.DataFrame) -> Dict[str, Any]:
        """
        Build macroeconomic factor model using time series regression.
        
        Parameters:
        -----------
        macro_factors : pd.DataFrame
            Macroeconomic factor time series
            
        Returns:
        --------
        Dict[str, Any]
            Factor model results
        """
        returns_df = self.returns.select_dtypes(include=[np.number]).dropna()
        
        # Align data
        common_dates = returns_df.index.intersection(macro_factors.index)
        aligned_returns = returns_df.loc[common_dates]
        aligned_factors = macro_factors.loc[common_dates]
        
        # Time series regression for each asset
        factor_loadings = pd.DataFrame(
            index=aligned_returns.columns,
            columns=aligned_factors.columns
        )
        
        alphas = pd.Series(index=aligned_returns.columns)
        specific_risks = pd.Series(index=aligned_returns.columns)
        r_squared = pd.Series(index=aligned_returns.columns)
        
        for asset in aligned_returns.columns:
            y = aligned_returns[asset].values
            X = np.column_stack([np.ones(len(y)), aligned_factors.values])  # Add intercept
            
            try:
                # OLS regression
                coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
                
                alphas[asset] = coefficients[0]  # Intercept
                factor_loadings.loc[asset] = coefficients[1:]  # Factor loadings
                
                # Calculate residuals and specific risk
                predicted = X @ coefficients
                residuals = y - predicted
                specific_risks[asset] = np.var(residuals)
                
                # R-squared
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r_squared[asset] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
            except np.linalg.LinAlgError:
                factor_loadings.loc[asset] = np.nan
                alphas[asset] = np.nan
                specific_risks[asset] = np.nan
                r_squared[asset] = np.nan
        
        model_results = {
            'method': 'macroeconomic',
            'factor_loadings': factor_loadings.dropna(),
            'factor_returns': aligned_factors,
            'alphas': alphas.dropna(),
            'specific_risks': specific_risks.dropna(),
            'r_squared': r_squared.dropna(),
            'macro_factors': aligned_factors
        }
        
        self.factor_models['macroeconomic'] = model_results
        return model_results
    
    def validate_factor_model(self, model_type: str,
                            validation_period: int = 252) -> Dict[str, float]:
        """
        Validate factor model using out-of-sample tests.
        
        Parameters:
        -----------
        model_type : str
            Type of model to validate
        validation_period : int
            Number of periods for out-of-sample validation
            
        Returns:
        --------
        Dict[str, float]
            Validation metrics
        """
        if model_type not in self.factor_models:
            raise ValueError(f"Model type {model_type} not found")
        
        model = self.factor_models[model_type]
        returns_df = self.returns.select_dtypes(include=[np.number]).dropna()
        
        # Split data
        split_point = len(returns_df) - validation_period
        train_returns = returns_df.iloc[:split_point]
        test_returns = returns_df.iloc[split_point:]
        
        if model_type == 'statistical':
            # Use fitted PCA model to predict test returns
            scaler = model['scaler']
            pca_model = model['pca_model']
            
            test_standardized = scaler.transform(test_returns)
            test_factors = pca_model.transform(test_standardized)
            predicted_standardized = test_factors @ pca_model.components_
            
            prediction_errors = test_standardized - predicted_standardized
            
        elif model_type == 'fundamental':
            # Reconstruct returns using factor model
            factor_loadings = model['factor_loadings']
            factor_returns = model['factor_returns']
            
            # Get factor returns for test period
            test_factor_returns = factor_returns.loc[test_returns.index]
            
            # Predict returns
            predicted_returns = factor_loadings.T @ test_factor_returns.T
            prediction_errors = test_returns.values - predicted_returns.T
            
        else:  # macroeconomic
            factor_loadings = model['factor_loadings']
            macro_factors = model['macro_factors']
            alphas = model['alphas']
            
            test_macro = macro_factors.loc[test_returns.index]
            
            predicted_returns = np.outer(np.ones(len(test_returns)), alphas.values) + \
                              test_macro.values @ factor_loadings.T
            
            prediction_errors = test_returns.values - predicted_returns
        
        # Calculate validation metrics
        mse = np.mean(prediction_errors**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(prediction_errors))
        
        # Information coefficient (average correlation between predicted and actual)
        ic_values = []
        for i in range(len(test_returns)):
            if model_type == 'statistical':
                actual = test_returns.iloc[i].values
                predicted = predicted_standardized[i]
            else:
                actual = test_returns.iloc[i].values
                predicted = predicted_returns[i]
            
            corr = np.corrcoef(actual, predicted)[0, 1]
            if not np.isnan(corr):
                ic_values.append(corr)
        
        avg_ic = np.mean(ic_values) if ic_values else 0
        ic_t_stat = avg_ic / np.std(ic_values) * np.sqrt(len(ic_values)) if len(ic_values) > 1 else 0
        
        validation_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'avg_ic': avg_ic,
            'ic_t_statistic': ic_t_stat,
            'validation_periods': len(test_returns)
        }
        
        return validation_metrics
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics incorporating factor model insights."""
        base_metrics = super().calculate_risk_metrics()
        
        # Add factor model specific metrics
        for model_type, model in self.factor_models.items():
            if model_type == 'statistical':
                # Factor concentration
                explained_var = model['explained_variance']
                base_metrics[f'{model_type}_first_factor_var'] = explained_var[0]
                base_metrics[f'{model_type}_top3_factor_var'] = explained_var[:3].sum()
                
            elif model_type == 'fundamental':
                # Factor exposure concentration
                factor_loadings = model['factor_loadings']
                max_exposure = factor_loadings.abs().max().max()
                base_metrics[f'{model_type}_max_factor_exposure'] = max_exposure
                
            # Average specific risk
            specific_risks = model['specific_risks']
            base_metrics[f'{model_type}_avg_specific_risk'] = specific_risks.mean()
            base_metrics[f'{model_type}_specific_risk_concentration'] = specific_risks.std() / specific_risks.mean()
        
        return base_metrics

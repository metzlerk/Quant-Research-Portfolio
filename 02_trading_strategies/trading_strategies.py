"""
Advanced Trading Strategies Implementation

This module implements institutional-quality systematic trading strategies including:
- Statistical arbitrage using cointegration
- Cross-sectional and time-series momentum
- Mean reversion with Ornstein-Uhlenbeck modeling
- Machine learning enhanced signal generation
- Multi-asset portfolio strategies

Author: Kevin J. Metzler
Mathematical Foundation: Based on rigorous statistical inference and 
econometric theory suitable for institutional deployment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

@dataclass
class StrategyConfig:
    """Configuration class for trading strategies."""
    name: str
    lookback_period: int = 252
    holding_period: int = 22
    rebalance_freq: str = 'monthly'
    transaction_cost: float = 0.001
    max_position_size: float = 0.1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    Provides common interface and utilities for all trading strategies.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.signals = None
        self.positions = None
        self.returns = None
        self.performance_metrics = None
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on strategy logic."""
        pass
    
    @abstractmethod
    def calculate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert signals to position weights."""
        pass
    
    def backtest(self, 
                 price_data: pd.DataFrame, 
                 benchmark: Optional[pd.Series] = None) -> Dict[str, any]:
        """
        Comprehensive backtesting framework with transaction costs.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Price data for assets
        benchmark : pd.Series, optional
            Benchmark returns for comparison
            
        Returns:
        --------
        Dict containing backtest results
        """
        # Generate signals
        signals = self.generate_signals(price_data)
        
        # Calculate positions
        positions = self.calculate_positions(signals)
        
        # Calculate returns
        returns_data = price_data.pct_change().dropna()
        
        # Calculate strategy returns with transaction costs
        strategy_returns = self._calculate_strategy_returns(
            positions, returns_data, self.config.transaction_cost
        )
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(
            strategy_returns, benchmark
        )
        
        # Store results
        self.signals = signals
        self.positions = positions
        self.returns = strategy_returns
        self.performance_metrics = performance
        
        return {
            'signals': signals,
            'positions': positions,
            'returns': strategy_returns,
            'performance': performance
        }
    
    def _calculate_strategy_returns(self, 
                                   positions: pd.DataFrame, 
                                   returns: pd.DataFrame,
                                   transaction_cost: float) -> pd.Series:
        """Calculate strategy returns including transaction costs."""
        
        # Align data
        common_index = positions.index.intersection(returns.index)
        pos_aligned = positions.loc[common_index]
        ret_aligned = returns.loc[common_index]
        
        # Calculate gross returns
        gross_returns = (pos_aligned.shift(1) * ret_aligned).sum(axis=1)
        
        # Calculate turnover and transaction costs
        position_changes = pos_aligned.diff().abs()
        turnover = position_changes.sum(axis=1)
        transaction_costs = turnover * transaction_cost
        
        # Net returns after transaction costs
        net_returns = gross_returns - transaction_costs
        
        return net_returns.dropna()
    
    def _calculate_performance_metrics(self, 
                                     strategy_returns: pd.Series,
                                     benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        returns = strategy_returns.dropna()
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns, fisher=False)
        }
        
        # Add benchmark comparison if provided
        if benchmark is not None:
            aligned_data = pd.concat([returns, benchmark], axis=1).dropna()
            if len(aligned_data) > 1:
                strategy_rets = aligned_data.iloc[:, 0]
                benchmark_rets = aligned_data.iloc[:, 1]
                
                # Information ratio
                active_returns = strategy_rets - benchmark_rets
                tracking_error = active_returns.std() * np.sqrt(252)
                information_ratio = active_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
                
                # Beta and alpha
                if benchmark_rets.var() > 0:
                    beta = strategy_rets.cov(benchmark_rets) / benchmark_rets.var()
                    alpha = strategy_rets.mean() - beta * benchmark_rets.mean()
                    alpha_annualized = alpha * 252
                else:
                    beta = 0
                    alpha_annualized = 0
                
                metrics.update({
                    'information_ratio': information_ratio,
                    'tracking_error': tracking_error,
                    'beta': beta,
                    'alpha_annualized': alpha_annualized
                })
        
        return metrics

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Ornstein-Uhlenbeck Process
    
    Mathematical Foundation:
    ------------------------
    The strategy assumes asset prices follow an OU process:
    dX_t = theta(mu - X_t)dt + sigma dW_t
    
    Trading signals are generated based on the standardized residual:
    z_t = (X_t - mu) / sigma
    
    Positions are sized proportional to the deviation from equilibrium.
    """
    
    def __init__(self, config: StrategyConfig, 
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5):
        super().__init__(config)
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.ou_params = {}
    
    def fit_ou_process(self, price_series: pd.Series) -> Dict[str, float]:
        """
        Fit Ornstein-Uhlenbeck process to price series using MLE.
        
        Parameters:
        -----------
        price_series : pd.Series
            Price time series data
            
        Returns:
        --------
        Dict containing OU parameters
        """
        # Convert to log prices
        log_prices = np.log(price_series.dropna())
        
        # Calculate first differences
        delta_x = log_prices.diff().dropna()
        x_lagged = log_prices.shift(1).dropna()
        
        # Align data
        common_index = delta_x.index.intersection(x_lagged.index)
        delta_x = delta_x.loc[common_index]
        x_lagged = x_lagged.loc[common_index]
        
        # OLS regression: Delta x_t = a + b*x_{t-1} + epsilon_t
        # where theta = -b, mu = -a/b, sigma^2 = Var(epsilon)
        
        X = np.column_stack([np.ones(len(x_lagged)), x_lagged])
        y = delta_x.values
        
        # OLS estimation
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        
        # Extract OU parameters
        a, b = beta
        theta = -b
        mu = -a / b if b != 0 else np.mean(log_prices)
        sigma = np.std(residuals)
        
        # Half-life of mean reversion
        half_life = np.log(2) / theta if theta > 0 else np.inf
        
        return {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'half_life': half_life,
            'r_squared': 1 - np.var(residuals) / np.var(y)
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals for each asset."""
        
        signals = pd.DataFrame(index=data.index, columns=data.columns)
        
        for asset in data.columns:
            price_series = data[asset].dropna()
            
            # Fit OU process on rolling window
            rolling_signals = []
            
            for i in range(self.config.lookback_period, len(price_series)):
                # Get rolling window
                window_data = price_series.iloc[i-self.config.lookback_period:i]
                
                # Fit OU process
                ou_params = self.fit_ou_process(window_data)
                self.ou_params[f"{asset}_{i}"] = ou_params
                
                # Calculate current z-score
                current_price = price_series.iloc[i]
                log_price = np.log(current_price)
                z_score = (log_price - ou_params['mu']) / ou_params['sigma']
                
                # Generate signal
                if abs(z_score) > self.entry_threshold and ou_params['theta'] > 0:
                    signal = -np.sign(z_score)  # Mean reversion signal
                else:
                    signal = 0
                
                rolling_signals.append(signal)
            
            # Align signals with price data
            signal_index = price_series.index[self.config.lookback_period:]
            signals.loc[signal_index, asset] = rolling_signals
        
        return signals.fillna(0)
    
    def calculate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert signals to position weights with risk management."""
        
        positions = signals.copy()
        
        # Apply position sizing and risk management
        for asset in positions.columns:
            asset_positions = positions[asset]
            
            # Scale positions by maximum position size
            positions[asset] = asset_positions * self.config.max_position_size
            
            # Apply stop-loss if specified
            if self.config.stop_loss is not None:
                # This would require price data to implement properly
                pass
        
        return positions

class MomentumStrategy(BaseStrategy):
    """
    Cross-Sectional Momentum Strategy with Machine Learning Enhancement
    
    Mathematical Foundation:
    ------------------------
    Traditional momentum: r_{i,t+1} = alpha + beta * r_{i,t-k:t} + epsilon_{i,t+1}
    
    ML Enhancement: Uses ensemble methods to combine multiple momentum signals
    and improve signal quality through feature engineering.
    """
    
    def __init__(self, config: StrategyConfig,
                 momentum_periods: List[int] = [1, 3, 6, 12],
                 use_ml_enhancement: bool = True):
        super().__init__(config)
        self.momentum_periods = momentum_periods
        self.use_ml_enhancement = use_ml_enhancement
        self.ml_model = None
        self.scaler = StandardScaler()
        
    def calculate_momentum_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate multiple momentum features."""
        
        features = pd.DataFrame(index=returns.index)
        
        for asset in returns.columns:
            asset_returns = returns[asset]
            
            # Traditional momentum (cumulative returns)
            for period in self.momentum_periods:
                period_name = f"{asset}_mom_{period}m"
                features[period_name] = asset_returns.rolling(window=period*22).apply(
                    lambda x: (1 + x).prod() - 1
                )
            
            # Risk-adjusted momentum
            for period in self.momentum_periods:
                period_name = f"{asset}_risk_adj_mom_{period}m"
                rolling_mean = asset_returns.rolling(window=period*22).mean()
                rolling_std = asset_returns.rolling(window=period*22).std()
                features[period_name] = rolling_mean / rolling_std
            
            # Technical indicators
            features[f"{asset}_rsi"] = self._calculate_rsi(asset_returns)
            features[f"{asset}_macd"] = self._calculate_macd(asset_returns)
            
        return features.dropna()
    
    def _calculate_rsi(self, returns: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, returns: pd.Series, 
                        fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        
        prices = (1 + returns).cumprod()
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        return macd - macd_signal
    
    def train_ml_model(self, features: pd.DataFrame, 
                      forward_returns: pd.DataFrame) -> None:
        """Train machine learning model for signal enhancement."""
        
        if not self.use_ml_enhancement:
            return
        
        # Prepare training data
        X_list = []
        y_list = []
        
        for asset in forward_returns.columns:
            asset_features = features[[col for col in features.columns if asset in col]]
            asset_forward_returns = forward_returns[asset]
            
            # Align data
            common_index = asset_features.index.intersection(asset_forward_returns.index)
            if len(common_index) > 100:  # Minimum samples for training
                X_asset = asset_features.loc[common_index]
                y_asset = asset_forward_returns.loc[common_index]
                
                # Create binary labels (top/bottom tercile)
                y_binary = pd.qcut(y_asset, q=3, labels=[0, 1, 2])
                
                X_list.append(X_asset)
                y_list.append(y_binary)
        
        if X_list:
            X_combined = pd.concat(X_list, axis=0)
            y_combined = pd.concat(y_list, axis=0)
            
            # Remove NaN values
            mask = ~(X_combined.isna().any(axis=1) | y_combined.isna())
            X_clean = X_combined[mask]
            y_clean = y_combined[mask]
            
            if len(X_clean) > 100:
                # Scale features
                X_scaled = self.scaler.fit_transform(X_clean)
                
                # Train ensemble model
                self.ml_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                
                self.ml_model.fit(X_scaled, y_clean)
                
                print(f"ML model trained on {len(X_clean)} samples")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals with optional ML enhancement."""
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Calculate momentum features
        features = self.calculate_momentum_features(returns)
        
        # Calculate forward returns for training (if ML enhancement is used)
        if self.use_ml_enhancement:
            forward_returns = returns.shift(-self.config.holding_period)
            
            # Train ML model
            self.train_ml_model(features, forward_returns)
        
        # Generate signals
        signals = pd.DataFrame(index=data.index, columns=data.columns)
        
        for i in range(len(features)):
            if i < self.config.lookback_period:
                continue
            
            current_features = features.iloc[i]
            
            # Cross-sectional ranking approach
            asset_scores = {}
            
            for asset in data.columns:
                # Get asset-specific features
                asset_feature_cols = [col for col in features.columns if asset in col]
                asset_features = current_features[asset_feature_cols]
                
                if not asset_features.isna().any():
                    if self.use_ml_enhancement and self.ml_model is not None:
                        # ML-enhanced score
                        X_scaled = self.scaler.transform(asset_features.values.reshape(1, -1))
                        ml_score = self.ml_model.predict(X_scaled)[0]
                        
                        # Combine with traditional momentum
                        traditional_score = asset_features.iloc[0]  # Use first momentum feature
                        asset_scores[asset] = 0.7 * ml_score + 0.3 * traditional_score
                    else:
                        # Traditional momentum score
                        asset_scores[asset] = asset_features.iloc[0]
            
            # Cross-sectional ranking
            if asset_scores:
                sorted_assets = sorted(asset_scores.items(), key=lambda x: x[1], reverse=True)
                n_assets = len(sorted_assets)
                n_long = n_assets // 3  # Top tercile
                n_short = n_assets // 3  # Bottom tercile
                
                # Generate signals
                current_signals = {}
                for j, (asset, score) in enumerate(sorted_assets):
                    if j < n_long:
                        current_signals[asset] = 1  # Long
                    elif j >= n_assets - n_short:
                        current_signals[asset] = -1  # Short
                    else:
                        current_signals[asset] = 0  # Neutral
                
                # Set signals
                for asset in data.columns:
                    signals.loc[features.index[i], asset] = current_signals.get(asset, 0)
        
        return signals.fillna(0)
    
    def calculate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert signals to position weights with risk management."""
        
        positions = pd.DataFrame(index=signals.index, columns=signals.columns)
        
        for i, (idx, row) in enumerate(signals.iterrows()):
            long_assets = row[row == 1].index.tolist()
            short_assets = row[row == -1].index.tolist()
            
            # Equal weight within long/short portfolios
            if long_assets:
                long_weight = self.config.max_position_size / len(long_assets)
                for asset in long_assets:
                    positions.loc[idx, asset] = long_weight
            
            if short_assets:
                short_weight = -self.config.max_position_size / len(short_assets)
                for asset in short_assets:
                    positions.loc[idx, asset] = short_weight
            
            # Fill remaining with zeros
            positions.loc[idx] = positions.loc[idx].fillna(0)
        
        return positions

class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical Arbitrage Strategy using Cointegration
    
    Mathematical Foundation:
    ------------------------
    Two assets are cointegrated if:
    P_{1,t} ~ I(1), P_{2,t} ~ I(1), but P_{1,t} - betaP_{2,t} ~ I(0)

    The strategy trades the spread when it deviates from equilibrium.
    """
    
    def __init__(self, config: StrategyConfig,
                 cointegration_lookback: int = 252,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.1):
        super().__init__(config)
        self.cointegration_lookback = cointegration_lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
    def test_cointegration(self, price1: pd.Series, price2: pd.Series) -> Dict[str, float]:
        """Test for cointegration between two price series."""
        
        # Ensure series are aligned
        common_index = price1.index.intersection(price2.index)
        p1 = price1.loc[common_index].dropna()
        p2 = price2.loc[common_index].dropna()
        
        if len(p1) < 50 or len(p2) < 50:
            return {'cointegrated': False, 'p_value': 1.0}
        
        # Step 1: OLS regression to get cointegration coefficient
        X = np.column_stack([np.ones(len(p2)), p2])
        y = p1.values
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            
            # Step 2: Test residuals for stationarity (ADF test)
            from statsmodels.tsa.stattools import adfuller
            
            adf_result = adfuller(residuals, maxlag=1, regression='c', autolag=None)
            p_value = adf_result[1]
            
            return {
                'cointegrated': p_value < 0.05,
                'p_value': p_value,
                'beta': beta[1],
                'alpha': beta[0],
                'residuals': pd.Series(residuals, index=common_index)
            }
            
        except:
            # Fallback if statsmodels not available
            # Simple stationarity proxy using variance ratio
            residuals_series = pd.Series(residuals, index=common_index)
            
            # Calculate variance ratio test statistic
            var_1 = residuals_series.rolling(10).var().mean()
            var_total = residuals_series.var()
            
            pseudo_p_value = min(1.0, abs(var_1 / var_total - 0.1))
            
            return {
                'cointegrated': pseudo_p_value < 0.1,
                'p_value': pseudo_p_value,
                'beta': beta[1],
                'alpha': beta[0],
                'residuals': residuals_series
            }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate pairs trading signals based on cointegration."""
        
        signals = pd.DataFrame(index=data.index, columns=data.columns, data=0.0)
        
        # Find all possible pairs
        assets = data.columns.tolist()
        pairs_data = {}
        
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                pair_name = f"{asset1}_{asset2}"
                
                # Test cointegration on rolling basis
                for k in range(self.cointegration_lookback, len(data)):
                    window_data = data.iloc[k-self.cointegration_lookback:k]
                    
                    # Test cointegration
                    coint_result = self.test_cointegration(
                        window_data[asset1], window_data[asset2]
                    )
                    
                    if coint_result['cointegrated']:
                        # Calculate current spread
                        current_p1 = data[asset1].iloc[k]
                        current_p2 = data[asset2].iloc[k]
                        
                        spread = current_p1 - coint_result['beta'] * current_p2 - coint_result['alpha']
                        
                        # Standardize spread
                        residuals = coint_result['residuals']
                        z_score = (spread - residuals.mean()) / residuals.std()
                        
                        # Generate signals
                        if abs(z_score) > self.entry_threshold:
                            # Mean reversion signal
                            signal_strength = min(1.0, abs(z_score) / self.entry_threshold)
                            
                            if z_score > 0:  # Spread too high
                                signals.loc[data.index[k], asset1] -= signal_strength * 0.5
                                signals.loc[data.index[k], asset2] += signal_strength * 0.5 * coint_result['beta']
                            else:  # Spread too low
                                signals.loc[data.index[k], asset1] += signal_strength * 0.5
                                signals.loc[data.index[k], asset2] -= signal_strength * 0.5 * coint_result['beta']
        
        return signals
    
    def calculate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert signals to position weights."""
        
        positions = signals.copy()
        
        # Normalize positions to respect maximum position size
        for idx, row in positions.iterrows():
            total_abs_exposure = row.abs().sum()
            if total_abs_exposure > 1.0:
                positions.loc[idx] = row / total_abs_exposure
            
            # Apply maximum position size constraint
            positions.loc[idx] = positions.loc[idx].clip(
                -self.config.max_position_size, 
                self.config.max_position_size
            )
        
        return positions

def create_strategy_comparison(price_data: pd.DataFrame,
                              benchmark: Optional[pd.Series] = None) -> Dict[str, any]:
    """
    Create comprehensive strategy comparison analysis.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Price data for assets
    benchmark : pd.Series, optional
        Benchmark returns for comparison
        
    Returns:
    --------
    Dict containing strategy comparison results
    """
    
    strategies = {}
    results = {}
    
    # Define strategy configurations
    configs = {
        'mean_reversion': StrategyConfig(
            name='Mean Reversion',
            lookback_period=126,
            holding_period=22,
            transaction_cost=0.001,
            max_position_size=0.2
        ),
        'momentum': StrategyConfig(
            name='Momentum',
            lookback_period=252,
            holding_period=22,
            transaction_cost=0.001,
            max_position_size=0.3
        ),
        'statistical_arbitrage': StrategyConfig(
            name='Statistical Arbitrage',
            lookback_period=252,
            holding_period=5,
            transaction_cost=0.0005,
            max_position_size=0.4
        )
    }
    
    # Initialize strategies
    strategies['mean_reversion'] = MeanReversionStrategy(configs['mean_reversion'])
    strategies['momentum'] = MomentumStrategy(configs['momentum'])
    strategies['statistical_arbitrage'] = StatisticalArbitrageStrategy(configs['statistical_arbitrage'])
    
    print("=== STRATEGY COMPARISON ANALYSIS ===")
    
    # Run backtests
    for name, strategy in strategies.items():
        print(f"\nBacktesting {name} strategy...")
        
        try:
            result = strategy.backtest(price_data, benchmark)
            results[name] = result
            
            perf = result['performance']
            print(f" {name}:")
            print(f"  Annual Return: {perf['annualized_return']:.2%}")
            print(f"  Volatility: {perf['volatility']:.2%}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {perf['max_drawdown']:.2%}")
            
        except Exception as e:
            print(f"Error in {name}: {e}")
    
    # Create performance comparison
    if results:
        performance_df = pd.DataFrame({
            name: result['performance'] 
            for name, result in results.items()
        }).T
        
        print(f"\n=== PERFORMANCE COMPARISON ===")
        print(performance_df.round(4))
        
        # Strategy ranking
        ranking_metrics = ['sharpe_ratio', 'calmar_ratio', 'information_ratio']
        rankings = {}
        
        for metric in ranking_metrics:
            if metric in performance_df.columns:
                ranked = performance_df[metric].sort_values(ascending=False)
                rankings[metric] = ranked.index.tolist()
        
        print(f"\n=== STRATEGY RANKINGS ===")
        for metric, ranking in rankings.items():
            print(f"{metric.replace('_', ' ').title()}: {ranking}")
    
    return {
        'strategies': strategies,
        'results': results,
        'performance_comparison': performance_df if results else None
    }

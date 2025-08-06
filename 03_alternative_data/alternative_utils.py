"""
Alternative Data Utility Functions

This module provides helper functions for processing and analyzing alternative data sources.

Author: Kevin J. Metzler
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit


def calculate_lead_lag_correlation(price_series: pd.Series, 
                                  feature_series: pd.Series,
                                  max_lag: int = 10,
                                  max_lead: int = 10) -> pd.DataFrame:
    """
    Calculate lead-lag correlations between a price series and a feature.
    
    Parameters:
    -----------
    price_series : pd.Series
        Price series data
    feature_series : pd.Series
        Feature series data to correlate with price
    max_lag : int
        Maximum lag periods (feature behind price)
    max_lead : int
        Maximum lead periods (feature ahead of price)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with lag periods and correlations
    """
    correlations = []
    
    # Calculate lagged correlations (feature behind price)
    for lag in range(0, max_lag + 1):
        lagged_feature = feature_series.shift(lag)
        corr = price_series.corr(lagged_feature)
        correlations.append((-lag, corr))
    
    # Calculate lead correlations (feature ahead of price)
    for lead in range(1, max_lead + 1):
        leading_feature = feature_series.shift(-lead)
        corr = price_series.corr(leading_feature)
        correlations.append((lead, corr))
    
    # Convert to DataFrame and sort by lag/lead
    corr_df = pd.DataFrame(correlations, columns=['lag_lead', 'correlation'])
    corr_df = corr_df.sort_values('lag_lead').reset_index(drop=True)
    
    return corr_df


def plot_lead_lag_correlation(corr_df: pd.DataFrame, title: str = None) -> None:
    """
    Plot lead-lag correlation results.
    
    Parameters:
    -----------
    corr_df : pd.DataFrame
        DataFrame with lag_lead and correlation columns
    title : str, optional
        Title for the plot
    """
    plt.figure(figsize=(12, 6))
    plt.bar(corr_df['lag_lead'], corr_df['correlation'])
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add vertical lines at significant correlation points
    max_corr_idx = corr_df['correlation'].abs().idxmax()
    max_corr_lag = corr_df.loc[max_corr_idx, 'lag_lead']
    max_corr_val = corr_df.loc[max_corr_idx, 'correlation']
    plt.axvline(x=max_corr_lag, color='g', linestyle='--', alpha=0.5)
    
    plt.xlabel('Lag (negative) / Lead (positive)')
    plt.ylabel('Correlation')
    
    if title:
        plt.title(title)
    else:
        plt.title('Lead-Lag Correlation Analysis')
        
    # Add annotation for maximum correlation
    plt.annotate(f'Max: {max_corr_val:.3f} at lag {max_corr_lag}',
                xy=(max_corr_lag, max_corr_val),
                xytext=(max_corr_lag + (1 if max_corr_lag < 0 else -1), 
                       max_corr_val + 0.05),
                arrowprops=dict(arrowstyle='->'))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Calculate mutual information between features and target variable.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
        
    Returns:
    --------
    pd.Series
        Mutual information scores for each feature
    """
    # Fill missing values
    X_filled = X.fillna(X.mean())
    y_filled = y.fillna(y.mean())
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X_filled, y_filled)
    
    # Create Series with feature names
    mi_series = pd.Series(mi_scores, index=X.columns)
    
    return mi_series.sort_values(ascending=False)


def evaluate_feature_predictive_power(data: pd.DataFrame, 
                                     feature_cols: List[str],
                                     target_col: str,
                                     prediction_horizon: int = 1,
                                     n_splits: int = 5) -> Dict:
    """
    Evaluate the predictive power of features using cross-validation.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with features and target
    feature_cols : List[str]
        List of feature column names
    target_col : str
        Target column name
    prediction_horizon : int
        Number of periods ahead to predict
    n_splits : int
        Number of cross-validation splits
        
    Returns:
    --------
    Dict
        Dictionary with evaluation results
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Prepare target with specified horizon
    target = data[target_col].shift(-prediction_horizon).dropna()
    
    # Align features with target
    aligned_data = data.loc[target.index]
    X = aligned_data[feature_cols].fillna(0)
    y = target
    
    # Initialize cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Initialize metrics
    mse_scores = []
    r2_scores = []
    feature_importances = []
    
    # Cross-validation loop
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        mse_scores.append(mse)
        r2_scores.append(r2)
        feature_importances.append(model.feature_importances_)
    
    # Calculate average feature importance
    mean_importance = np.mean(feature_importances, axis=0)
    feature_importance = pd.Series(mean_importance, index=feature_cols)
    
    return {
        'mse': np.mean(mse_scores),
        'r2': np.mean(r2_scores),
        'feature_importance': feature_importance.sort_values(ascending=False),
        'mse_std': np.std(mse_scores),
        'r2_std': np.std(r2_scores)
    }


def visualize_feature_importance(feature_importance: pd.Series, title: str = None) -> None:
    """
    Visualize feature importance.
    
    Parameters:
    -----------
    feature_importance : pd.Series
        Feature importance scores
    title : str, optional
        Title for the plot
    """
    plt.figure(figsize=(12, len(feature_importance) * 0.4 + 2))
    feature_importance.sort_values().plot(kind='barh')
    
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    if title:
        plt.title(title)
    else:
        plt.title('Feature Importance')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_information_coefficient(predicted: pd.Series, actual: pd.Series) -> float:
    """
    Calculate Information Coefficient (IC) - the correlation between predicted and actual values.
    
    Parameters:
    -----------
    predicted : pd.Series
        Predicted values
    actual : pd.Series
        Actual values
        
    Returns:
    --------
    float
        Information Coefficient
    """
    # Align series
    aligned = pd.DataFrame({'predicted': predicted, 'actual': actual})
    aligned = aligned.dropna()
    
    # Calculate Spearman rank correlation
    if len(aligned) < 2:
        return 0.0
    
    ic = aligned['predicted'].corr(aligned['actual'], method='spearman')
    
    return ic


def backtest_signal_accuracy(signals: pd.Series, 
                           returns: pd.Series,
                           lookahead_periods: int = 1) -> Dict:
    """
    Backtest the accuracy of signals for predicting future returns.
    
    Parameters:
    -----------
    signals : pd.Series
        Signal series (positive for bullish, negative for bearish)
    returns : pd.Series
        Return series
    lookahead_periods : int
        Number of periods to look ahead for returns
        
    Returns:
    --------
    Dict
        Accuracy metrics
    """
    # Align signals and returns
    aligned_data = pd.DataFrame({'signal': signals, 'return': returns})
    aligned_data = aligned_data.dropna()
    
    # Calculate future returns
    aligned_data['future_return'] = aligned_data['return'].shift(-lookahead_periods)
    aligned_data = aligned_data.dropna()
    
    # Calculate signal accuracy
    correct_positive = ((aligned_data['signal'] > 0) & (aligned_data['future_return'] > 0)).sum()
    correct_negative = ((aligned_data['signal'] < 0) & (aligned_data['future_return'] < 0)).sum()
    total_positive = (aligned_data['signal'] > 0).sum()
    total_negative = (aligned_data['signal'] < 0).sum()
    
    # Calculate metrics
    accuracy_positive = correct_positive / total_positive if total_positive > 0 else 0
    accuracy_negative = correct_negative / total_negative if total_negative > 0 else 0
    overall_accuracy = (correct_positive + correct_negative) / len(aligned_data)
    
    # Calculate average returns by signal
    avg_return_positive = aligned_data.loc[aligned_data['signal'] > 0, 'future_return'].mean()
    avg_return_negative = aligned_data.loc[aligned_data['signal'] < 0, 'future_return'].mean()
    avg_return_all = aligned_data['future_return'].mean()
    
    # Calculate information coefficient
    ic = calculate_information_coefficient(aligned_data['signal'], aligned_data['future_return'])
    
    return {
        'accuracy_positive': accuracy_positive,
        'accuracy_negative': accuracy_negative,
        'overall_accuracy': overall_accuracy,
        'avg_return_positive': avg_return_positive,
        'avg_return_negative': avg_return_negative,
        'avg_return_all': avg_return_all,
        'information_coefficient': ic,
        'signal_count': len(aligned_data),
        'positive_signals': total_positive,
        'negative_signals': total_negative
    }


def create_alternative_data_dashboard(data: pd.DataFrame, 
                                    price_col: str = 'Close',
                                    sentiment_cols: List[str] = None,
                                    volume_col: str = 'Volume',
                                    return_col: str = 'return') -> None:
    """
    Create a comprehensive dashboard visualizing alternative data alongside market data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with market and alternative data
    price_col : str
        Column name for price data
    sentiment_cols : List[str]
        Column names for sentiment features
    volume_col : str
        Column name for volume data
    return_col : str
        Column name for returns data
    """
    if sentiment_cols is None:
        sentiment_cols = [col for col in data.columns if 'sentiment' in col.lower() or 'compound' in col.lower()]
    
    # Create dashboard layout
    fig, axes = plt.subplots(4, 1, figsize=(15, 15), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Plot 1: Price
    data[price_col].plot(ax=axes[0], color='blue')
    axes[0].set_title('Price')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Price')
    
    # Plot 2: Volume
    if volume_col in data.columns:
        data[volume_col].plot(ax=axes[1], color='green', alpha=0.7)
        axes[1].set_title('Volume')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylabel('Volume')
    else:
        axes[1].set_visible(False)
    
    # Plot 3: Returns
    if return_col in data.columns:
        data[return_col].plot(ax=axes[2], color='purple')
        axes[2].set_title('Returns')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylabel('Return')
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    else:
        axes[2].set_visible(False)
    
    # Plot 4: Sentiment
    if sentiment_cols:
        for col in sentiment_cols:
            if col in data.columns:
                data[col].plot(ax=axes[3], alpha=0.7, label=col)
        
        axes[3].set_title('Sentiment Indicators')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylabel('Sentiment')
        axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[3].legend(loc='upper left')
    else:
        axes[3].set_visible(False)
    
    plt.tight_layout()
    plt.show()

"""
Data utilities for quantitative research portfolio.
Handles data fetching, cleaning, and preprocessing across all projects.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional, Union
import warnings
from datetime import datetime, timedelta
import sqlite3
import os

warnings.filterwarnings('ignore')

class DataManager:
    """
    Centralized data management for all quantitative research projects.
    Handles data fetching, caching, and preprocessing.
    """
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        self.cache_db = os.path.join(data_dir, "cache.db")
        self._ensure_data_dir()
        self._init_cache_db()
    
    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "alternative"), exist_ok=True)
    
    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        conn = sqlite3.connect(self.cache_db)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS price_cache (
                symbol TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                PRIMARY KEY (symbol, date)
            )
        ''')
        conn.commit()
        conn.close()
    
    def fetch_equity_data(
        self, 
        symbols: Union[str, List[str]], 
        start_date: str, 
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch equity price data with caching capability.
        
        Parameters:
        -----------
        symbols : str or list of str
            Stock symbols to fetch
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        use_cache : bool
            Whether to use cached data
            
        Returns:
        --------
        pd.DataFrame
            Multi-index DataFrame with prices
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        all_data = {}
        
        for symbol in symbols:
            if use_cache:
                cached_data = self._get_cached_data(symbol, start_date, end_date)
                if cached_data is not None and len(cached_data) > 0:
                    all_data[symbol] = cached_data
                    continue
            
            # Fetch from Yahoo Finance
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if len(data) > 0:
                    # Cache the data
                    if use_cache:
                        self._cache_data(symbol, data)
                    all_data[symbol] = data
                else:
                    print(f"Warning: No data found for {symbol}")
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all symbols into multi-index DataFrame
        combined_data = pd.concat(all_data, axis=1)
        combined_data.columns.names = ['Symbol', 'Price_Type']
        
        return combined_data
    
    def _get_cached_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data for a symbol."""
        conn = sqlite3.connect(self.cache_db)
        query = '''
            SELECT * FROM price_cache 
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date
        '''
        
        try:
            df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            if len(df) > 0:
                df['Date'] = pd.to_datetime(df['date'])
                df.set_index('Date', inplace=True)
                df.drop(['symbol', 'date'], axis=1, inplace=True)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                return df
        except Exception as e:
            print(f"Error reading cached data: {e}")
        finally:
            conn.close()
        
        return None
    
    def _cache_data(self, symbol: str, data: pd.DataFrame):
        """Cache data to SQLite database."""
        conn = sqlite3.connect(self.cache_db)
        
        cache_data = data.copy()
        cache_data['symbol'] = symbol
        cache_data['date'] = cache_data.index.date
        cache_data = cache_data.rename(columns={
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        })
        
        # Insert or replace data
        cache_data[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']].to_sql(
            'price_cache', conn, if_exists='replace', index=False, method='multi'
        )
        
        conn.close()

    def calculate_returns(
        self, 
        prices: pd.DataFrame, 
        method: str = 'simple',
        period: int = 1
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        method : str
            'simple' or 'log' returns
        period : int
            Period for return calculation
            
        Returns:
        --------
        pd.DataFrame
            Returns data
        """
        if method == 'simple':
            returns = prices.pct_change(periods=period)
        elif method == 'log':
            returns = np.log(prices / prices.shift(period))
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        return returns.dropna()
    
    def clean_data(self, data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Clean financial data by handling missing values and outliers.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw data to clean
        method : str
            Method for handling missing values
            
        Returns:
        --------
        pd.DataFrame
            Cleaned data
        """
        cleaned_data = data.copy()
        
        # Handle missing values
        if method == 'forward_fill':
            cleaned_data = cleaned_data.fillna(method='ffill')
        elif method == 'interpolate':
            cleaned_data = cleaned_data.interpolate()
        elif method == 'drop':
            cleaned_data = cleaned_data.dropna()
        
        # Remove extreme outliers (beyond 5 standard deviations)
        if len(cleaned_data) > 0:
            for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                mean = cleaned_data[col].mean()
                std = cleaned_data[col].std()
                lower_bound = mean - 5 * std
                upper_bound = mean + 5 * std
                cleaned_data[col] = cleaned_data[col].clip(lower_bound, upper_bound)
        
        return cleaned_data
    
    def get_benchmark_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get common benchmark data (S&P 500, Treasury rates, VIX)."""
        benchmarks = {
            'SPY': 'S&P 500 ETF',
            '^VIX': 'VIX',
            '^TNX': '10-Year Treasury',
            '^IRX': '3-Month Treasury'
        }
        
        benchmark_data = {}
        for symbol, name in benchmarks.items():
            try:
                data = self.fetch_equity_data(symbol, start_date, end_date)
                if len(data) > 0:
                    benchmark_data[name] = data[symbol]['Close']
            except Exception as e:
                print(f"Warning: Could not fetch {name}: {e}")
        
        return pd.DataFrame(benchmark_data)


def load_alternative_data_sources():
    """
    Configuration for alternative data sources.
    Returns dictionary of data source configurations.
    """
    return {
        'economic_indicators': {
            'fred_series': [
                'GDP',       # Gross Domestic Product
                'UNRATE',    # Unemployment Rate
                'CPIAUCSL',  # Consumer Price Index
                'FEDFUNDS',  # Federal Funds Rate
                'DGS10',     # 10-Year Treasury Rate
                'DEXUSEU',   # USD/EUR Exchange Rate
            ]
        },
        'sentiment_sources': {
            'news_apis': ['newsapi', 'alpha_vantage_news'],
            'social_media': ['twitter', 'reddit'], #at some point might have to switch twitter to X
            'analyst_reports': ['refinitiv', 'bloomberg']
        },
        'high_frequency': {
            'exchanges': ['nasdaq', 'nyse'],
            'data_types': ['level1', 'level2', 'trades', 'quotes']
        }
    }


def validate_data_quality(data: pd.DataFrame, symbol: str = None) -> Dict[str, any]:
    """
    Comprehensive data quality assessment.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to validate
    symbol : str, optional
        Symbol name for reporting
        
    Returns:
    --------
    Dict
        Quality metrics and issues
    """
    quality_report = {
        'symbol': symbol,
        'total_observations': len(data),
        'date_range': (data.index.min(), data.index.max()) if len(data) > 0 else None,
        'missing_values': data.isnull().sum().to_dict(),
        'duplicate_dates': data.index.duplicated().sum(),
        'data_types': data.dtypes.to_dict(),
        'issues': []
    }
    
    # Check for issues
    if quality_report['missing_values'] and sum(quality_report['missing_values'].values()) > 0:
        quality_report['issues'].append('Missing values detected')
    
    if quality_report['duplicate_dates'] > 0:
        quality_report['issues'].append('Duplicate dates found')
    
    # Check for price inconsistencies (if OHLC data)
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        invalid_prices = (
            (data['High'] < data['Low']) |
            (data['High'] < data['Open']) |
            (data['High'] < data['Close']) |
            (data['Low'] > data['Open']) |
            (data['Low'] > data['Close'])
        ).sum()
        
        if invalid_prices > 0:
            quality_report['issues'].append(f'{invalid_prices} invalid OHLC relationships')
    
    # Check for extreme returns (if price data available)
    if 'Close' in data.columns:
        returns = data['Close'].pct_change().dropna()
        extreme_returns = (returns.abs() > 0.5).sum()  # >50% daily returns
        if extreme_returns > 0:
            quality_report['issues'].append(f'{extreme_returns} extreme return observations')
    
    quality_report['quality_score'] = 100 - len(quality_report['issues']) * 10
    
    return quality_report

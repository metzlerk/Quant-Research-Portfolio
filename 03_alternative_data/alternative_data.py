"""
Alternative Data Integration for Quantitative Strategies

This module implements methods to extract, process, and integrate alternative
data sources with traditional financial models, including:
- News sentiment analysis for market prediction
- Social media data processing for investor sentiment
- Satellite imagery analysis for economic activity
- Web traffic data for corporate performance estimation

Author: Kevin J. Metzler
Mathematical Foundation: Based on natural language processing, computer vision,
and time series alignment techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import os
import json
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, NMF
import warnings
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class AlternativeDataConfig:
    """Configuration for alternative data processing."""
    data_dir: str
    cache_data: bool = True
    max_articles_per_query: int = 100
    sentiment_window: int = 30  # Days
    api_request_delay: float = 1.0  # Seconds
    

class BaseAlternativeData:
    """Base class for alternative data sources."""
    
    def __init__(self, config: AlternativeDataConfig):
        """
        Initialize alternative data processor.
        
        Parameters:
        -----------
        config : AlternativeDataConfig
            Configuration object
        """
        self.config = config
        self.data_cache = {}
        
        # Create data directory if it doesn't exist
        if not os.path.exists(config.data_dir):
            os.makedirs(config.data_dir)
            
    def _save_to_cache(self, key: str, data: any) -> None:
        """Save data to cache."""
        if self.config.cache_data:
            cache_path = os.path.join(self.config.data_dir, f"{key}.json")
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
                
            self.data_cache[key] = data
    
    def _load_from_cache(self, key: str) -> Optional[any]:
        """Load data from cache."""
        if key in self.data_cache:
            return self.data_cache[key]
            
        if self.config.cache_data:
            cache_path = os.path.join(self.config.data_dir, f"{key}.json")
            
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    self.data_cache[key] = data
                    return data
                    
        return None
        
    def _get_cache_key(self, source: str, query: str, start_date: str, end_date: str) -> str:
        """Generate cache key for data retrieval."""
        sanitized_query = query.replace(' ', '_').replace('/', '_').lower()
        return f"{source}_{sanitized_query}_{start_date}_{end_date}"
    
    def align_with_market_data(self, alternative_data: pd.DataFrame, 
                              market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Align alternative data with market data.
        
        Parameters:
        -----------
        alternative_data : pd.DataFrame
            DataFrame with alternative data and DatetimeIndex
        market_data : pd.DataFrame
            DataFrame with market data and DatetimeIndex
            
        Returns:
        --------
        pd.DataFrame
            Aligned data with alternative data features
        """
        # Convert to daily frequency if necessary
        if alternative_data.index.inferred_freq != 'D':
            alternative_data = alternative_data.resample('D').ffill()
            
        # Align with market data
        aligned_data = pd.DataFrame(index=market_data.index)
        
        # Forward fill alternative data
        for column in alternative_data.columns:
            aligned_data[column] = alternative_data[column].reindex(
                aligned_data.index, method='ffill'
            )
        
        # Handle cases with NaN values at the beginning
        aligned_data = aligned_data.fillna(method='bfill')
        
        return aligned_data
        
class NewsSentimentAnalyzer(BaseAlternativeData):
    """
    News sentiment analysis for financial markets.
    
    Mathematical Foundation:
    ------------------------
    Sentiment analysis using lexicon-based approaches and NLP techniques.
    Topic modeling using Latent Dirichlet Allocation (LDA) and Non-negative 
    Matrix Factorization (NMF).
    """
    
    def __init__(self, config: AlternativeDataConfig):
        """Initialize the news sentiment analyzer."""
        super().__init__(config)
        
        # Initialize NLTK resources
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
            nltk.download('punkt')
            
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
    def fetch_news(self, query: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Fetch news articles from NewsAPI.
        
        Parameters:
        -----------
        query : str
            Search query (e.g., company name or topic)
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        List[Dict]
            List of news articles
        """
        cache_key = self._get_cache_key("news", query, start_date, end_date)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        if not self.news_api_key:
            print("Warning: NEWS_API_KEY environment variable not set")
            return []
            
        # Format dates for API
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_articles = []
        current_date = start
        
        # NewsAPI has a limit of 100 articles per request, so we paginate by date
        while current_date <= end:
            next_date = current_date + timedelta(days=7)  # Get weekly chunks
            
            if next_date > end:
                next_date = end
                
            from_date = current_date.strftime("%Y-%m-%d")
            to_date = next_date.strftime("%Y-%m-%d")
            
            url = f"https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "to": to_date,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "apiKey": self.news_api_key
            }
            
            try:
                response = requests.get(url, params=params)
                data = response.json()
                
                if data.get("status") == "ok":
                    articles = data.get("articles", [])
                    for article in articles:
                        article['publishedAt'] = article.get('publishedAt', from_date)
                    all_articles.extend(articles)
                    
                import time
                time.sleep(self.config.api_request_delay)
                    
            except Exception as e:
                print(f"Error fetching news: {e}")
                
            current_date = next_date + timedelta(days=1)
            
        # Cache the results
        self._save_to_cache(cache_key, all_articles)
        
        return all_articles
    
    def analyze_sentiment(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Analyze sentiment from news articles.
        
        Parameters:
        -----------
        articles : List[Dict]
            List of news articles with 'title', 'description', and 'publishedAt'
            
        Returns:
        --------
        pd.DataFrame
            Daily sentiment scores with DatetimeIndex
        """
        if not articles:
            return pd.DataFrame()
            
        # Extract text and dates
        dates = []
        compound_scores = []
        positive_scores = []
        negative_scores = []
        
        for article in articles:
            # Extract text
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            
            full_text = f"{title} {description} {content}"
            
            # Get published date
            pub_date_str = article.get('publishedAt', '')
            try:
                pub_date = datetime.strptime(pub_date_str[:10], "%Y-%m-%d")
            except (ValueError, TypeError):
                # Skip articles with invalid dates
                continue
                
            # Analyze sentiment
            sentiment = self.sentiment_analyzer.polarity_scores(full_text)
            
            dates.append(pub_date)
            compound_scores.append(sentiment['compound'])
            positive_scores.append(sentiment['pos'])
            negative_scores.append(sentiment['neg'])
            
        # Create DataFrame
        sentiment_df = pd.DataFrame({
            'compound': compound_scores,
            'positive': positive_scores,
            'negative': negative_scores
        }, index=dates)
        
        # Aggregate by date (average sentiment)
        daily_sentiment = sentiment_df.groupby(sentiment_df.index.date).mean()
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        
        # Add rolling averages
        window = self.config.sentiment_window
        daily_sentiment[f'compound_{window}d_avg'] = daily_sentiment['compound'].rolling(window).mean()
        daily_sentiment[f'positive_{window}d_avg'] = daily_sentiment['positive'].rolling(window).mean()
        daily_sentiment[f'negative_{window}d_avg'] = daily_sentiment['negative'].rolling(window).mean()
        daily_sentiment['sentiment_momentum'] = daily_sentiment[f'compound_{window}d_avg'].diff()
        
        return daily_sentiment
    
    def extract_topics(self, articles: List[Dict], n_topics: int = 5) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract topics from news articles using Non-negative Matrix Factorization.
        
        Parameters:
        -----------
        articles : List[Dict]
            List of news articles
        n_topics : int
            Number of topics to extract
            
        Returns:
        --------
        Tuple[pd.DataFrame, List[str]]
            DataFrame with topic strengths per day and list of top words per topic
        """
        if not articles or len(articles) < 10:  # Need sufficient articles for topic modeling
            return pd.DataFrame(), []
            
        # Extract text and dates
        texts = []
        dates = []
        
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            
            full_text = f"{title} {description} {content}"
            
            # Skip empty articles
            if not full_text.strip():
                continue
                
            # Get published date
            pub_date_str = article.get('publishedAt', '')
            try:
                pub_date = datetime.strptime(pub_date_str[:10], "%Y-%m-%d")
            except (ValueError, TypeError):
                continue
                
            texts.append(full_text)
            dates.append(pub_date)
            
        if not texts:
            return pd.DataFrame(), []
            
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', 
                                    min_df=2, max_df=0.85)
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Apply Non-negative Matrix Factorization
        nmf = NMF(n_components=n_topics, random_state=42)
        topic_strengths = nmf.fit_transform(tfidf_matrix)
        
        # Get top words for each topic
        components = nmf.components_
        topic_words = []
        
        for topic_idx, topic in enumerate(components):
            top_word_indices = topic.argsort()[-10:][::-1]  # Top 10 words
            top_words = [feature_names[i] for i in top_word_indices]
            topic_words.append(top_words)
            
        # Create topic strength DataFrame
        topic_df = pd.DataFrame(
            topic_strengths, 
            columns=[f"topic_{i}" for i in range(n_topics)]
        )
        topic_df['date'] = dates
        
        # Aggregate by date
        daily_topics = topic_df.groupby(topic_df['date'].dt.date).mean()
        daily_topics.index = pd.to_datetime(daily_topics.index)
        
        return daily_topics, topic_words
        
    def integrate_with_market_data(self, ticker: str, market_data: pd.DataFrame,
                                  start_date: str, end_date: str) -> pd.DataFrame:
        """
        Integrate news sentiment with market data.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        market_data : pd.DataFrame
            Market data with DatetimeIndex
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.DataFrame
            Market data with sentiment features
        """
        # Fetch news and analyze sentiment
        articles = self.fetch_news(ticker, start_date, end_date)
        sentiment_data = self.analyze_sentiment(articles)
        
        if sentiment_data.empty:
            print(f"No sentiment data available for {ticker}")
            return market_data
            
        # Extract topics
        topic_data, topic_words = self.extract_topics(articles)
        
        # Combine sentiment and topic data
        alternative_data = sentiment_data
        
        if not topic_data.empty:
            topic_data = topic_data.reindex(sentiment_data.index, method='ffill')
            alternative_data = pd.concat([sentiment_data, topic_data], axis=1)
            
        # Align with market data
        aligned_data = self.align_with_market_data(alternative_data, market_data)
        
        # Combine with market data
        result = pd.concat([market_data, aligned_data], axis=1)
        
        return result
        

class SocialMediaSentiment(BaseAlternativeData):
    """
    Social media sentiment analysis for financial markets.
    
    Mathematical Foundation:
    ------------------------
    Social media data analysis using NLP techniques, time series alignment,
    and sentiment aggregation methods.
    """
    
    def __init__(self, config: AlternativeDataConfig):
        """Initialize the social media sentiment analyzer."""
        super().__init__(config)
        self.twitter_api_key = os.getenv("TWITTER_API_KEY")
        self.twitter_api_secret = os.getenv("TWITTER_API_SECRET")
        
        # Initialize NLTK resources if not already done
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
            nltk.download('punkt')
            
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def fetch_tweets(self, query: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Fetch tweets using Twitter API.
        
        Parameters:
        -----------
        query : str
            Search query (e.g., $AAPL for Apple stock)
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        List[Dict]
            List of tweets
        """
        cache_key = self._get_cache_key("twitter", query, start_date, end_date)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        if not self.twitter_api_key or not self.twitter_api_secret:
            print("Warning: Twitter API credentials not set")
            return []
            
        try:
            import tweepy
            
            # Authenticate with Twitter API
            auth = tweepy.OAuthHandler(self.twitter_api_key, self.twitter_api_secret)
            api = tweepy.API(auth)
            
            # Format dates
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Search tweets
            tweets = []
            for tweet in tweepy.Cursor(api.search_tweets, q=query, 
                                     lang="en", 
                                     since=start_date,
                                     until=end_date,
                                     tweet_mode="extended").items(self.config.max_articles_per_query):
                                     
                tweet_data = {
                    'id': tweet.id_str,
                    'text': tweet.full_text,
                    'created_at': tweet.created_at.strftime("%Y-%m-%d"),
                    'user': tweet.user.screen_name,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count
                }
                
                tweets.append(tweet_data)
                
            # Cache results
            self._save_to_cache(cache_key, tweets)
            
            return tweets
            
        except ImportError:
            print("Warning: tweepy package not installed")
            return []
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            return []
    
    def analyze_social_sentiment(self, tweets: List[Dict]) -> pd.DataFrame:
        """
        Analyze sentiment from tweets.
        
        Parameters:
        -----------
        tweets : List[Dict]
            List of tweets with 'text' and 'created_at'
            
        Returns:
        --------
        pd.DataFrame
            Daily sentiment scores with DatetimeIndex
        """
        if not tweets:
            return pd.DataFrame()
            
        # Extract text and dates
        dates = []
        compound_scores = []
        positive_scores = []
        negative_scores = []
        engagement_scores = []
        
        for tweet in tweets:
            # Extract text
            text = tweet.get('text', '')
            
            # Get published date
            date_str = tweet.get('created_at', '')
            try:
                tweet_date = datetime.strptime(date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                continue
                
            # Calculate engagement
            retweet_count = tweet.get('retweet_count', 0)
            favorite_count = tweet.get('favorite_count', 0)
            engagement = retweet_count + favorite_count
                
            # Analyze sentiment
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            
            dates.append(tweet_date)
            compound_scores.append(sentiment['compound'])
            positive_scores.append(sentiment['pos'])
            negative_scores.append(sentiment['neg'])
            engagement_scores.append(engagement)
            
        # Create DataFrame
        sentiment_df = pd.DataFrame({
            'compound': compound_scores,
            'positive': positive_scores,
            'negative': negative_scores,
            'engagement': engagement_scores
        }, index=dates)
        
        # Aggregate by date (weighted by engagement)
        daily_sentiment = sentiment_df.groupby(sentiment_df.index.date).agg({
            'compound': 'mean',
            'positive': 'mean',
            'negative': 'mean',
            'engagement': 'sum'
        })
        
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        
        # Add rolling metrics
        window = self.config.sentiment_window
        daily_sentiment[f'compound_{window}d_avg'] = daily_sentiment['compound'].rolling(window).mean()
        daily_sentiment[f'positive_{window}d_avg'] = daily_sentiment['positive'].rolling(window).mean()
        daily_sentiment[f'negative_{window}d_avg'] = daily_sentiment['negative'].rolling(window).mean()
        daily_sentiment[f'engagement_{window}d_sum'] = daily_sentiment['engagement'].rolling(window).sum()
        daily_sentiment['sentiment_momentum'] = daily_sentiment[f'compound_{window}d_avg'].diff()
        
        return daily_sentiment
        
    def integrate_with_market_data(self, ticker: str, market_data: pd.DataFrame,
                                  start_date: str, end_date: str) -> pd.DataFrame:
        """
        Integrate social media sentiment with market data.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (will be prefixed with $ for search)
        market_data : pd.DataFrame
            Market data with DatetimeIndex
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.DataFrame
            Market data with sentiment features
        """
        # Format search query (include cashtag)
        search_query = f"${ticker}"
        
        # Fetch tweets and analyze sentiment
        tweets = self.fetch_tweets(search_query, start_date, end_date)
        sentiment_data = self.analyze_social_sentiment(tweets)
        
        if sentiment_data.empty:
            print(f"No social media data available for {ticker}")
            return market_data
        
        # Align with market data
        aligned_data = self.align_with_market_data(sentiment_data, market_data)
        
        # Combine with market data
        result = pd.concat([market_data, aligned_data], axis=1)
        
        return result

        
class AlternativeDataIntegration:
    """
    Main class for integrating multiple alternative data sources.
    """
    
    def __init__(self, data_dir: str = "../data/alternative"):
        """
        Initialize alternative data integration.
        
        Parameters:
        -----------
        data_dir : str
            Directory to store cached data
        """
        self.config = AlternativeDataConfig(data_dir=data_dir)
        self.news_analyzer = NewsSentimentAnalyzer(self.config)
        self.social_analyzer = SocialMediaSentiment(self.config)
        
    def integrate_all_sources(self, ticker: str, market_data: pd.DataFrame,
                             start_date: str, end_date: str) -> pd.DataFrame:
        """
        Integrate all alternative data sources with market data.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        market_data : pd.DataFrame
            Market data with DatetimeIndex
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.DataFrame
            Market data enriched with alternative data features
        """
        # Step 1: Add news sentiment
        print(f"Integrating news sentiment for {ticker}...")
        result = self.news_analyzer.integrate_with_market_data(
            ticker, market_data, start_date, end_date
        )
        
        # Step 2: Add social media sentiment
        print(f"Integrating social media sentiment for {ticker}...")
        result = self.social_analyzer.integrate_with_market_data(
            ticker, result, start_date, end_date
        )
        
        # Additional alternative data sources can be added here
        
        return result
        
    def create_feature_importance_analysis(self, data: pd.DataFrame, 
                                         target_col: str = 'return') -> Dict:
        """
        Analyze feature importance of alternative data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with market and alternative features
        target_col : str
            Target column for prediction
            
        Returns:
        --------
        Dict
            Feature importance analysis
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np
        
        # Ensure target column exists
        if target_col not in data.columns:
            if 'close' in data.columns:
                # Calculate returns if close prices are available
                data[target_col] = data['close'].pct_change()
            else:
                raise ValueError(f"Target column {target_col} not found in data")
                
        # Prepare data
        feature_cols = [col for col in data.columns 
                      if col != target_col and 'date' not in col.lower()]
        
        X = data[feature_cols].fillna(method='ffill').fillna(0)
        y = data[target_col].fillna(0)
        
        # Remove initial NaN values
        valid_idx = ~np.isnan(y)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        importances = []
        scores = []
        
        # Train models and get feature importance
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            scores.append(score)
            
            importances.append(model.feature_importances_)
            
        # Average importance across folds
        mean_importance = np.mean(importances, axis=0)
        
        # Create result dictionary
        importance_dict = {
            'feature_names': feature_cols,
            'importance': mean_importance.tolist(),
            'r2_score': np.mean(scores),
            'top_features': [feature_cols[i] for i in mean_importance.argsort()[-10:][::-1]]
        }
        
        return importance_dict
        
    def visualize_sentiment_impact(self, data: pd.DataFrame, ticker: str,
                                 sentiment_col: str = 'compound_30d_avg',
                                 price_col: str = 'close') -> None:
        """
        Visualize impact of sentiment on price movements.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with market and sentiment features
        ticker : str
            Stock ticker symbol
        sentiment_col : str
            Sentiment column to visualize
        price_col : str
            Price column to visualize
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if sentiment_col not in data.columns or price_col not in data.columns:
            print(f"Required columns not found: {sentiment_col}, {price_col}")
            return
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot price
        ax1.plot(data.index, data[price_col], color='blue')
        ax1.set_title(f"{ticker} Price")
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        # Plot sentiment
        ax2.plot(data.index, data[sentiment_col], color='green')
        ax2.set_title(f"Sentiment ({sentiment_col})")
        ax2.set_ylabel('Sentiment Score')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add correlation coefficient
        corr = data[[price_col, sentiment_col]].corr().iloc[0, 1]
        fig.suptitle(f"{ticker} Price vs. Sentiment (Correlation: {corr:.3f})", fontsize=16)
        
        plt.tight_layout()
        plt.show()
        
        # Lag analysis
        print("\n=== Sentiment Lag Analysis ===")
        max_lag = 10
        correlations = []
        
        for lag in range(max_lag + 1):
            lagged_sentiment = data[sentiment_col].shift(lag)
            lag_corr = data[price_col].corr(lagged_sentiment)
            correlations.append((lag, lag_corr))
            print(f"Lag {lag} days: {lag_corr:.4f}")
            
        # Find optimal lag
        optimal_lag = max(correlations, key=lambda x: abs(x[1]))
        print(f"\nOptimal sentiment lag: {optimal_lag[0]} days (correlation: {optimal_lag[1]:.4f})")

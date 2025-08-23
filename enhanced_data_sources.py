#!/usr/bin/env python3
"""
Enhanced Data Sources Integration
Provides integration with multiple financial data providers including
Alpha Vantage, Polygon.io, Finnhub, FRED, and alternative data sources.

Author: olaitanojo
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import json
import time
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    api_key: str = None
    base_url: str = None
    rate_limit: int = 60  # requests per minute
    enabled: bool = True

class EnhancedDataProvider:
    """Enhanced data provider with multiple source integration"""
    
    def __init__(self, config: Dict[str, DataSource]):
        self.sources = config
        self.session = None
        self.rate_limiters = {name: [] for name in config.keys()}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self, source_name: str) -> bool:
        """Check if we're within rate limits for a source"""
        now = time.time()
        source_requests = self.rate_limiters[source_name]
        
        # Remove requests older than 1 minute
        self.rate_limiters[source_name] = [req_time for req_time in source_requests 
                                          if now - req_time < 60]
        
        # Check if we can make another request
        if len(self.rate_limiters[source_name]) < self.sources[source_name].rate_limit:
            self.rate_limiters[source_name].append(now)
            return True
        
        return False
    
    async def fetch_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch stock data with fallback sources"""
        
        # Primary: yfinance (free, reliable)
        try:
            logger.info(f"Fetching {symbol} data from yfinance...")
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if not data.empty:
                data['source'] = 'yfinance'
                return data
        except Exception as e:
            logger.warning(f"yfinance failed for {symbol}: {e}")
        
        # Fallback: Alpha Vantage
        if 'alpha_vantage' in self.sources and self.sources['alpha_vantage'].enabled:
            try:
                data = await self._fetch_alpha_vantage_data(symbol, period)
                if data is not None and not data.empty:
                    data['source'] = 'alpha_vantage'
                    return data
            except Exception as e:
                logger.warning(f"Alpha Vantage failed for {symbol}: {e}")
        
        # Fallback: Polygon.io
        if 'polygon' in self.sources and self.sources['polygon'].enabled:
            try:
                data = await self._fetch_polygon_data(symbol, period)
                if data is not None and not data.empty:
                    data['source'] = 'polygon'
                    return data
            except Exception as e:
                logger.warning(f"Polygon failed for {symbol}: {e}")
        
        logger.error(f"All data sources failed for {symbol}")
        return pd.DataFrame()
    
    async def _fetch_alpha_vantage_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from Alpha Vantage"""
        
        if not self._check_rate_limit('alpha_vantage'):
            logger.warning("Alpha Vantage rate limit exceeded")
            return None
        
        source = self.sources['alpha_vantage']
        if not source.api_key:
            logger.warning("Alpha Vantage API key not provided")
            return None
        
        # Determine function based on period
        if period in ['1d', '5d']:
            function = 'TIME_SERIES_INTRADAY'
            interval = '15min'
        else:
            function = 'TIME_SERIES_DAILY_ADJUSTED'
            interval = None
        
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': source.api_key,
            'datatype': 'json'
        }
        
        if interval:
            params['interval'] = interval
        
        url = f"{source.base_url or 'https://www.alphavantage.co/query'}"
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            
            if 'Error Message' in data:
                raise Exception(f"Alpha Vantage error: {data['Error Message']}")
            
            # Parse response based on function
            if function == 'TIME_SERIES_INTRADAY':
                time_series_key = f'Time Series ({interval})'
            else:
                time_series_key = 'Time Series (Daily)'
            
            if time_series_key not in data:
                logger.warning(f"Unexpected Alpha Vantage response format: {list(data.keys())}")
                return None
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in data[time_series_key].items():
                row = {
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values.get('1. open', 0)),
                    'High': float(values.get('2. high', 0)),
                    'Low': float(values.get('3. low', 0)),
                    'Close': float(values.get('4. close', 0)),
                    'Volume': int(float(values.get('5. volume', 0)))
                }
                
                # Adjusted close for daily data
                if '5. adjusted close' in values:
                    row['Adj Close'] = float(values['5. adjusted close'])
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            return df
    
    async def _fetch_polygon_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from Polygon.io"""
        
        if not self._check_rate_limit('polygon'):
            logger.warning("Polygon rate limit exceeded")
            return None
        
        source = self.sources['polygon']
        if not source.api_key:
            logger.warning("Polygon API key not provided")
            return None
        
        # Calculate date range
        end_date = datetime.now()
        if period == '1d':
            start_date = end_date - timedelta(days=1)
            multiplier, timespan = 1, 'minute'
        elif period == '5d':
            start_date = end_date - timedelta(days=5)
            multiplier, timespan = 5, 'minute'
        elif period == '1mo':
            start_date = end_date - timedelta(days=30)
            multiplier, timespan = 1, 'day'
        else:  # Default to 1 year
            start_date = end_date - timedelta(days=365)
            multiplier, timespan = 1, 'day'
        
        url = f"{source.base_url or 'https://api.polygon.io'}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'apikey': source.api_key
        }
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            
            if data.get('status') != 'OK':
                raise Exception(f"Polygon error: {data.get('error', 'Unknown error')}")
            
            if 'results' not in data or not data['results']:
                logger.warning("No data returned from Polygon")
                return None
            
            # Convert to DataFrame
            df_data = []
            for bar in data['results']:
                row = {
                    'Date': pd.to_datetime(bar['t'], unit='ms'),
                    'Open': bar['o'],
                    'High': bar['h'],
                    'Low': bar['l'],
                    'Close': bar['c'],
                    'Volume': bar['v']
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            
            return df
    
    async def fetch_economic_data(self, indicator: str, start_date: str = None) -> pd.DataFrame:
        """Fetch economic indicators from FRED"""
        
        # For demo purposes, simulate economic data
        # In production, you would use the FRED API
        logger.info(f"Fetching economic indicator: {indicator}")
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        # Generate synthetic economic data
        dates = pd.date_range(start_date, datetime.now(), freq='D')
        
        if 'GDP' in indicator.upper():
            # GDP growth simulation
            base_value = 2.1
            noise = np.random.normal(0, 0.3, len(dates))
            seasonal = np.sin(np.arange(len(dates))/365*2*np.pi) * 0.5
            values = base_value + noise + seasonal
        elif 'UNEMPLOYMENT' in indicator.upper() or 'UNRATE' in indicator.upper():
            # Unemployment rate simulation
            base_value = 4.2
            noise = np.random.normal(0, 0.4, len(dates))
            trend = -0.001 * np.arange(len(dates))  # Slight downward trend
            values = base_value + noise + trend
        elif 'INFLATION' in indicator.upper() or 'CPI' in indicator.upper():
            # Inflation rate simulation
            base_value = 2.8
            noise = np.random.normal(0, 0.3, len(dates))
            cycle = np.sin(np.arange(len(dates))/90*2*np.pi) * 0.4  # Quarterly cycle
            values = base_value + noise + cycle
        elif 'INTEREST' in indicator.upper() or 'FEDFUNDS' in indicator.upper():
            # Interest rate simulation
            base_value = 2.5
            noise = np.random.normal(0, 0.1, len(dates))
            values = base_value + np.cumsum(noise) * 0.01
        else:
            # Generic economic indicator
            base_value = 100
            noise = np.random.normal(0, 5, len(dates))
            trend = 0.01 * np.arange(len(dates))  # Slight upward trend
            values = base_value + noise + trend
        
        df = pd.DataFrame({
            'Date': dates,
            indicator: values
        })
        df.set_index('Date', inplace=True)
        
        return df
    
    async def fetch_news_sentiment(self, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """Fetch news sentiment data"""
        
        # For demo purposes, simulate news sentiment
        # In production, you would integrate with news APIs like NewsAPI, Finnhub, etc.
        logger.info(f"Fetching news sentiment for {symbol}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Generate synthetic sentiment data
        np.random.seed(hash(symbol) % 2**32)  # Consistent randomness per symbol
        
        sentiment_scores = np.random.normal(0, 0.3, len(dates))
        article_counts = np.random.poisson(5, len(dates))  # Average 5 articles per day
        
        # Add some correlation with "market events"
        for i in range(len(dates)):
            if np.random.random() < 0.1:  # 10% chance of significant event
                sentiment_scores[i] += np.random.normal(0, 0.8)
                article_counts[i] += np.random.poisson(10)
        
        df = pd.DataFrame({
            'Date': dates,
            'sentiment_score': sentiment_scores,
            'article_count': article_counts,
            'positive_articles': np.maximum(0, (sentiment_scores + 1) * article_counts / 2).astype(int),
            'negative_articles': np.maximum(0, (1 - sentiment_scores) * article_counts / 2).astype(int),
        })
        df.set_index('Date', inplace=True)
        
        return df
    
    async def fetch_options_data(self, symbol: str, expiry_date: str = None) -> pd.DataFrame:
        """Fetch options chain data"""
        
        # For demo purposes, simulate options data
        # In production, you would use options data providers
        logger.info(f"Fetching options data for {symbol}")
        
        if expiry_date is None:
            # Default to next monthly expiry
            today = datetime.now()
            if today.day < 15:  # Before mid-month, use current month
                expiry_month = today.month
                expiry_year = today.year
            else:  # After mid-month, use next month
                expiry_month = today.month + 1
                expiry_year = today.year
                if expiry_month > 12:
                    expiry_month = 1
                    expiry_year += 1
            
            # Third Friday of the month
            expiry_date = pd.Timestamp(year=expiry_year, month=expiry_month, day=15)
            while expiry_date.weekday() != 4:  # Friday
                expiry_date += pd.Timedelta(days=1)
        
        # Get current stock price (simplified)
        try:
            stock = yf.Ticker(symbol)
            current_price = stock.history(period="1d")['Close'].iloc[-1]
        except:
            current_price = 100  # Default fallback
        
        # Generate options chain
        strikes = np.arange(
            current_price * 0.8, 
            current_price * 1.2, 
            current_price * 0.05
        )
        strikes = np.round(strikes / 5) * 5  # Round to nearest $5
        
        options_data = []
        for strike in strikes:
            # Simplified Black-Scholes-like pricing
            moneyness = strike / current_price
            time_to_expiry = (pd.Timestamp(expiry_date) - pd.Timestamp.now()).days / 365
            
            # Call options
            call_price = max(0, current_price - strike) + np.random.normal(2, 1)
            call_volume = np.random.poisson(50) if abs(moneyness - 1) < 0.1 else np.random.poisson(10)
            call_oi = np.random.poisson(200) if abs(moneyness - 1) < 0.1 else np.random.poisson(50)
            
            # Put options
            put_price = max(0, strike - current_price) + np.random.normal(2, 1)
            put_volume = np.random.poisson(30) if abs(moneyness - 1) < 0.1 else np.random.poisson(8)
            put_oi = np.random.poisson(150) if abs(moneyness - 1) < 0.1 else np.random.poisson(40)
            
            options_data.extend([
                {
                    'strike': strike,
                    'expiry': expiry_date,
                    'type': 'call',
                    'price': max(0.01, call_price),
                    'volume': call_volume,
                    'open_interest': call_oi,
                    'implied_volatility': np.random.normal(0.25, 0.05)
                },
                {
                    'strike': strike,
                    'expiry': expiry_date,
                    'type': 'put',
                    'price': max(0.01, put_price),
                    'volume': put_volume,
                    'open_interest': put_oi,
                    'implied_volatility': np.random.normal(0.23, 0.05)
                }
            ])
        
        df = pd.DataFrame(options_data)
        return df
    
    async def fetch_insider_trading(self, symbol: str, months_back: int = 6) -> pd.DataFrame:
        """Fetch insider trading data"""
        
        # For demo purposes, simulate insider trading data
        # In production, you would use SEC filing data or specialized providers
        logger.info(f"Fetching insider trading data for {symbol}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)
        
        # Generate random insider transactions
        num_transactions = np.random.poisson(15)  # Average 15 transactions over period
        
        insider_data = []
        for _ in range(num_transactions):
            transaction_date = start_date + timedelta(
                days=np.random.randint(0, (end_date - start_date).days)
            )
            
            # Random insider details
            insider_names = [f"Executive_{i}" for i in range(1, 11)] + \
                          [f"Director_{i}" for i in range(1, 6)] + \
                          [f"Officer_{i}" for i in range(1, 8)]
            
            transaction_type = np.random.choice(['Buy', 'Sell'], p=[0.3, 0.7])
            shares = np.random.randint(1000, 50000)
            price = np.random.normal(100, 20)  # Simplified price
            
            insider_data.append({
                'date': transaction_date,
                'insider_name': np.random.choice(insider_names),
                'transaction_type': transaction_type,
                'shares': shares,
                'price': max(1, price),
                'total_value': shares * max(1, price)
            })
        
        df = pd.DataFrame(insider_data)
        if not df.empty:
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    def get_data_quality_score(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate data quality metrics"""
        
        if data.empty:
            return {'quality_score': 0, 'issues': ['No data available']}
        
        issues = []
        score = 100
        
        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > 0.05:
            issues.append(f"High missing data: {missing_pct:.1%}")
            score -= missing_pct * 100
        
        # Check for data freshness
        if hasattr(data.index, 'max'):
            latest_date = data.index.max()
            days_old = (datetime.now() - latest_date).days
            if days_old > 1:
                issues.append(f"Data is {days_old} days old")
                score -= min(days_old * 2, 20)
        
        # Check for obvious anomalies (if price data)
        if 'Close' in data.columns:
            returns = data['Close'].pct_change()
            extreme_returns = returns[abs(returns) > 0.2]  # >20% daily moves
            if len(extreme_returns) / len(returns) > 0.01:  # >1% of days
                issues.append("High frequency of extreme price movements")
                score -= 10
        
        # Check data source reliability
        if 'source' in data.columns:
            source = data['source'].iloc[0] if len(data) > 0 else 'unknown'
            if source in ['yfinance', 'alpha_vantage']:
                score += 5  # Bonus for reliable sources
        
        return {
            'quality_score': max(0, min(100, score)),
            'issues': issues,
            'data_points': len(data),
            'date_range': f"{data.index.min()} to {data.index.max()}" if hasattr(data.index, 'min') else "N/A"
        }

# Configuration examples
def create_data_source_config() -> Dict[str, DataSource]:
    """Create default data source configuration"""
    
    return {
        'alpha_vantage': DataSource(
            name='Alpha Vantage',
            api_key=None,  # Set your API key here
            base_url='https://www.alphavantage.co/query',
            rate_limit=5,  # 5 requests per minute for free tier
            enabled=False  # Disabled until API key provided
        ),
        
        'polygon': DataSource(
            name='Polygon.io',
            api_key=None,  # Set your API key here
            base_url='https://api.polygon.io',
            rate_limit=5,  # 5 requests per minute for free tier
            enabled=False  # Disabled until API key provided
        ),
        
        'finnhub': DataSource(
            name='Finnhub',
            api_key=None,  # Set your API key here
            base_url='https://finnhub.io/api/v1',
            rate_limit=60,  # 60 requests per minute for free tier
            enabled=False  # Disabled until API key provided
        )
    }

async def main():
    """Demonstrate enhanced data sources"""
    print("=" * 60)
    print("ENHANCED DATA SOURCES INTEGRATION")
    print("=" * 60)
    
    # Create data provider
    config = create_data_source_config()
    
    async with EnhancedDataProvider(config) as provider:
        symbol = "AAPL"
        
        print(f"\n📊 Fetching data for {symbol}...")
        
        # Stock data
        stock_data = await provider.fetch_stock_data(symbol, period="1mo")
        print(f"Stock data: {len(stock_data)} records from {stock_data['source'].iloc[0] if not stock_data.empty else 'N/A'}")
        
        # Data quality assessment
        quality = provider.get_data_quality_score(stock_data, symbol)
        print(f"Data quality score: {quality['quality_score']:.1f}/100")
        if quality['issues']:
            print(f"Issues: {', '.join(quality['issues'])}")
        
        # Economic indicators
        gdp_data = await provider.fetch_economic_data('GDP_GROWTH', start_date='2023-01-01')
        print(f"GDP data: {len(gdp_data)} records")
        
        unemployment_data = await provider.fetch_economic_data('UNEMPLOYMENT_RATE', start_date='2023-01-01')
        print(f"Unemployment data: {len(unemployment_data)} records")
        
        # News sentiment
        sentiment_data = await provider.fetch_news_sentiment(symbol, days_back=30)
        print(f"Sentiment data: {len(sentiment_data)} records")
        print(f"Average sentiment: {sentiment_data['sentiment_score'].mean():.3f}")
        
        # Options data
        options_data = await provider.fetch_options_data(symbol)
        print(f"Options data: {len(options_data)} contracts")
        
        # Insider trading
        insider_data = await provider.fetch_insider_trading(symbol, months_back=6)
        print(f"Insider trading: {len(insider_data)} transactions")
        
        if not stock_data.empty:
            print(f"\n📈 Latest stock data for {symbol}:")
            print(f"Close: ${stock_data['Close'].iloc[-1]:.2f}")
            print(f"Volume: {stock_data['Volume'].iloc[-1]:,}")
            print(f"Date: {stock_data.index[-1]}")

if __name__ == "__main__":
    asyncio.run(main())

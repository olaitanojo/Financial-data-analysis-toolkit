#!/usr/bin/env python3
"""
Financial Data Analysis Toolkit
A comprehensive suite of financial analysis tools including market sentiment analysis,
portfolio optimization, economic indicator tracking, earnings prediction models,
advanced ML models, enhanced data sources, and interactive analytics.

Author: olaitanojo
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from textblob import TextBlob
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Import enhanced modules
try:
    from advanced_ml_models import AdvancedStockPredictor
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False
    print("⚠️  Advanced ML models not available. Install required dependencies.")

try:
    from enhanced_data_sources import EnhancedDataProvider, create_data_source_config
    DATA_SOURCES_AVAILABLE = True
except ImportError:
    DATA_SOURCES_AVAILABLE = False
    print("⚠️  Enhanced data sources not available.")

try:
    from enhanced_analytics import EnhancedAnalytics, EnhancedVisualizer
    ENHANCED_ANALYTICS_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYTICS_AVAILABLE = False
    print("⚠️  Enhanced analytics not available.")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketSentimentAnalyzer:
    """Analyzes market sentiment from various sources"""
    
    def __init__(self):
        self.sentiment_data = []
        
    def analyze_news_sentiment(self, ticker: str, num_articles: int = 20) -> dict:
        """
        Analyze news sentiment for a given ticker
        Note: This is a simplified version - real implementation would use news APIs
        """
        # Simulate news headlines for demo purposes
        sample_headlines = [
            f"{ticker} reports strong earnings beating expectations",
            f"Analysts upgrade {ticker} stock rating to buy",
            f"{ticker} announces new strategic partnership",
            f"Market volatility affects {ticker} performance",
            f"{ticker} CEO optimistic about future growth",
            f"Economic uncertainty impacts {ticker} outlook",
            f"{ticker} launches innovative product line",
            f"Regulatory concerns weigh on {ticker}",
            f"{ticker} exceeds revenue guidance for quarter",
            f"Supply chain issues challenge {ticker} operations"
        ]
        
        sentiments = []
        for headline in sample_headlines[:num_articles]:
            blob = TextBlob(headline)
            sentiments.append({
                'headline': headline,
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            })
        
        avg_sentiment = np.mean([s['polarity'] for s in sentiments])
        sentiment_label = 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral'
        
        return {
            'ticker': ticker,
            'average_sentiment': avg_sentiment,
            'sentiment_label': sentiment_label,
            'total_articles': len(sentiments),
            'detailed_sentiments': sentiments
        }
    
    def calculate_fear_greed_index(self, market_data: pd.DataFrame) -> dict:
        """Calculate a simplified Fear & Greed Index"""
        
        # Price momentum (last 20 days)
        returns_20d = market_data['Close'].pct_change(20).iloc[-1]
        momentum_score = min(100, max(0, (returns_20d + 0.1) * 500))
        
        # Volatility (VIX-like calculation)
        volatility_20d = market_data['Close'].pct_change().rolling(20).std().iloc[-1]
        volatility_score = min(100, max(0, (0.02 - volatility_20d) * 2500))
        
        # Volume momentum
        volume_avg = market_data['Volume'].rolling(20).mean().iloc[-1]
        volume_current = market_data['Volume'].iloc[-1]
        volume_score = min(100, max(0, (volume_current / volume_avg) * 50))
        
        # RSI
        delta = market_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_score = 100 - abs(rsi.iloc[-1] - 50) * 2
        
        overall_score = (momentum_score * 0.3 + volatility_score * 0.3 + 
                        volume_score * 0.2 + rsi_score * 0.2)
        
        if overall_score >= 75:
            sentiment = "Extreme Greed"
        elif overall_score >= 55:
            sentiment = "Greed"
        elif overall_score >= 45:
            sentiment = "Neutral"
        elif overall_score >= 25:
            sentiment = "Fear"
        else:
            sentiment = "Extreme Fear"
        
        return {
            'overall_score': overall_score,
            'sentiment': sentiment,
            'components': {
                'momentum': momentum_score,
                'volatility': volatility_score,
                'volume': volume_score,
                'rsi': rsi_score
            }
        }

class PortfolioOptimizer:
    """Modern Portfolio Theory optimizer"""
    
    def __init__(self, tickers: list, period: str = "1y"):
        self.tickers = tickers
        self.period = period
        self.returns_data = None
        self.cov_matrix = None
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical data for all tickers"""
        logger.info(f"Fetching data for {len(self.tickers)} tickers...")
        
        price_data = pd.DataFrame()
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=self.period)
                price_data[ticker] = hist['Close']
            except Exception as e:
                logger.warning(f"Could not fetch data for {ticker}: {e}")
        
        # Calculate daily returns
        self.returns_data = price_data.pct_change().dropna()
        self.cov_matrix = self.returns_data.cov() * 252  # Annualized
        
        return price_data
    
    def calculate_portfolio_metrics(self, weights: np.array) -> tuple:
        """Calculate portfolio return and risk"""
        portfolio_return = np.sum(self.returns_data.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_std
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def optimize_portfolio(self, target_return: float = None) -> dict:
        """Optimize portfolio using Modern Portfolio Theory"""
        if self.returns_data is None:
            self.fetch_data()
        
        num_assets = len(self.tickers)
        
        # Objective function (minimize portfolio variance)
        def objective(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(self.returns_data.mean() * x) * 252 - target_return
            })
        
        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess (equal weights)
        initial_guess = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(objective, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            port_return, port_std, sharpe = self.calculate_portfolio_metrics(optimal_weights)
            
            return {
                'success': True,
                'weights': dict(zip(self.tickers, optimal_weights)),
                'expected_return': port_return,
                'volatility': port_std,
                'sharpe_ratio': sharpe,
                'allocation_pct': {ticker: f"{weight*100:.1f}%" 
                                 for ticker, weight in zip(self.tickers, optimal_weights)}
            }
        else:
            return {'success': False, 'message': result.message}
    
    def efficient_frontier(self, num_portfolios: int = 50) -> pd.DataFrame:
        """Generate efficient frontier data"""
        if self.returns_data is None:
            self.fetch_data()
        
        mean_returns = self.returns_data.mean() * 252
        min_return = mean_returns.min()
        max_return = mean_returns.max()
        
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        efficient_portfolios = []
        
        for target in target_returns:
            result = self.optimize_portfolio(target_return=target)
            if result['success']:
                efficient_portfolios.append({
                    'return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio']
                })
        
        return pd.DataFrame(efficient_portfolios)

class EconomicIndicatorTracker:
    """Track and analyze economic indicators"""
    
    def __init__(self):
        self.indicators = {}
    
    def fetch_fred_data(self, series_id: str, start_date: str = None) -> pd.Series:
        """
        Fetch data from FRED (Federal Reserve Economic Data)
        Note: This is a simplified version - real implementation would use FRED API
        """
        # Simulate economic data for demo
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        # Generate sample data for different indicators
        dates = pd.date_range(start_date, datetime.now(), freq='D')
        
        if 'GDP' in series_id:
            # GDP growth simulation
            data = np.random.normal(2.1, 0.8, len(dates)) + np.sin(np.arange(len(dates))/365*2*np.pi) * 0.5
        elif 'UNRATE' in series_id:
            # Unemployment rate simulation
            data = np.random.normal(4.2, 1.2, len(dates)) + np.cos(np.arange(len(dates))/365*2*np.pi) * 0.3
        elif 'CPIAUCSL' in series_id:
            # Inflation rate simulation
            data = np.random.normal(2.8, 0.9, len(dates)) + np.sin(np.arange(len(dates))/365*4*np.pi) * 0.2
        else:
            # Generic economic indicator
            data = np.random.normal(100, 10, len(dates))
        
        return pd.Series(data, index=dates, name=series_id)
    
    def track_key_indicators(self) -> dict:
        """Track key economic indicators"""
        indicators = {
            'GDP_Growth': self.fetch_fred_data('GDPPOT'),
            'Unemployment_Rate': self.fetch_fred_data('UNRATE'),
            'Inflation_Rate': self.fetch_fred_data('CPIAUCSL'),
            'Interest_Rate': self.fetch_fred_data('FEDFUNDS')
        }
        
        # Calculate recent changes
        analysis = {}
        for name, data in indicators.items():
            current_value = data.iloc[-1]
            prev_month_value = data.iloc[-30] if len(data) >= 30 else data.iloc[0]
            change = current_value - prev_month_value
            pct_change = (change / prev_month_value) * 100 if prev_month_value != 0 else 0
            
            analysis[name] = {
                'current_value': current_value,
                'monthly_change': change,
                'monthly_pct_change': pct_change,
                'trend': 'Rising' if change > 0 else 'Falling' if change < 0 else 'Stable'
            }
        
        return analysis

class EarningsPredictionModel:
    """Machine learning model for earnings prediction"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_names = None
    
    def prepare_features(self, ticker: str, periods: int = 12) -> pd.DataFrame:
        """Prepare features for earnings prediction"""
        stock = yf.Ticker(ticker)
        
        # Get historical data
        hist = stock.history(period="2y")
        
        # Calculate technical indicators as features
        features = pd.DataFrame(index=hist.index)
        
        # Price-based features
        features['price'] = hist['Close']
        features['volume'] = hist['Volume']
        features['high_low_ratio'] = hist['High'] / hist['Low']
        features['price_change'] = hist['Close'].pct_change()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = hist['Close'].rolling(window).mean()
            features[f'price_sma_{window}_ratio'] = features['price'] / features[f'sma_{window}']
        
        # Volatility
        features['volatility_10d'] = features['price_change'].rolling(10).std()
        features['volatility_30d'] = features['price_change'].rolling(30).std()
        
        # RSI
        delta = features['price_change']
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def train_model(self, ticker: str) -> dict:
        """Train the earnings prediction model"""
        logger.info(f"Training earnings prediction model for {ticker}...")
        
        # Prepare features
        features_df = self.prepare_features(ticker)
        
        # For demo purposes, create synthetic earnings data
        # In reality, this would come from actual earnings reports
        np.random.seed(42)
        earnings_data = []
        for i in range(0, len(features_df), 63):  # Quarterly earnings (roughly every 63 trading days)
            base_earnings = np.random.normal(1.5, 0.3)  # Base EPS
            price_factor = features_df.iloc[i]['price'] / 100  # Price influence
            volume_factor = features_df.iloc[i]['volume'] / features_df['volume'].mean()
            
            # Calculate earnings with some relationship to stock metrics
            earnings = base_earnings * (1 + (price_factor - 1) * 0.1 + (volume_factor - 1) * 0.05)
            earnings_data.append((features_df.index[i], earnings))
        
        # Create earnings DataFrame
        earnings_df = pd.DataFrame(earnings_data, columns=['date', 'eps'])
        earnings_df.set_index('date', inplace=True)
        
        # Align features with earnings dates
        aligned_features = []
        aligned_earnings = []
        
        for date, eps in earnings_df.iterrows():
            if date in features_df.index:
                aligned_features.append(features_df.loc[date].values)
                aligned_earnings.append(eps)
        
        if len(aligned_features) < 4:
            return {'success': False, 'message': 'Insufficient data for training'}
        
        X = np.array(aligned_features)
        y = np.array(aligned_earnings)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.feature_names = features_df.columns.tolist()
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'success': True,
            'mse': mse,
            'r2_score': r2,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict_earnings(self, ticker: str) -> dict:
        """Predict next earnings for a ticker"""
        if not self.is_trained:
            train_result = self.train_model(ticker)
            if not train_result['success']:
                return train_result
        
        # Get current features
        features_df = self.prepare_features(ticker)
        current_features = features_df.iloc[-1:].values
        
        # Make prediction
        predicted_eps = self.model.predict(current_features)[0]
        
        # Calculate confidence interval (simplified)
        feature_importances = self.model.feature_importances_
        prediction_confidence = np.mean(feature_importances) * 100
        
        return {
            'success': True,
            'predicted_eps': predicted_eps,
            'confidence': prediction_confidence,
            'prediction_date': features_df.index[-1],
            'model_features': len(self.feature_names)
        }

class FinancialAnalysisToolkit:
    """Main toolkit class combining all analysis tools"""
    
    def __init__(self):
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.portfolio_optimizer = None
        self.economic_tracker = EconomicIndicatorTracker()
        self.earnings_predictor = EarningsPredictionModel()
    
    def comprehensive_analysis(self, ticker: str) -> dict:
        """Run comprehensive analysis on a ticker"""
        logger.info(f"Running comprehensive analysis for {ticker}...")
        
        results = {
            'ticker': ticker,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        try:
            # Market sentiment analysis
            sentiment_result = self.sentiment_analyzer.analyze_news_sentiment(ticker)
            results['sentiment_analysis'] = sentiment_result
            
            # Fear & Greed Index
            stock_data = yf.Ticker(ticker).history(period="3mo")
            fear_greed = self.sentiment_analyzer.calculate_fear_greed_index(stock_data)
            results['fear_greed_index'] = fear_greed
            
            # Economic indicators
            economic_data = self.economic_tracker.track_key_indicators()
            results['economic_indicators'] = economic_data
            
            # Earnings prediction
            earnings_pred = self.earnings_predictor.predict_earnings(ticker)
            results['earnings_prediction'] = earnings_pred
            
            # Generate overall recommendation
            recommendation = self._generate_recommendation(results)
            results['recommendation'] = recommendation
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_recommendation(self, analysis_results: dict) -> dict:
        """Generate investment recommendation based on analysis"""
        score = 0
        factors = []
        
        # Sentiment scoring
        sentiment = analysis_results.get('sentiment_analysis', {})
        if sentiment.get('sentiment_label') == 'Positive':
            score += 2
            factors.append("Positive news sentiment")
        elif sentiment.get('sentiment_label') == 'Negative':
            score -= 2
            factors.append("Negative news sentiment")
        
        # Fear & Greed scoring
        fear_greed = analysis_results.get('fear_greed_index', {})
        fg_sentiment = fear_greed.get('sentiment', '')
        if 'Greed' in fg_sentiment:
            score -= 1
            factors.append("Market showing greed (potential pullback)")
        elif 'Fear' in fg_sentiment:
            score += 1
            factors.append("Market showing fear (potential opportunity)")
        
        # Economic indicators
        econ_data = analysis_results.get('economic_indicators', {})
        positive_trends = sum(1 for indicator in econ_data.values() 
                            if indicator.get('trend') == 'Rising' and 'GDP' in str(indicator))
        if positive_trends > 0:
            score += 1
            factors.append("Positive economic trends")
        
        # Earnings prediction
        earnings = analysis_results.get('earnings_prediction', {})
        if earnings.get('success') and earnings.get('predicted_eps', 0) > 0:
            score += 1
            factors.append("Positive earnings outlook")
        
        # Generate recommendation
        if score >= 3:
            recommendation = "Strong Buy"
        elif score >= 1:
            recommendation = "Buy"
        elif score >= -1:
            recommendation = "Hold"
        elif score >= -3:
            recommendation = "Sell"
        else:
            recommendation = "Strong Sell"
        
        return {
            'recommendation': recommendation,
            'score': score,
            'factors': factors
        }

def main():
    """Main function to demonstrate the financial analysis toolkit"""
    logger.info("Starting Financial Data Analysis Toolkit")
    
    toolkit = FinancialAnalysisToolkit()
    
    # Example analysis for AAPL
    ticker = "AAPL"
    
    print("="*60)
    print("FINANCIAL DATA ANALYSIS TOOLKIT")
    print("="*60)
    print(f"Analyzing: {ticker}")
    print()
    
    # Run comprehensive analysis
    analysis = toolkit.comprehensive_analysis(ticker)
    
    # Display results
    print("📊 SENTIMENT ANALYSIS")
    print("-" * 30)
    sentiment = analysis.get('sentiment_analysis', {})
    print(f"Sentiment: {sentiment.get('sentiment_label', 'N/A')}")
    print(f"Score: {sentiment.get('average_sentiment', 0):.3f}")
    print(f"Articles Analyzed: {sentiment.get('total_articles', 0)}")
    print()
    
    print("😱 FEAR & GREED INDEX")
    print("-" * 30)
    fg_index = analysis.get('fear_greed_index', {})
    print(f"Overall Score: {fg_index.get('overall_score', 0):.1f}/100")
    print(f"Sentiment: {fg_index.get('sentiment', 'N/A')}")
    print()
    
    print("🏦 ECONOMIC INDICATORS")
    print("-" * 30)
    econ_data = analysis.get('economic_indicators', {})
    for indicator, data in econ_data.items():
        print(f"{indicator}: {data.get('current_value', 0):.2f} ({data.get('trend', 'N/A')})")
    print()
    
    print("📈 EARNINGS PREDICTION")
    print("-" * 30)
    earnings = analysis.get('earnings_prediction', {})
    if earnings.get('success'):
        print(f"Predicted EPS: ${earnings.get('predicted_eps', 0):.2f}")
        print(f"Confidence: {earnings.get('confidence', 0):.1f}%")
    else:
        print("Earnings prediction not available")
    print()
    
    print("🎯 INVESTMENT RECOMMENDATION")
    print("-" * 30)
    recommendation = analysis.get('recommendation', {})
    print(f"Recommendation: {recommendation.get('recommendation', 'N/A')}")
    print(f"Score: {recommendation.get('score', 0)}")
    print("Factors:")
    for factor in recommendation.get('factors', []):
        print(f"  • {factor}")
    print()
    
    # Portfolio optimization example
    print("📊 PORTFOLIO OPTIMIZATION EXAMPLE")
    print("-" * 30)
    
    portfolio_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    optimizer = PortfolioOptimizer(portfolio_tickers)
    
    try:
        optimal_portfolio = optimizer.optimize_portfolio()
        if optimal_portfolio['success']:
            print("Optimal Portfolio Allocation:")
            for ticker, allocation in optimal_portfolio['allocation_pct'].items():
                print(f"  {ticker}: {allocation}")
            print(f"Expected Return: {optimal_portfolio['expected_return']:.1%}")
            print(f"Risk (Volatility): {optimal_portfolio['volatility']:.1%}")
            print(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.2f}")
        else:
            print("Portfolio optimization failed")
    except Exception as e:
        print(f"Portfolio optimization error: {e}")
    
    # Enhanced Features Demonstration
    print("\n" + "="*60)
    print("ENHANCED FEATURES")
    print("="*60)
    
    # Advanced ML Models
    if ML_MODELS_AVAILABLE:
        print("\n🤖 ADVANCED ML PREDICTIONS")
        print("-" * 30)
        try:
            predictor = AdvancedStockPredictor(ticker)
            predictor.fetch_enhanced_data(period="1y")
            predictor.prepare_features_target(target_days=1)
            
            # Train XGBoost model (quick version)
            xgb_results = predictor.train_xgboost_model(hyperparameter_tuning=False)
            print(f"XGBoost R² Score: {xgb_results['test_r2']:.4f}")
            print(f"XGBoost MAE: {xgb_results['test_mae']:.4f}")
            
            # Make predictions
            predictions = predictor.predict_with_all_models()
            for model_name, pred in predictions.items():
                print(f"{model_name.title()} Prediction: {pred:.2f}% price change")
                
        except Exception as e:
            print(f"ML models error: {e}")
    
    # Enhanced Analytics
    if ENHANCED_ANALYTICS_AVAILABLE:
        print("\n📊 ENHANCED ANALYTICS")
        print("-" * 30)
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")
            
            # Initialize analytics
            analytics = EnhancedAnalytics()
            analytics.load_data(data)
            
            # Calculate advanced metrics
            metrics = analytics.calculate_advanced_metrics()
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"VaR (95%): {metrics['var_95']:.2f}%")
            
            # Technical analysis
            technical = analytics.technical_analysis()
            print(f"RSI: {technical['momentum']['rsi']:.1f}")
            print(f"Trend Strength: {technical['trend_analysis']['trend_strength']:.1f}")
            
            # Create visualizations
            visualizer = EnhancedVisualizer()
            chart = visualizer.create_comprehensive_chart(data, f"{ticker} Enhanced Analysis")
            chart.write_html(f"{ticker}_enhanced_chart.html")
            print(f"Enhanced chart saved as {ticker}_enhanced_chart.html")
            
        except Exception as e:
            print(f"Enhanced analytics error: {e}")
    
    # Enhanced Data Sources (async demo)
    if DATA_SOURCES_AVAILABLE:
        print("\n📡 ENHANCED DATA SOURCES")
        print("-" * 30)
        try:
            async def demo_data_sources():
                config = create_data_source_config()
                async with EnhancedDataProvider(config) as provider:
                    # Economic data
                    gdp_data = await provider.fetch_economic_data('GDP_GROWTH')
                    print(f"GDP data points: {len(gdp_data)}")
                    
                    # News sentiment
                    sentiment_data = await provider.fetch_news_sentiment(ticker)
                    print(f"Sentiment data points: {len(sentiment_data)}")
                    print(f"Average sentiment: {sentiment_data['sentiment_score'].mean():.3f}")
                    
                    # Options data
                    options_data = await provider.fetch_options_data(ticker)
                    print(f"Options contracts: {len(options_data)}")
            
            # Run async demo
            asyncio.run(demo_data_sources())
            
        except Exception as e:
            print(f"Enhanced data sources error: {e}")
    
    print("\n✅ All analyses completed successfully!")
    logger.info("Financial Data Analysis Toolkit completed")

async def run_enhanced_demo():
    """Run enhanced features demo with async support"""
    print("Running Enhanced Financial Analysis Demo...")
    
    if DATA_SOURCES_AVAILABLE:
        config = create_data_source_config()
        async with EnhancedDataProvider(config) as provider:
            ticker = "AAPL"
            
            # Fetch enhanced data
            stock_data = await provider.fetch_stock_data(ticker)
            print(f"Fetched {len(stock_data)} stock records")
            
            # Quality assessment
            quality = provider.get_data_quality_score(stock_data, ticker)
            print(f"Data quality: {quality['quality_score']:.1f}/100")
            
            # Multiple data types
            sentiment_data = await provider.fetch_news_sentiment(ticker)
            options_data = await provider.fetch_options_data(ticker)
            insider_data = await provider.fetch_insider_trading(ticker)
            
            print(f"Sentiment records: {len(sentiment_data)}")
            print(f"Options contracts: {len(options_data)}")
            print(f"Insider transactions: {len(insider_data)}")
    
    print("Enhanced demo completed!")

if __name__ == "__main__":
    main()

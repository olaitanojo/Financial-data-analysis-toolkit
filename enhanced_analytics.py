#!/usr/bin/env python3
"""
Enhanced Analytics and Visualization Tools
Provides advanced charting, statistical analysis, and interactive visualizations
for financial data analysis.

Author: olaitanojo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAnalytics:
    """Enhanced analytics with advanced statistical analysis"""
    
    def __init__(self):
        self.data = None
        self.analysis_results = {}
        
    def load_data(self, data: pd.DataFrame):
        """Load financial data for analysis"""
        self.data = data.copy()
        if 'Close' in self.data.columns:
            self.data['returns'] = self.data['Close'].pct_change()
            self.data['log_returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        logger.info(f"Loaded data with {len(self.data)} records")
        
    def calculate_advanced_metrics(self) -> dict:
        """Calculate advanced risk and performance metrics"""
        if self.data is None or 'Close' not in self.data.columns:
            raise ValueError("No price data loaded")
        
        returns = self.data['returns'].dropna()
        prices = self.data['Close']
        
        # Basic metrics
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        annualized_return = ((1 + returns.mean())**252 - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Risk-adjusted metrics
        sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return / 100) / (downside_deviation / 100) if downside_deviation > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        # Expected Shortfall (CVaR)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Calmar Ratio
        calmar_ratio = (annualized_return / 100) / abs(max_drawdown / 100) if max_drawdown != 0 else 0
        
        # Beta (using SPY as proxy for market)
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY").history(period="1y")
            spy_returns = spy['Close'].pct_change().dropna()
            
            # Align dates
            common_dates = returns.index.intersection(spy_returns.index)
            if len(common_dates) > 30:  # Need sufficient data
                aligned_returns = returns.loc[common_dates]
                aligned_spy = spy_returns.loc[common_dates]
                
                covariance = np.cov(aligned_returns, aligned_spy)[0][1]
                market_variance = np.var(aligned_spy)
                beta = covariance / market_variance if market_variance > 0 else 1
                
                # Alpha
                risk_free_rate = 0.02  # Assume 2% risk-free rate
                alpha = (annualized_return / 100) - risk_free_rate - beta * ((aligned_spy.mean() * 252) - risk_free_rate)
            else:
                beta = 1.0  # Default
                alpha = 0.0
        except:
            beta = 1.0  # Default
            alpha = 0.0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'calmar_ratio': calmar_ratio,
            'beta': beta,
            'alpha': alpha,
            'downside_deviation': downside_deviation
        }
        
        self.analysis_results['metrics'] = metrics
        return metrics
    
    def technical_analysis(self) -> dict:
        """Perform comprehensive technical analysis"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        data = self.data.copy()
        
        # Support and Resistance levels
        support_resistance = self._find_support_resistance(data['Close'])
        
        # Chart patterns
        patterns = self._detect_patterns(data)
        
        # Momentum indicators
        momentum = self._calculate_momentum_indicators(data)
        
        # Volume analysis
        volume_analysis = self._analyze_volume(data) if 'Volume' in data.columns else {}
        
        # Trend analysis
        trend_analysis = self._analyze_trends(data)
        
        technical_results = {
            'support_resistance': support_resistance,
            'patterns': patterns,
            'momentum': momentum,
            'volume_analysis': volume_analysis,
            'trend_analysis': trend_analysis
        }
        
        self.analysis_results['technical'] = technical_results
        return technical_results
    
    def _find_support_resistance(self, prices: pd.Series, window: int = 20) -> dict:
        """Find support and resistance levels"""
        
        # Calculate local minima and maxima
        from scipy.signal import argrelextrema
        
        # Convert to numpy array for scipy
        price_array = prices.values
        
        # Find local minima (support) and maxima (resistance)
        local_min_idx = argrelextrema(price_array, np.less, order=window)[0]
        local_max_idx = argrelextrema(price_array, np.greater, order=window)[0]
        
        # Get the actual price levels
        support_levels = price_array[local_min_idx]
        resistance_levels = price_array[local_max_idx]
        
        # Cluster nearby levels
        def cluster_levels(levels, threshold=0.02):
            if len(levels) == 0:
                return []
            
            clustered = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - np.mean(current_cluster)) / np.mean(current_cluster) < threshold:
                    current_cluster.append(level)
                else:
                    clustered.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clustered.append(np.mean(current_cluster))
            return sorted(clustered)
        
        clustered_support = cluster_levels(support_levels)
        clustered_resistance = cluster_levels(resistance_levels)
        
        return {
            'support_levels': clustered_support[-5:],  # Last 5 support levels
            'resistance_levels': clustered_resistance[-5:],  # Last 5 resistance levels
            'current_price': prices.iloc[-1],
            'nearest_support': min(clustered_support, key=lambda x: abs(x - prices.iloc[-1])) if clustered_support else None,
            'nearest_resistance': min(clustered_resistance, key=lambda x: abs(x - prices.iloc[-1])) if clustered_resistance else None
        }
    
    def _detect_patterns(self, data: pd.DataFrame) -> dict:
        """Detect common chart patterns"""
        
        patterns = {}
        prices = data['Close']
        
        # Head and Shoulders pattern detection (simplified)
        def detect_head_shoulders(prices, window=20):
            if len(prices) < window * 3:
                return False
            
            recent_prices = prices.tail(window * 3)
            peaks = []
            
            # Find three peaks
            for i in range(window, len(recent_prices) - window):
                if (recent_prices.iloc[i] > recent_prices.iloc[i-window:i].max() and 
                    recent_prices.iloc[i] > recent_prices.iloc[i+1:i+window+1].max()):
                    peaks.append((i, recent_prices.iloc[i]))
            
            if len(peaks) >= 3:
                # Check if middle peak is highest (head) and others are similar (shoulders)
                peaks = sorted(peaks, key=lambda x: x[0])[-3:]
                left_shoulder, head, right_shoulder = peaks
                
                if (head[1] > left_shoulder[1] * 1.02 and 
                    head[1] > right_shoulder[1] * 1.02 and
                    abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.05):
                    return True
            
            return False
        
        patterns['head_and_shoulders'] = detect_head_shoulders(prices)
        
        # Double top/bottom detection (simplified)
        def detect_double_pattern(prices, pattern_type='top'):
            recent_prices = prices.tail(100)  # Last 100 periods
            if pattern_type == 'top':
                extremes = recent_prices.nlargest(10)  # Top 10 highs
            else:
                extremes = recent_prices.nsmallest(10)  # Bottom 10 lows
            
            # Check if two similar extremes exist
            for i, price1 in enumerate(extremes):
                for price2 in extremes[i+1:]:
                    if abs(price1 - price2) / price1 < 0.02:  # Within 2%
                        return True
            return False
        
        patterns['double_top'] = detect_double_pattern(prices, 'top')
        patterns['double_bottom'] = detect_double_pattern(prices, 'bottom')
        
        # Triangle pattern (simplified convergence detection)
        def detect_triangle(prices, window=50):
            if len(prices) < window:
                return False
            
            recent = prices.tail(window)
            highs = recent.rolling(5).max()
            lows = recent.rolling(5).min()
            
            # Check if range is converging
            early_range = (highs.iloc[:window//2].mean() - lows.iloc[:window//2].mean())
            late_range = (highs.iloc[window//2:].mean() - lows.iloc[window//2:].mean())
            
            return late_range < early_range * 0.7  # Range decreased by 30%
        
        patterns['triangle'] = detect_triangle(prices)
        
        return patterns
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> dict:
        """Calculate various momentum indicators"""
        
        momentum = {}
        prices = data['Close']
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        momentum['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
        
        # MACD
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        momentum['macd'] = macd.iloc[-1] if not macd.empty else 0
        momentum['macd_signal'] = macd_signal.iloc[-1] if not macd_signal.empty else 0
        momentum['macd_histogram'] = momentum['macd'] - momentum['macd_signal']
        
        # Stochastic Oscillator
        if len(prices) >= 14:
            low_14 = data['Low'].rolling(14).min() if 'Low' in data else prices.rolling(14).min()
            high_14 = data['High'].rolling(14).max() if 'High' in data else prices.rolling(14).max()
            k_percent = 100 * ((prices - low_14) / (high_14 - low_14))
            momentum['stoch_k'] = k_percent.iloc[-1] if not k_percent.empty else 50
            momentum['stoch_d'] = k_percent.rolling(3).mean().iloc[-1] if not k_percent.empty else 50
        else:
            momentum['stoch_k'] = 50
            momentum['stoch_d'] = 50
        
        # Williams %R
        if len(prices) >= 14:
            momentum['williams_r'] = -100 * ((high_14.iloc[-1] - prices.iloc[-1]) / (high_14.iloc[-1] - low_14.iloc[-1])) if 'Low' in data else -50
        else:
            momentum['williams_r'] = -50
        
        return momentum
    
    def _analyze_volume(self, data: pd.DataFrame) -> dict:
        """Analyze volume patterns"""
        
        if 'Volume' not in data.columns:
            return {}
        
        volume = data['Volume']
        prices = data['Close']
        
        # Volume moving averages
        volume_sma_20 = volume.rolling(20).mean()
        volume_ratio = volume.iloc[-1] / volume_sma_20.iloc[-1] if not volume_sma_20.empty else 1
        
        # On-Balance Volume (OBV)
        obv = []
        obv_value = 0
        for i in range(1, len(data)):
            if prices.iloc[i] > prices.iloc[i-1]:
                obv_value += volume.iloc[i]
            elif prices.iloc[i] < prices.iloc[i-1]:
                obv_value -= volume.iloc[i]
            obv.append(obv_value)
        
        # Volume Price Trend (VPT)
        vpt = []
        vpt_value = 0
        for i in range(1, len(data)):
            price_change = (prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]
            vpt_value += volume.iloc[i] * price_change
            vpt.append(vpt_value)
        
        return {
            'volume_ratio': volume_ratio,
            'avg_volume_20': volume_sma_20.iloc[-1] if not volume_sma_20.empty else volume.mean(),
            'current_volume': volume.iloc[-1],
            'obv_trend': 'rising' if len(obv) > 10 and obv[-1] > obv[-10] else 'falling',
            'vpt_trend': 'rising' if len(vpt) > 10 and vpt[-1] > vpt[-10] else 'falling'
        }
    
    def _analyze_trends(self, data: pd.DataFrame) -> dict:
        """Analyze price trends"""
        
        prices = data['Close']
        
        # Moving average trends
        sma_20 = prices.rolling(20).mean()
        sma_50 = prices.rolling(50).mean()
        sma_200 = prices.rolling(200).mean()
        
        # Current position relative to MAs
        current_price = prices.iloc[-1]
        trend_signals = []
        
        if not sma_20.empty and current_price > sma_20.iloc[-1]:
            trend_signals.append('Above 20-day MA')
        if not sma_50.empty and current_price > sma_50.iloc[-1]:
            trend_signals.append('Above 50-day MA')
        if not sma_200.empty and current_price > sma_200.iloc[-1]:
            trend_signals.append('Above 200-day MA')
        
        # Golden Cross / Death Cross
        golden_cross = False
        death_cross = False
        
        if not sma_50.empty and not sma_200.empty and len(sma_50) > 5:
            if (sma_50.iloc[-1] > sma_200.iloc[-1] and 
                sma_50.iloc[-5] <= sma_200.iloc[-5]):
                golden_cross = True
            elif (sma_50.iloc[-1] < sma_200.iloc[-1] and 
                  sma_50.iloc[-5] >= sma_200.iloc[-5]):
                death_cross = True
        
        # Trend strength (ADX-like calculation)
        if len(prices) >= 14:
            high = data['High'] if 'High' in data else prices
            low = data['Low'] if 'Low' in data else prices
            
            tr1 = high - low
            tr2 = abs(high - prices.shift(1))
            tr3 = abs(low - prices.shift(1))
            true_range = pd.DataFrame([tr1, tr2, tr3]).max()
            atr = true_range.rolling(14).mean()
            
            plus_dm = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)
            minus_dm = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)
            
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean()
            
            trend_strength = adx.iloc[-1] if not adx.empty and not np.isnan(adx.iloc[-1]) else 25
        else:
            trend_strength = 25  # Default moderate strength
        
        return {
            'trend_signals': trend_signals,
            'golden_cross': golden_cross,
            'death_cross': death_cross,
            'trend_strength': trend_strength,
            'ma_alignment': 'bullish' if len(trend_signals) >= 2 else 'bearish' if len(trend_signals) == 0 else 'mixed'
        }

class EnhancedVisualizer:
    """Enhanced visualization tools with interactive charts"""
    
    def __init__(self):
        self.style_config = {
            'figure_size': (15, 10),
            'color_palette': 'Set2',
            'grid_alpha': 0.3
        }
    
    def create_comprehensive_chart(self, data: pd.DataFrame, title: str = "Financial Analysis") -> go.Figure:
        """Create comprehensive multi-panel chart"""
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            row_heights=[0.5, 0.2, 0.15, 0.15],
            subplot_titles=('Price & Technical Indicators', 'Volume', 'RSI', 'MACD'),
            vertical_spacing=0.05,
            shared_xaxes=True
        )
        
        # Main price chart with candlesticks
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # Add moving averages
        if len(data) >= 20:
            sma_20 = data['Close'].rolling(20).mean()
            sma_50 = data['Close'].rolling(50).mean()
            
            fig.add_trace(
                go.Scatter(x=data.index, y=sma_20, name='SMA 20', 
                          line=dict(color='orange', dash='dash')),
                row=1, col=1
            )
            
            if len(data) >= 50:
                fig.add_trace(
                    go.Scatter(x=data.index, y=sma_50, name='SMA 50', 
                              line=dict(color='red', dash='dot')),
                    row=1, col=1
                )
        
        # Volume chart
        if 'Volume' in data.columns:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(data['Close'], data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # RSI
        if len(data) >= 14:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(x=data.index, y=rsi, name='RSI', 
                          line=dict(color='purple')),
                row=3, col=1
            )
            
            # RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Overbought", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         annotation_text="Oversold", row=3, col=1)
        
        # MACD
        if len(data) >= 26:
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal
            
            fig.add_trace(
                go.Scatter(x=data.index, y=macd, name='MACD', 
                          line=dict(color='blue')),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=data.index, y=macd_signal, name='Signal', 
                          line=dict(color='red', dash='dash')),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Bar(x=data.index, y=macd_histogram, name='Histogram', 
                      marker_color='gray', opacity=0.6),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap"""
        
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            xaxis_title='Variables',
            yaxis_title='Variables',
            height=600
        )
        
        return fig
    
    def create_performance_dashboard(self, metrics: dict) -> go.Figure:
        """Create performance metrics dashboard"""
        
        # Create subplots for different metric categories
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Return Metrics', 'Risk Metrics', 'Risk-Adjusted Metrics', 'Distribution Metrics'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Return metrics
        return_metrics = ['total_return', 'annualized_return']
        return_values = [metrics.get(metric, 0) for metric in return_metrics]
        return_labels = ['Total Return (%)', 'Annualized Return (%)']
        
        fig.add_trace(
            go.Bar(x=return_labels, y=return_values, name='Returns',
                  marker_color='green'),
            row=1, col=1
        )
        
        # Risk metrics
        risk_metrics = ['volatility', 'max_drawdown', 'var_95']
        risk_values = [abs(metrics.get(metric, 0)) for metric in risk_metrics]
        risk_labels = ['Volatility (%)', 'Max Drawdown (%)', 'VaR 95% (%)']
        
        fig.add_trace(
            go.Bar(x=risk_labels, y=risk_values, name='Risk',
                  marker_color='red'),
            row=1, col=2
        )
        
        # Risk-adjusted metrics
        risk_adj_metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
        risk_adj_values = [metrics.get(metric, 0) for metric in risk_adj_metrics]
        risk_adj_labels = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
        
        fig.add_trace(
            go.Bar(x=risk_adj_labels, y=risk_adj_values, name='Risk-Adjusted',
                  marker_color='blue'),
            row=2, col=1
        )
        
        # Distribution metrics
        dist_metrics = ['skewness', 'kurtosis', 'beta']
        dist_values = [metrics.get(metric, 0) for metric in dist_metrics]
        dist_labels = ['Skewness', 'Kurtosis', 'Beta']
        
        fig.add_trace(
            go.Bar(x=dist_labels, y=dist_values, name='Distribution',
                  marker_color='purple'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Performance Metrics Dashboard',
            height=600,
            showlegend=False
        )
        
        return fig

def main():
    """Demonstrate enhanced analytics"""
    print("=" * 60)
    print("ENHANCED ANALYTICS AND VISUALIZATION")
    print("=" * 60)
    
    # Load sample data
    import yfinance as yf
    
    ticker = "AAPL"
    print(f"\n📊 Analyzing {ticker}...")
    
    # Fetch data
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")
    
    # Initialize analytics
    analytics = EnhancedAnalytics()
    analytics.load_data(data)
    
    # Calculate advanced metrics
    print("\n📈 Calculating Advanced Metrics...")
    metrics = analytics.calculate_advanced_metrics()
    
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
    print(f"Volatility: {metrics['volatility']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"VaR (95%): {metrics['var_95']:.2f}%")
    
    # Technical analysis
    print("\n🔍 Performing Technical Analysis...")
    technical = analytics.technical_analysis()
    
    print(f"RSI: {technical['momentum']['rsi']:.1f}")
    print(f"MACD: {technical['momentum']['macd']:.3f}")
    print(f"Trend Strength (ADX-like): {technical['trend_analysis']['trend_strength']:.1f}")
    
    # Support/Resistance
    sr = technical['support_resistance']
    print(f"\nNearest Support: ${sr['nearest_support']:.2f}" if sr['nearest_support'] else "No support level identified")
    print(f"Nearest Resistance: ${sr['nearest_resistance']:.2f}" if sr['nearest_resistance'] else "No resistance level identified")
    
    # Pattern detection
    patterns = technical['patterns']
    detected_patterns = [pattern for pattern, detected in patterns.items() if detected]
    if detected_patterns:
        print(f"Detected Patterns: {', '.join(detected_patterns)}")
    else:
        print("No significant patterns detected")
    
    # Create visualizations
    print("\n📊 Creating Visualizations...")
    visualizer = EnhancedVisualizer()
    
    # Comprehensive chart
    chart = visualizer.create_comprehensive_chart(data, f"{ticker} Analysis")
    chart.write_html(f"{ticker}_comprehensive_analysis.html")
    print(f"Comprehensive chart saved as {ticker}_comprehensive_analysis.html")
    
    # Performance dashboard
    dashboard = visualizer.create_performance_dashboard(metrics)
    dashboard.write_html(f"{ticker}_performance_dashboard.html")
    print(f"Performance dashboard saved as {ticker}_performance_dashboard.html")
    
    print("\n✅ Enhanced analytics completed!")

if __name__ == "__main__":
    main()

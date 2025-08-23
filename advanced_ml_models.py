#!/usr/bin/env python3
"""
Advanced Machine Learning Models for Financial Analysis
Includes LSTM for time series prediction, XGBoost for feature-based prediction,
and ensemble methods combining multiple models.

Author: olaitanojo
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedStockPredictor:
    """Advanced ML models for stock price prediction"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.models = {}
        self.scalers = {}
        self.data = None
        self.features = None
        self.target = None
        
    def fetch_enhanced_data(self, period: str = "2y") -> pd.DataFrame:
        """Fetch and prepare enhanced dataset with technical indicators"""
        logger.info(f"Fetching enhanced data for {self.ticker}...")
        
        # Fetch stock data
        stock = yf.Ticker(self.ticker)
        data = stock.history(period=period)
        
        # Add technical indicators
        data = self._add_technical_indicators(data)
        
        # Add economic features
        data = self._add_economic_features(data)
        
        # Add sentiment features (simplified)
        data = self._add_sentiment_features(data)
        
        self.data = data.dropna()
        logger.info(f"Dataset prepared with {len(self.data)} samples and {len(self.data.columns)} features")
        
        return self.data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        
        # Price-based indicators
        data['price_sma_5'] = data['Close'].rolling(5).mean()
        data['price_sma_10'] = data['Close'].rolling(10).mean()
        data['price_sma_20'] = data['Close'].rolling(20).mean()
        data['price_sma_50'] = data['Close'].rolling(50).mean()
        data['price_ema_12'] = data['Close'].ewm(span=12).mean()
        data['price_ema_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['macd'] = data['price_ema_12'] - data['price_ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma_20 = data['Close'].rolling(20).mean()
        std_20 = data['Close'].rolling(20).std()
        data['bb_upper'] = sma_20 + (std_20 * 2)
        data['bb_lower'] = sma_20 - (std_20 * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / sma_20
        data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        data['stoch_k'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
        data['stoch_d'] = data['stoch_k'].rolling(3).mean()
        
        # Average True Range (ATR)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data['atr'] = true_range.rolling(14).mean()
        
        # Williams %R
        data['williams_r'] = -100 * ((high_14 - data['Close']) / (high_14 - low_14))
        
        # Volume indicators
        data['volume_sma'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma']
        data['price_volume'] = data['Close'] * data['Volume']
        
        # Volatility
        data['volatility_10'] = data['Close'].pct_change().rolling(10).std()
        data['volatility_30'] = data['Close'].pct_change().rolling(30).std()
        
        # Price momentum
        data['momentum_5'] = data['Close'] / data['Close'].shift(5)
        data['momentum_10'] = data['Close'] / data['Close'].shift(10)
        data['momentum_20'] = data['Close'] / data['Close'].shift(20)
        
        return data
    
    def _add_economic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add economic indicators (simplified simulation)"""
        
        # Simulate economic data (in practice, fetch from FRED or other sources)
        np.random.seed(42)
        
        # Interest rates (simulate daily changes)
        base_rate = 2.5
        rate_changes = np.random.normal(0, 0.01, len(data))
        data['interest_rate'] = base_rate + np.cumsum(rate_changes) * 0.1
        
        # Market volatility index (VIX-like)
        data['market_volatility'] = np.random.normal(20, 5, len(data)) + data['volatility_30'] * 100
        
        # Dollar strength index (simplified)
        data['dollar_strength'] = np.random.normal(100, 10, len(data))
        
        # Commodity prices (oil, gold)
        data['oil_price'] = np.random.normal(70, 15, len(data))
        data['gold_price'] = np.random.normal(1800, 100, len(data))
        
        return data
    
    def _add_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment-based features (simplified)"""
        
        # Simulate sentiment scores
        np.random.seed(42)
        
        # News sentiment (daily)
        data['news_sentiment'] = np.random.normal(0, 0.3, len(data))
        
        # Social media sentiment
        data['social_sentiment'] = np.random.normal(0, 0.4, len(data))
        
        # Analyst sentiment
        data['analyst_sentiment'] = np.random.normal(0, 0.2, len(data))
        
        # Fear & greed index
        data['fear_greed'] = np.random.normal(50, 20, len(data))
        
        return data
    
    def prepare_features_target(self, target_days: int = 1, feature_columns: list = None):
        """Prepare features and target for ML models"""
        
        if self.data is None:
            self.fetch_enhanced_data()
        
        # Define target (next N days return)
        self.data['target'] = (self.data['Close'].shift(-target_days) / self.data['Close'] - 1) * 100
        
        # Select features
        if feature_columns is None:
            # Exclude OHLCV and derived target columns
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'target']
            feature_columns = [col for col in self.data.columns if col not in exclude_cols]
        
        # Prepare features and target
        self.features = self.data[feature_columns].dropna()
        self.target = self.data.loc[self.features.index, 'target'].dropna()
        
        # Align features and target
        common_index = self.features.index.intersection(self.target.index)
        self.features = self.features.loc[common_index]
        self.target = self.target.loc[common_index]
        
        logger.info(f"Prepared {len(self.features)} samples with {len(feature_columns)} features")
        
        return self.features, self.target
    
    def train_xgboost_model(self, hyperparameter_tuning: bool = True) -> dict:
        """Train XGBoost model with optional hyperparameter tuning"""
        logger.info("Training XGBoost model...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if hyperparameter_tuning:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42)
            grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_scaled, y_train)
            
            best_model = grid_search.best_estimator_
            logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
        else:
            # Default parameters
            best_model = xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.9, random_state=42
            )
            best_model.fit(X_train_scaled, y_train)
        
        # Predictions and evaluation
        train_pred = best_model.predict(X_train_scaled)
        test_pred = best_model.predict(X_test_scaled)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Store model and scaler
        self.models['xgboost'] = best_model
        self.scalers['xgboost'] = scaler
        
        # Feature importance
        feature_importance = dict(zip(self.features.columns, best_model.feature_importances_))
        
        return {
            'model_type': 'XGBoost',
            'train_mse': train_mse,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'feature_importance': feature_importance,
            'best_params': grid_search.best_params_ if hyperparameter_tuning else None
        }
    
    def train_lstm_model(self, sequence_length: int = 60) -> dict:
        """Train LSTM model for time series prediction"""
        logger.info("Training LSTM model...")
        
        # Prepare sequences for LSTM
        X_sequences, y_sequences = self._prepare_sequences(sequence_length)
        
        # Train-test split
        split_idx = int(len(X_sequences) * 0.8)
        X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
        
        # Scale data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1]))
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Build LSTM model
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(25),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=100, batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Predictions
        train_pred_scaled = model.predict(X_train_scaled)
        test_pred_scaled = model.predict(X_test_scaled)
        
        # Inverse transform predictions
        train_pred = scaler_y.inverse_transform(train_pred_scaled).flatten()
        test_pred = scaler_y.inverse_transform(test_pred_scaled).flatten()
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Store model and scalers
        self.models['lstm'] = model
        self.scalers['lstm_X'] = scaler_X
        self.scalers['lstm_y'] = scaler_y
        
        return {
            'model_type': 'LSTM',
            'train_mse': train_mse,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'sequence_length': sequence_length,
            'training_epochs': len(history.history['loss'])
        }
    
    def _prepare_sequences(self, sequence_length: int) -> tuple:
        """Prepare sequences for LSTM training"""
        
        # Use normalized features for sequences
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(features_scaled)):
            X_sequences.append(features_scaled[i-sequence_length:i])
            y_sequences.append(self.target.iloc[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_ensemble_model(self) -> dict:
        """Train ensemble model combining multiple algorithms"""
        logger.info("Training ensemble model...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Individual models
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        cat_model = CatBoostRegressor(iterations=100, random_state=42, verbose=False)
        
        # Voting regressor
        ensemble_model = VotingRegressor([
            ('rf', rf_model),
            ('gb', gb_model),
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('cat', cat_model)
        ])
        
        # Train ensemble
        ensemble_model.fit(X_train_scaled, y_train)
        
        # Predictions
        train_pred = ensemble_model.predict(X_train_scaled)
        test_pred = ensemble_model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(ensemble_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        
        # Store model and scaler
        self.models['ensemble'] = ensemble_model
        self.scalers['ensemble'] = scaler
        
        return {
            'model_type': 'Ensemble',
            'train_mse': train_mse,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'cv_mean_score': -cv_scores.mean(),
            'cv_std_score': cv_scores.std(),
            'estimators': [name for name, _ in ensemble_model.named_estimators_.items()]
        }
    
    def predict_with_all_models(self, days_ahead: int = 1) -> dict:
        """Make predictions with all trained models"""
        predictions = {}
        
        # Get latest features
        latest_features = self.features.iloc[-1:].values
        
        # XGBoost prediction
        if 'xgboost' in self.models:
            scaler = self.scalers['xgboost']
            scaled_features = scaler.transform(latest_features)
            pred = self.models['xgboost'].predict(scaled_features)[0]
            predictions['xgboost'] = pred
        
        # Ensemble prediction
        if 'ensemble' in self.models:
            scaler = self.scalers['ensemble']
            scaled_features = scaler.transform(latest_features)
            pred = self.models['ensemble'].predict(scaled_features)[0]
            predictions['ensemble'] = pred
        
        # LSTM prediction (requires sequence)
        if 'lstm' in self.models:
            sequence_length = 60  # Default sequence length
            if len(self.features) >= sequence_length:
                scaler_X = self.scalers['lstm_X']
                scaler_y = self.scalers['lstm_y']
                
                # Get sequence
                sequence = self.features.iloc[-sequence_length:].values
                sequence_scaled = scaler_X.transform(sequence)
                sequence_scaled = sequence_scaled.reshape(1, sequence_length, -1)
                
                # Predict
                pred_scaled = self.models['lstm'].predict(sequence_scaled)
                pred = scaler_y.inverse_transform(pred_scaled)[0][0]
                predictions['lstm'] = pred
        
        # Calculate ensemble average
        if predictions:
            predictions['average'] = np.mean(list(predictions.values()))
        
        return predictions
    
    def save_models(self, path: str = './models/'):
        """Save trained models and scalers"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name == 'lstm':
                model.save(f"{path}/{self.ticker}_{model_name}_model.h5")
            else:
                joblib.dump(model, f"{path}/{self.ticker}_{model_name}_model.pkl")
        
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{path}/{self.ticker}_{scaler_name}_scaler.pkl")
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str = './models/'):
        """Load trained models and scalers"""
        import os
        from tensorflow.keras.models import load_model
        
        # Load models
        for model_type in ['xgboost', 'ensemble', 'lstm']:
            model_path = f"{path}/{self.ticker}_{model_type}_model"
            
            if model_type == 'lstm' and os.path.exists(f"{model_path}.h5"):
                self.models[model_type] = load_model(f"{model_path}.h5")
            elif os.path.exists(f"{model_path}.pkl"):
                self.models[model_type] = joblib.load(f"{model_path}.pkl")
        
        # Load scalers
        for scaler_file in os.listdir(path):
            if scaler_file.endswith('_scaler.pkl') and self.ticker in scaler_file:
                scaler_name = scaler_file.replace(f'{self.ticker}_', '').replace('_scaler.pkl', '')
                self.scalers[scaler_name] = joblib.load(f"{path}/{scaler_file}")
        
        logger.info(f"Models loaded from {path}")

def main():
    """Demonstrate advanced ML models"""
    print("=" * 60)
    print("ADVANCED ML MODELS FOR FINANCIAL ANALYSIS")
    print("=" * 60)
    
    # Initialize predictor
    ticker = "AAPL"
    predictor = AdvancedStockPredictor(ticker)
    
    # Fetch and prepare data
    predictor.fetch_enhanced_data()
    predictor.prepare_features_target(target_days=1)
    
    print(f"\nAnalyzing {ticker} with {len(predictor.features)} samples")
    print(f"Features: {len(predictor.features.columns)}")
    
    # Train models
    print("\n🚀 Training Models...")
    
    # XGBoost
    xgb_results = predictor.train_xgboost_model(hyperparameter_tuning=False)
    print(f"\nXGBoost Results:")
    print(f"  Test R² Score: {xgb_results['test_r2']:.4f}")
    print(f"  Test MAE: {xgb_results['test_mae']:.4f}")
    print(f"  Test MSE: {xgb_results['test_mse']:.4f}")
    
    # Ensemble
    ensemble_results = predictor.train_ensemble_model()
    print(f"\nEnsemble Results:")
    print(f"  Test R² Score: {ensemble_results['test_r2']:.4f}")
    print(f"  Test MAE: {ensemble_results['test_mae']:.4f}")
    print(f"  CV Score: {ensemble_results['cv_mean_score']:.4f} ± {ensemble_results['cv_std_score']:.4f}")
    
    # LSTM (commented out for faster demo)
    # lstm_results = predictor.train_lstm_model()
    # print(f"\nLSTM Results:")
    # print(f"  Test R² Score: {lstm_results['test_r2']:.4f}")
    # print(f"  Test MAE: {lstm_results['test_mae']:.4f}")
    
    # Make predictions
    print("\n📈 Making Predictions...")
    predictions = predictor.predict_with_all_models()
    
    for model_name, pred in predictions.items():
        print(f"  {model_name.title()}: {pred:.2f}% price change")
    
    # Feature importance (XGBoost)
    print("\n🎯 Top Features (XGBoost):")
    importance = xgb_results['feature_importance']
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for feature, importance_score in sorted_features:
        print(f"  {feature}: {importance_score:.4f}")

if __name__ == "__main__":
    main()

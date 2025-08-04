import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedQuarterlyStockPredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.prediction_days = 65  # ~3 months
        
    def get_current_price(self, symbol):
        """Get the most current price available"""
        try:
            stock = yf.Ticker(symbol)
            # Get the last few days to ensure we have the most recent data
            recent_data = stock.history(period="5d")
            if not recent_data.empty:
                current_price = recent_data['Close'].iloc[-1]
                last_date = recent_data.index[-1]
                return current_price, last_date
            else:
                raise ValueError("No recent data available")
        except Exception as e:
            print(f"Warning: Could not get current price for {symbol}: {e}")
            return None, None
    
    def calculate_quarterly_indicators(self, data):
        """Calculate indicators for quarterly predictions"""
        df = data.copy()
        
        # Long-term moving averages
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        df['MA_60'] = df['Close'].rolling(window=60).mean()
        
        # Quarterly momentum
        df['Price_Change_30'] = df['Close'].pct_change(periods=30)
        df['Price_Change_60'] = df['Close'].pct_change(periods=60)
        
        # Technical indicators
        df['RSI_30'] = self.calculate_rsi(df['Close'], 30)
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatility
        df['Volatility_30'] = df['Close'].pct_change().rolling(window=30).std()
        df['Volatility_60'] = df['Close'].pct_change().rolling(window=60).std()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Seasonal factors
        df['Quarter'] = df.index.quarter
        df['Month'] = df.index.month
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_quarterly_data(self, symbol, period="2y"):
        """Prepare data for 3-month predictions"""
        print(f"📊 Downloading {period} of data for quarterly analysis of {symbol}...")
        
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        print(f"✅ Downloaded {len(data)} days of data")
        print(f"📅 Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Calculate indicators
        data = self.calculate_quarterly_indicators(data)
        
        # Create target: price 3 months ahead
        data['Target'] = data['Close'].shift(-self.prediction_days)
        
        # Remove missing data
        data = data.dropna()
        
        if len(data) < 100:
            raise ValueError(f"Not enough data for quarterly analysis")
        
        # Select features
        features = [
            'Open', 'High', 'Low', 'Volume',
            'MA_10', 'MA_30', 'MA_60',
            'Price_Change_30', 'Price_Change_60',
            'RSI_30', 'MACD', 'BB_Position',
            'Volatility_30', 'Volatility_60',
            'Volume_Ratio', 'Quarter', 'Month'
        ]
        
        available_features = [f for f in features if f in data.columns]
        X = data[available_features]
        y = data['Target']
        
        return X, y, data, available_features
    
    def create_prediction_chart(self, symbol, historical_data, current_price, predicted_price, prediction_date, current_date):
        """Create a chart showing historical data and prediction"""
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot historical price data (last 6 months)
        recent_data = historical_data.tail(120)  # ~6 months
        ax.plot(recent_data.index, recent_data['Close'], 
                color='#2E8B57', linewidth=2, label='Historical Price')
        
        # Add moving averages
        if 'MA_30' in recent_data.columns:
            ax.plot(recent_data.index, recent_data['MA_30'], 
                    color='#FF6B6B', linewidth=1, alpha=0.7, label='30-Day MA')
        
        # Current price point
        ax.scatter([current_date], [current_price], 
                  color='#4169E1', s=100, zorder=5, label=f'Current Price (${current_price:.2f})')
        
        # Prediction point
        ax.scatter([prediction_date], [predicted_price], 
                  color='#DC143C', s=150, marker='^', zorder=5, 
                  label=f'3-Month Prediction (${predicted_price:.2f})')
        
        # Draw prediction line
        ax.plot([current_date, prediction_date], [current_price, predicted_price], 
                color='#DC143C', linewidth=2, linestyle='--', alpha=0.7)
        
        # Formatting
        ax.set_title(f'{symbol} Price Prediction: Current to 3-Month Forecast\n'
                    f'Current: ${current_price:.2f} → Predicted: ${predicted_price:.2f}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
        
        plt.tight_layout()
        return fig
    
    def calculate_quarterly_confidence(self, y_test, predictions, recent_volatility):
        """Calculate confidence for 3-month predictions"""
        # Price accuracy
        mape = mean_absolute_percentage_error(y_test, predictions)
        price_accuracy = max(0, (1 - mape) * 100)
        
        # Direction accuracy
        actual_changes = np.diff(y_test.values if hasattr(y_test, 'values') else y_test)
        pred_changes = np.diff(predictions)
        
        if len(actual_changes) > 0 and len(pred_changes) > 0:
            direction_accuracy = np.mean((actual_changes > 0) == (pred_changes > 0)) * 100
        else:
            direction_accuracy = 50
        
        # Volatility adjustment
        volatility_penalty = min(15, recent_volatility * 200)
        
        # Combined confidence (higher baseline for quarterly predictions)
        base_confidence = (price_accuracy * 0.6 + direction_accuracy * 0.4)
        final_confidence = max(55, min(90, base_confidence - volatility_penalty))
        
        return final_confidence, {
            'price_accuracy': price_accuracy,
            'direction_accuracy': direction_accuracy,
            'volatility_penalty': volatility_penalty
        }
    
    def train_and_predict(self, symbol):
        """Train model for 3-month prediction with enhanced output"""
        try:
            # Get current price first
            current_price, current_date = self.get_current_price(symbol)
            
            # Prepare historical data
            X, y, historical_data, features = self.prepare_quarterly_data(symbol)
            
            print(f"🔧 Training quarterly model with {len(features)} features")
            print(f"📅 Predicting price ~3 months ahead")
            
            # Use current price if available, otherwise use last historical price
            if current_price is not None:
                actual_current_price = current_price
                actual_current_date = current_date
                print(f"💰 Current price: ${actual_current_price:.2f} (as of {actual_current_date.strftime('%Y-%m-%d')})")
            else:
                actual_current_price = historical_data['Close'].iloc[-1]
                actual_current_date = historical_data.index[-1]
                print(f"💰 Using last available price: ${actual_current_price:.2f} (as of {actual_current_date.strftime('%Y-%m-%d')})")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            print("🤖 Training quarterly prediction model...")
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Calculate confidence
            test_predictions = self.model.predict(X_test)
            recent_volatility = historical_data['Volatility_60'].iloc[-30:].mean()
            confidence, confidence_details = self.calculate_quarterly_confidence(
                y_test, test_predictions, recent_volatility
            )
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Make prediction
            last_features = X_scaled[-1:].reshape(1, -1)
            predicted_price = self.model.predict(last_features)[0]
            
            # Calculate change based on actual current price
            change = predicted_price - actual_current_price
            change_percent = (change / actual_current_price) * 100
            
            # Calculate prediction date (3 months from current date)
            prediction_date = actual_current_date + timedelta(days=90)
            
            # Create prediction chart
            chart_fig = self.create_prediction_chart(
                symbol, historical_data, actual_current_price, predicted_price, 
                prediction_date, actual_current_date
            )
            
            return {
                'symbol': symbol,
                'current_price': round(actual_current_price, 2),
                'current_date': actual_current_date.strftime('%Y-%m-%d'),
                'predicted_price': round(predicted_price, 2),
                'prediction_date': prediction_date.strftime('%B %d, %Y'),
                'prediction_date_short': prediction_date.strftime('%b %Y'),
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'confidence': round(confidence, 1),
                'prediction_timeframe': '3 months',
                'model_accuracy': round(confidence_details['price_accuracy'], 1),
                'direction_accuracy': round(confidence_details['direction_accuracy'], 1),
                'data_points': len(historical_data),
                'top_features': feature_importance.head(3)['feature'].tolist(),
                'model_type': 'Quarterly Trend Prediction Model',
                'chart_figure': chart_fig,
                
                # Additional insights
                'trading_days_ahead': self.prediction_days,
                'data_freshness': f"Data current as of {actual_current_date.strftime('%B %d, %Y')}",
                'volatility_score': round(recent_volatility * 100, 1)
            }
            
        except Exception as e:
            raise Exception(f"Error in quarterly prediction for {symbol}: {str(e)}")

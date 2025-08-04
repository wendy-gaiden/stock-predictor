import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class QuarterlyStockPredictor:
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
        
        # Combined confidence
        base_confidence = (price_accuracy * 0.6 + direction_accuracy * 0.4)
        final_confidence = max(50, min(90, base_confidence - volatility_penalty))
        
        return final_confidence, {
            'price_accuracy': price_accuracy,
            'direction_accuracy': direction_accuracy,
            'volatility_penalty': volatility_penalty
        }
    
    def train_and_predict(self, symbol):
        """Train model for 3-month prediction"""
        try:
            # Prepare data
            X, y, data, features = self.prepare_quarterly_data(symbol)
            
            print(f"🔧 Training quarterly model with {len(features)} features")
            print(f"📅 Predicting price ~3 months ahead")
            
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
            recent_volatility = data['Volatility_60'].iloc[-30:].mean()
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
            
            current_price = data['Close'].iloc[-1]
            change = predicted_price - current_price
            change_percent = (change / current_price) * 100
            
            # Prediction date
            from datetime import datetime, timedelta
            prediction_date = datetime.now() + timedelta(days=90)
            
            return {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'predicted_price': round(predicted_price, 2),
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'confidence': round(confidence, 1),
                'prediction_timeframe': '3 months',
                'prediction_date': prediction_date.strftime('%B %Y'),
                'model_accuracy': round(confidence_details['price_accuracy'], 1),
                'direction_accuracy': round(confidence_details['direction_accuracy'], 1),
                'data_points': len(data),
                'top_features': feature_importance.head(3)['feature'].tolist(),
                'model_type': 'Quarterly Trend Prediction Model'
            }
            
        except Exception as e:
            raise Exception(f"Error in quarterly prediction for {symbol}: {str(e)}")

# Test function
def test_quarterly_predictor():
    """Test the quarterly predictor"""
    predictor = QuarterlyStockPredictor()
    
    try:
        result = predictor.train_and_predict("AAPL")
        print(f"✅ 3-Month Prediction for AAPL:")
        print(f"Current: ${result['current_price']}")
        print(f"Predicted ({result['prediction_date']}): ${result['predicted_price']}")
        print(f"Change: {result['change_percent']:+.1f}%")
        print(f"Confidence: {result['confidence']}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_quarterly_predictor()

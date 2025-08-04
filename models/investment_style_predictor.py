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

class InvestmentStylePredictor:
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
        
    def get_current_price_robust(self, symbol):
        """Get current price with multiple fallback methods"""
        try:
            stock = yf.Ticker(symbol)
            
            # Method 1: Try last 10 days to fill gaps
            recent_data = stock.history(period="10d")
            if not recent_data.empty:
                # Find the most recent non-null price
                valid_prices = recent_data['Close'].dropna()
                if not valid_prices.empty:
                    current_price = valid_prices.iloc[-1]
                    last_date = valid_prices.index[-1]
                    return current_price, last_date
            
            # Method 2: Try getting info (sometimes more current)
            info = stock.info
            if 'currentPrice' in info and info['currentPrice']:
                return info['currentPrice'], datetime.now()
            
            # Method 3: Use regularMarketPrice
            if 'regularMarketPrice' in info and info['regularMarketPrice']:
                return info['regularMarketPrice'], datetime.now()
                
            raise ValueError("No current price data available")
            
        except Exception as e:
            print(f"Warning: Could not get current price for {symbol}: {e}")
            return None, None
    
    def fill_price_gaps(self, data):
        """Fill gaps in price data using forward fill and interpolation"""
        # Forward fill first
        data = data.fillna(method='ffill')
        
        # Then interpolate any remaining gaps
        data = data.interpolate(method='time')
        
        return data
    
    def calculate_prediction_scenarios(self, base_prediction, historical_volatility, confidence_score):
        """Calculate best, average, worst case scenarios like analyst targets"""
        
        # Use historical volatility to determine scenario spread
        volatility_factor = historical_volatility * 0.5  # Scale down for 3-month period
        
        # Calculate scenarios based on confidence and volatility
        confidence_factor = confidence_score / 100
        
        # Best case: Base prediction + optimistic adjustment
        best_case_adjustment = volatility_factor * (1 + confidence_factor)
        best_case = base_prediction * (1 + best_case_adjustment)
        
        # Average case: Base prediction with slight confidence adjustment
        average_case = base_prediction * (1 + (volatility_factor * 0.2))
        
        # Worst case: Base prediction - pessimistic adjustment
        worst_case_adjustment = volatility_factor * (1 - confidence_factor * 0.5)
        worst_case = base_prediction * (1 - worst_case_adjustment)
        
        return {
            'best_case': best_case,
            'average_case': average_case,
            'worst_case': worst_case,
            'base_model': base_prediction
        }
    
    def calculate_quarterly_indicators(self, data):
        """Calculate indicators for quarterly predictions"""
        df = data.copy()
        
        # Fill any gaps in the data first
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            df[col] = self.fill_price_gaps(df[col])
        
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
    
    def create_investment_chart(self, symbol, historical_data, current_price, scenarios, prediction_date, current_date):
        """Create investment-style chart with scenarios"""
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
        
        # Top chart: Historical data and predictions
        recent_data = historical_data.tail(180)  # ~6 months, more data to fill gaps
        ax1.plot(recent_data.index, recent_data['Close'], 
                color='#2E8B57', linewidth=2, label='Historical Price', alpha=0.8)
        
        # Add moving average
        if 'MA_30' in recent_data.columns:
            ma_30_clean = recent_data['MA_30'].dropna()
            ax1.plot(ma_30_clean.index, ma_30_clean, 
                    color='#FF6B6B', linewidth=1, alpha=0.6, label='30-Day MA')
        
        # Current price point
        ax1.scatter([current_date], [current_price], 
                  color='#4169E1', s=120, zorder=5, label=f'Current: ${current_price:.2f}')
        
        # Prediction scenarios
        prediction_x = [prediction_date] * 4
        prediction_prices = [
            scenarios['best_case'],
            scenarios['average_case'], 
            scenarios['worst_case'],
            scenarios['base_model']
        ]
        
        colors = ['#28a745', '#007bff', '#dc3545', '#6c757d']
        labels = ['Best Case', 'Average Case', 'Worst Case', 'Base Model']
        markers = ['^', 's', 'v', 'o']
        
        for i, (price, color, label, marker) in enumerate(zip(prediction_prices, colors, labels, markers)):
            ax1.scatter(prediction_x[i], price, color=color, s=100, 
                       marker=marker, zorder=5, label=f'{label}: ${price:.2f}')
            
            # Draw prediction lines
            ax1.plot([current_date, prediction_date], [current_price, price], 
                    color=color, linewidth=1.5, linestyle='--', alpha=0.7)
        
        # Formatting top chart
        ax1.set_title(f'{symbol} Investment Analysis: 3-Month Price Targets\n'
                     f'Current: ${current_price:.2f} | Target Date: {prediction_date.strftime("%B %Y")}', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
        
        # Bottom chart: Scenario comparison bar chart
        scenario_names = ['Worst Case\n(Bear)', 'Average Case\n(Neutral)', 'Best Case\n(Bull)']
        scenario_values = [scenarios['worst_case'], scenarios['average_case'], scenarios['best_case']]
        scenario_changes = [(p - current_price) / current_price * 100 for p in scenario_values]
        
        bars = ax2.bar(scenario_names, scenario_values, 
                      color=['#dc3545', '#007bff', '#28a745'], alpha=0.7)
        
        # Add percentage labels on bars
        for bar, change in zip(bars, scenario_changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'${height:.2f}\n({change:+.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add current price line
        ax2.axhline(y=current_price, color='#4169E1', linestyle='-', 
                   linewidth=2, alpha=0.8, label=f'Current: ${current_price:.2f}')
        
        ax2.set_title('3-Month Price Target Scenarios', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Target Price ($)', fontsize=10)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def prepare_quarterly_data(self, symbol, period="2y"):
        """Prepare data for 3-month predictions with gap filling"""
        print(f"📊 Downloading {period} of data for quarterly analysis of {symbol}...")
        
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        print(f"✅ Downloaded {len(data)} days of data")
        print(f"📅 Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Fill gaps in data
        data = self.fill_price_gaps(data)
        
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
    
    def calculate_confidence_and_risk(self, y_test, predictions, recent_volatility):
        """Calculate confidence and risk metrics"""
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
        
        # Risk assessment
        risk_score = min(100, recent_volatility * 300)  # Convert to 0-100 scale
        
        # Combined confidence (higher baseline for quarterly predictions)
        base_confidence = (price_accuracy * 0.6 + direction_accuracy * 0.4)
        final_confidence = max(55, min(85, base_confidence - (risk_score * 0.1)))
        
        return final_confidence, {
            'price_accuracy': price_accuracy,
            'direction_accuracy': direction_accuracy,
            'risk_score': risk_score,
            'volatility': recent_volatility
        }
    
    def train_and_predict(self, symbol):
        """Train model and generate investment-style scenarios"""
        try:
            # Get current price with robust method
            current_price, current_date = self.get_current_price_robust(symbol)
            
            # Prepare historical data
            X, y, historical_data, features = self.prepare_quarterly_data(symbol)
            
            print(f"🔧 Training investment model with {len(features)} features")
            print(f"📅 Generating 3-month price targets")
            
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
            
            print("🤖 Training investment prediction model...")
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Calculate confidence and risk
            test_predictions = self.model.predict(X_test)
            recent_volatility = historical_data['Volatility_60'].iloc[-30:].mean()
            confidence, risk_metrics = self.calculate_confidence_and_risk(
                y_test, test_predictions, recent_volatility
            )
            
            # Make base prediction
            last_features = X_scaled[-1:].reshape(1, -1)
            base_prediction = self.model.predict(last_features)[0]
            
            # Generate investment scenarios
            scenarios = self.calculate_prediction_scenarios(
                base_prediction, recent_volatility, confidence
            )
            
            # Calculate prediction date (3 months from current date)
            prediction_date = actual_current_date + timedelta(days=90)
            
            # Create investment chart
            chart_fig = self.create_investment_chart(
                symbol, historical_data, actual_current_price, scenarios, 
                prediction_date, actual_current_date
            )
            
            # Calculate scenario changes
            scenario_changes = {
                'best_case_change': (scenarios['best_case'] - actual_current_price) / actual_current_price * 100,
                'average_case_change': (scenarios['average_case'] - actual_current_price) / actual_current_price * 100,
                'worst_case_change': (scenarios['worst_case'] - actual_current_price) / actual_current_price * 100
            }
            
            return {
                'symbol': symbol,
                'current_price': round(actual_current_price, 2),
                'current_date': actual_current_date.strftime('%Y-%m-%d'),
                'prediction_date': prediction_date.strftime('%B %d, %Y'),
                'prediction_date_short': prediction_date.strftime('%b %Y'),
                
                # Investment scenarios
                'best_case_price': round(scenarios['best_case'], 2),
                'average_case_price': round(scenarios['average_case'], 2),
                'worst_case_price': round(scenarios['worst_case'], 2),
                'base_model_price': round(scenarios['base_model'], 2),
                
                'best_case_change': round(scenario_changes['best_case_change'], 2),
                'average_case_change': round(scenario_changes['average_case_change'], 2),
                'worst_case_change': round(scenario_changes['worst_case_change'], 2),
                
                'confidence': round(confidence, 1),
                'risk_score': round(risk_metrics['risk_score'], 1),
                'data_points': len(historical_data),
                'model_type': 'Investment Analysis Model',
                'chart_figure': chart_fig,
                
                # Risk assessment
                'risk_level': self._get_risk_level(risk_metrics['risk_score']),
                'data_freshness': f"Data current as of {actual_current_date.strftime('%B %d, %Y')}",
                'volatility_score': round(recent_volatility * 100, 1),
                
                # Investment insights
                'upside_potential': round(scenario_changes['best_case_change'], 1),
                'downside_risk': round(abs(scenario_changes['worst_case_change']), 1),
                'expected_return': round(scenario_changes['average_case_change'], 1)
            }
            
        except Exception as e:
            raise Exception(f"Error in investment analysis for {symbol}: {str(e)}")
    
    def _get_risk_level(self, risk_score):
        """Convert risk score to descriptive level"""
        if risk_score < 20:
            return "Low Risk"
        elif risk_score < 40:
            return "Moderate Risk"
        elif risk_score < 60:
            return "High Risk"
        else:
            return "Very High Risk"

# Test function
def test_investment_predictor():
    """Test the investment-style predictor"""
    predictor = InvestmentStylePredictor()
    
    try:
        result = predictor.train_and_predict("AAPL")
        print(f"✅ Investment Analysis for AAPL:")
        print(f"Current: ${result['current_price']} (as of {result['current_date']})")
        print(f"Target Date: {result['prediction_date_short']}")
        print(f"Best Case: ${result['best_case_price']} ({result['best_case_change']:+.1f}%)")
        print(f"Average Case: ${result['average_case_price']} ({result['average_case_change']:+.1f}%)")
        print(f"Worst Case: ${result['worst_case_price']} ({result['worst_case_change']:+.1f}%)")
        print(f"Confidence: {result['confidence']}% | Risk: {result['risk_level']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_investment_predictor()
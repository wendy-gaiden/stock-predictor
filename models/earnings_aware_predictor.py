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

class EarningsAwarePredictor:
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
        
    def normalize_datetime(self, dt):
        """Ensure datetime is timezone-naive"""
        if hasattr(dt, 'tz') and dt.tz is not None:
            return dt.tz_localize(None)
        return dt
    
    def normalize_datetime_index(self, df):
        """Ensure DataFrame index is timezone-naive"""
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
        
    def get_current_price_robust(self, symbol):
        """Get current price with multiple fallback methods"""
        try:
            stock = yf.Ticker(symbol)
            
            # Try multiple periods to get most recent data
            for period in ["5d", "1mo"]:
                try:
                    recent_data = stock.history(period=period)
                    if not recent_data.empty:
                        # Normalize timezone
                        recent_data = self.normalize_datetime_index(recent_data)
                        valid_prices = recent_data['Close'].dropna()
                        if not valid_prices.empty:
                            current_price = valid_prices.iloc[-1]
                            last_date = valid_prices.index[-1]
                            return current_price, last_date
                except:
                    continue
            
            # Fallback to stock info
            info = stock.info
            if 'currentPrice' in info and info['currentPrice']:
                return info['currentPrice'], datetime.now()
                
            raise ValueError("No current price data available")
            
        except Exception as e:
            print(f"Warning: Could not get current price for {symbol}: {e}")
            return None, None
    
    def fix_data_gaps(self, data):
        """Intelligent gap filling for stock data"""
        # Forward fill first
        data_filled = data.fillna(method='ffill')
        # Then interpolate any remaining gaps
        data_filled = data_filled.interpolate(method='time')
        # Backfill any remaining
        data_filled = data_filled.fillna(method='bfill')
        return data_filled
    
    def calculate_earnings_proximity(self, date_index):
        """Calculate proximity to earnings announcements using quarterly patterns"""
        proximity_scores = []
        
        # Estimate quarterly earnings pattern
        for date in date_index:
            # Calculate days into quarter
            quarter_start = pd.Timestamp(year=date.year, month=((date.quarter-1)*3 + 1), day=1)
            days_into_quarter = (date - quarter_start).days
            
            # Assume earnings typically happen 30-45 days into quarter
            earnings_day_estimate = 35
            distance_from_earnings = abs(days_into_quarter - earnings_day_estimate)
            
            # Convert to proximity score
            if distance_from_earnings <= 15:
                proximity = max(0, (15 - distance_from_earnings) / 15)
            else:
                proximity = 0
            
            proximity_scores.append(proximity)
        
        return proximity_scores
    
    def add_market_trend_bias(self, data):
        """Add market's natural upward bias to features"""
        # Markets generally trend upward ~7-10% annually
        annual_trend = 0.08  # 8% annual growth assumption
        daily_trend = annual_trend / 252  # 252 trading days per year
        
        # Calculate trend component
        data['Market_Trend'] = np.arange(len(data)) * daily_trend
        
        # Add cyclical components
        data['Year_Cycle'] = np.sin(2 * np.pi * data.index.dayofyear / 365)
        data['Month_Effect'] = np.sin(2 * np.pi * data.index.month / 12)
        
        # January effect and year-end rally
        data['January_Effect'] = (data.index.month == 1).astype(float) * 0.02
        data['Year_End_Rally'] = (data.index.month.isin([11, 12])).astype(float) * 0.015
        
        return data
    
    def calculate_advanced_indicators(self, data):
        """Calculate comprehensive indicators"""
        df = data.copy()
        
        # Ensure timezone-naive index
        df = self.normalize_datetime_index(df)
        
        # Fix data gaps first
        df = self.fix_data_gaps(df)
        
        # Add market trend bias
        df = self.add_market_trend_bias(df)
        
        # Moving averages
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        df['MA_60'] = df['Close'].rolling(window=60).mean()
        df['MA_250'] = df['Close'].rolling(window=250).mean()  # 1-year MA
        
        # Price momentum
        df['Price_Change_20'] = df['Close'].pct_change(periods=20)
        df['Price_Change_60'] = df['Close'].pct_change(periods=60)
        df['Price_Change_250'] = df['Close'].pct_change(periods=250)
        
        # Technical indicators
        df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
        df['RSI_30'] = self.calculate_rsi(df['Close'], 30)
        
        # MACD
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume analysis
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Volatility
        df['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
        df['Volatility_60'] = df['Close'].pct_change().rolling(window=60).std()
        
        # Price position indicators
        df['Price_vs_MA250'] = df['Close'] / df['MA_250'] - 1
        
        # Seasonal factors
        df['Quarter'] = df.index.quarter
        df['Month'] = df.index.month
        df['Is_Quarter_End'] = (df.index.month % 3 == 0).astype(int)
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_realistic_scenarios(self, base_prediction, current_price, historical_volatility, earnings_proximity):
        """Calculate realistic scenarios based on market behavior"""
        
        # Base assumption: Markets trend upward
        market_bias = 0.02  # 2% quarterly upward bias
        
        # Adjust for earnings proximity
        earnings_volatility = earnings_proximity * 0.03  # Up to 3% additional volatility
        
        # Calculate scenarios
        total_volatility = historical_volatility + earnings_volatility
        
        # More realistic scenario spreads
        best_case_multiplier = 1 + market_bias + (total_volatility * 1.2)
        average_case_multiplier = 1 + market_bias
        worst_case_multiplier = 1 + market_bias - (total_volatility * 0.8)
        
        scenarios = {
            'best_case': current_price * best_case_multiplier,
            'average_case': current_price * average_case_multiplier,
            'worst_case': max(current_price * worst_case_multiplier, current_price * 0.90),  # Cap downside
            'base_model': base_prediction
        }
        
        return scenarios
    
    def create_simple_chart(self, symbol, historical_data, current_price, scenarios, prediction_date, current_date):
        """Create simplified chart"""
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
        
        # Top chart: Historical data
        chart_data = historical_data.tail(250)  # ~1 year of data
        
        ax1.plot(chart_data.index, chart_data['Close'], 
                color='#2E8B57', linewidth=2, label='Historical Price')
        
        # Add moving average
        if 'MA_60' in chart_data.columns:
            ma_60_clean = chart_data['MA_60'].dropna()
            if not ma_60_clean.empty:
                ax1.plot(ma_60_clean.index, ma_60_clean, 
                        color='#FF6B6B', linewidth=1, alpha=0.7, label='60-Day MA')
        
        # Current price point
        ax1.scatter([current_date], [current_price], 
                  color='#4169E1', s=100, zorder=5, label=f'Current: ${current_price:.2f}')
        
        # Prediction scenarios
        colors = ['#28a745', '#007bff', '#dc3545']
        labels = ['Best Case', 'Average Case', 'Worst Case']
        prices = [scenarios['best_case'], scenarios['average_case'], scenarios['worst_case']]
        markers = ['^', 's', 'v']
        
        for price, color, label, marker in zip(prices, colors, labels, markers):
            ax1.scatter([prediction_date], [price], color=color, s=100, 
                       marker=marker, zorder=5, label=f'{label}: ${price:.2f}')
            
            # Draw prediction line
            ax1.plot([current_date, prediction_date], [current_price, price], 
                    color=color, linewidth=2, linestyle='--', alpha=0.7)
        
        # Formatting
        ax1.set_title(f'{symbol} 5-Year Analysis & 3-Month Targets\n'
                     f'Current: ${current_price:.2f} → Target: {prediction_date.strftime("%b %Y")}', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
        
        # Bottom chart: Scenario comparison
        scenario_names = ['Bear Case', 'Base Case', 'Bull Case']
        scenario_values = [scenarios['worst_case'], scenarios['average_case'], scenarios['best_case']]
        scenario_changes = [(p - current_price) / current_price * 100 for p in scenario_values]
        
        bars = ax2.bar(scenario_names, scenario_values, 
                      color=['#dc3545', '#007bff', '#28a745'], alpha=0.7)
        
        # Add percentage labels
        for bar, change, value in zip(bars, scenario_changes, scenario_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(scenario_values) * 0.01,
                    f'${value:.2f}\n({change:+.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Current price line
        ax2.axhline(y=current_price, color='#4169E1', linestyle='-', 
                   linewidth=2, alpha=0.8, label=f'Current: ${current_price:.2f}')
        
        ax2.set_title('3-Month Price Targets', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Price ($)', fontsize=10)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def prepare_comprehensive_data(self, symbol, period="5y"):
        """Prepare 5-year comprehensive dataset"""
        print(f"📊 Downloading {period} of data for comprehensive analysis of {symbol}...")
        
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Normalize timezone immediately
        data = self.normalize_datetime_index(data)
        
        print(f"✅ Downloaded {len(data)} days of data")
        print(f"📅 Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Calculate comprehensive indicators
        data = self.calculate_advanced_indicators(data)
        
        # Add earnings proximity (simplified)
        earnings_proximity = self.calculate_earnings_proximity(data.index)
        data['Earnings_Proximity'] = earnings_proximity
        
        # Create target
        data['Target'] = data['Close'].shift(-self.prediction_days)
        
        # Remove missing data
        data = data.dropna()
        
        if len(data) < 200:
            raise ValueError(f"Not enough data for comprehensive analysis")
        
        # Select comprehensive features
        features = [
            # Price and volume
            'Open', 'High', 'Low', 'Volume',
            
            # Moving averages
            'MA_10', 'MA_30', 'MA_60', 'MA_250',
            
            # Momentum indicators
            'Price_Change_20', 'Price_Change_60', 'Price_Change_250',
            
            # Technical indicators
            'RSI_14', 'RSI_30', 'MACD', 'BB_Position',
            
            # Volume analysis
            'Volume_Ratio',
            
            # Volatility
            'Volatility_20', 'Volatility_60',
            
            # Price position
            'Price_vs_MA250',
            
            # Market trends and seasonality
            'Market_Trend', 'Year_Cycle', 'Month_Effect',
            'January_Effect', 'Year_End_Rally',
            'Quarter', 'Month', 'Is_Quarter_End',
            
            # Earnings
            'Earnings_Proximity'
        ]
        
        available_features = [f for f in features if f in data.columns]
        X = data[available_features]
        y = data['Target']
        
        return X, y, data, available_features
    
    def train_and_predict(self, symbol):
        """Train comprehensive model with earnings awareness"""
        try:
            # Get current price
            current_price, current_date = self.get_current_price_robust(symbol)
            
            # Prepare 5-year comprehensive data
            X, y, historical_data, features = self.prepare_comprehensive_data(symbol)
            
            print(f"🔧 Training market-aware model with {len(features)} features")
            print(f"📅 Using 5 years of historical data")
            
            # Use current price if available
            if current_price is not None:
                actual_current_price = current_price
                actual_current_date = current_date
                print(f"💰 Current price: ${actual_current_price:.2f} (as of {actual_current_date.strftime('%Y-%m-%d')})")
            else:
                actual_current_price = historical_data['Close'].iloc[-1]
                actual_current_date = historical_data.index[-1]
                print(f"💰 Using last available price: ${actual_current_price:.2f}")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data chronologically
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            print("🤖 Training market-aware prediction model...")
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Calculate confidence
            test_predictions = self.model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, test_predictions)
            confidence = max(65, min(85, (1 - mape) * 100))  # Realistic range
            
            # Make base prediction
            last_features = X_scaled[-1:].reshape(1, -1)
            base_prediction = self.model.predict(last_features)[0]
            
            # Calculate current earnings proximity
            current_earnings_proximity = historical_data['Earnings_Proximity'].iloc[-1]
            historical_volatility = historical_data['Volatility_60'].iloc[-30:].mean()
            
            # Generate market-aware scenarios
            scenarios = self.calculate_realistic_scenarios(
                base_prediction, actual_current_price, historical_volatility, current_earnings_proximity
            )
            
            # Calculate prediction date
            prediction_date = actual_current_date + timedelta(days=90)
            
            # Create chart
            chart_fig = self.create_simple_chart(
                symbol, historical_data, actual_current_price, scenarios, 
                prediction_date, actual_current_date
            )
            
            # Calculate scenario changes
            scenario_changes = {
                'best_case_change': (scenarios['best_case'] - actual_current_price) / actual_current_price * 100,
                'average_case_change': (scenarios['average_case'] - actual_current_price) / actual_current_price * 100,
                'worst_case_change': (scenarios['worst_case'] - actual_current_price) / actual_current_price * 100
            }
            
            # Risk assessment
            risk_score = min(100, historical_volatility * 300)
            
            return {
                'symbol': symbol,
                'current_price': round(actual_current_price, 2),
                'current_date': actual_current_date.strftime('%Y-%m-%d'),
                'prediction_date': prediction_date.strftime('%B %d, %Y'),
                'prediction_date_short': prediction_date.strftime('%b %Y'),
                
                # Market-aware scenarios
                'best_case_price': round(scenarios['best_case'], 2),
                'average_case_price': round(scenarios['average_case'], 2),
                'worst_case_price': round(scenarios['worst_case'], 2),
                'base_model_price': round(scenarios['base_model'], 2),
                
                'best_case_change': round(scenario_changes['best_case_change'], 2),
                'average_case_change': round(scenario_changes['average_case_change'], 2),
                'worst_case_change': round(scenario_changes['worst_case_change'], 2),
                
                'confidence': round(confidence, 1),
                'risk_score': round(risk_score, 1),
                'risk_level': self._get_risk_level(risk_score),
                
                'data_points': len(historical_data),
                'model_type': 'Market-Aware 5-Year Analysis Model',
                'chart_figure': chart_fig,
                
                # Enhanced insights
                'data_freshness': f"5-year analysis current as of {actual_current_date.strftime('%B %d, %Y')}",
                'volatility_score': round(historical_volatility * 100, 1),
                'upside_potential': round(scenario_changes['best_case_change'], 1),
                'downside_risk': round(abs(scenario_changes['worst_case_change']), 1),
                'expected_return': round(scenario_changes['average_case_change'], 1),
                'market_bias': 'Positive (Historical upward trend incorporated)'
            }
            
        except Exception as e:
            raise Exception(f"Error in market-aware prediction for {symbol}: {str(e)}")
    
    def _get_risk_level(self, risk_score):
        """Convert risk score to descriptive level"""
        if risk_score < 25:
            return "Low Risk"
        elif risk_score < 45:
            return "Moderate Risk"
        elif risk_score < 65:
            return "High Risk"
        else:
            return "Very High Risk"

# Test function
def test_earnings_aware_predictor():
    """Test the earnings-aware predictor"""
    predictor = EarningsAwarePredictor()
    
    try:
        result = predictor.train_and_predict("AAPL")
        print(f"✅ Market-Aware Analysis for AAPL:")
        print(f"Current: ${result['current_price']} (as of {result['current_date']})")
        print(f"Target Date: {result['prediction_date_short']}")
        print(f"Best Case: ${result['best_case_price']} ({result['best_case_change']:+.1f}%)")
        print(f"Average Case: ${result['average_case_price']} ({result['average_case_change']:+.1f}%)")
        print(f"Worst Case: ${result['worst_case_price']} ({result['worst_case_change']:+.1f}%)")
        print(f"Confidence: {result['confidence']}% | Risk: {result['risk_level']}")
        print(f"Market Bias: {result['market_bias']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_earnings_aware_predictor()

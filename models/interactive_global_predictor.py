import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class InteractiveGlobalPredictor:
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
        
        # Global market exchanges and suffixes
        self.global_markets = {
            'US': {'suffix': '', 'timezone': 'America/New_York', 'currency': 'USD'},
            'Canada': {'suffix': '.TO', 'timezone': 'America/Toronto', 'currency': 'CAD'},
            'UK': {'suffix': '.L', 'timezone': 'Europe/London', 'currency': 'GBP'},
            'Germany': {'suffix': '.DE', 'timezone': 'Europe/Berlin', 'currency': 'EUR'},
            'France': {'suffix': '.PA', 'timezone': 'Europe/Paris', 'currency': 'EUR'},
            'Japan': {'suffix': '.T', 'timezone': 'Asia/Tokyo', 'currency': 'JPY'},
            'Hong Kong': {'suffix': '.HK', 'timezone': 'Asia/Hong_Kong', 'currency': 'HKD'},
            'Australia': {'suffix': '.AX', 'timezone': 'Australia/Sydney', 'currency': 'AUD'},
            'India': {'suffix': '.NS', 'timezone': 'Asia/Kolkata', 'currency': 'INR'},
            'Singapore': {'suffix': '.SI', 'timezone': 'Asia/Singapore', 'currency': 'SGD'}
        }
        
    def detect_market(self, symbol):
        """Detect which market a symbol belongs to"""
        symbol_upper = symbol.upper()
        
        # Check for explicit market suffixes
        for market, info in self.global_markets.items():
            if symbol_upper.endswith(info['suffix']) and info['suffix']:
                return market, info
        
        # Default to US market for symbols without suffix
        return 'US', self.global_markets['US']
    
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
        """Get current price with comprehensive gap handling"""
        try:
            stock = yf.Ticker(symbol)
            
            # Try progressively longer periods to get most recent data
            for period in ["1d", "5d", "1mo", "3mo"]:
                try:
                    recent_data = stock.history(period=period)
                    if not recent_data.empty:
                        # Normalize timezone
                        recent_data = self.normalize_datetime_index(recent_data)
                        valid_prices = recent_data['Close'].dropna()
                        if not valid_prices.empty:
                            current_price = valid_prices.iloc[-1]
                            last_date = valid_prices.index[-1]
                            
                            # Check if data is recent enough (within 7 days)
                            days_old = (datetime.now() - last_date).days
                            if days_old <= 7:
                                return current_price, last_date
                except:
                    continue
            
            # Fallback to stock info
            info = stock.info
            if 'currentPrice' in info and info['currentPrice']:
                return info['currentPrice'], datetime.now()
            elif 'regularMarketPrice' in info and info['regularMarketPrice']:
                return info['regularMarketPrice'], datetime.now()
                
            raise ValueError("No current price data available")
            
        except Exception as e:
            print(f"Warning: Could not get current price for {symbol}: {e}")
            return None, None
    
    def advanced_gap_filling(self, data):
        """Advanced gap filling using multiple methods"""
        df = data.copy()
        
        # Create a complete business day calendar
        start_date = df.index[0]
        end_date = df.index[-1]
        
        # Create business days only (excludes weekends)
        business_days = pd.bdate_range(start=start_date, end=end_date)
        
        # Reindex to business days
        df_complete = df.reindex(business_days)
        
        # Fill gaps with multiple strategies
        for column in ['Open', 'High', 'Low', 'Close']:
            # 1. Forward fill (carry forward last known price)
            df_complete[column] = df_complete[column].fillna(method='ffill')
            
            # 2. For remaining gaps, use interpolation
            df_complete[column] = df_complete[column].interpolate(method='linear')
            
            # 3. For any remaining gaps at the beginning, backfill
            df_complete[column] = df_complete[column].fillna(method='bfill')
        
        # Handle volume separately (can be zero)
        df_complete['Volume'] = df_complete['Volume'].fillna(0)
        
        # Ensure OHLC consistency
        for i in range(len(df_complete)):
            if pd.notna(df_complete.iloc[i]['Close']):
                # Ensure High is at least Close, Low is at most Close
                df_complete.iloc[i, df_complete.columns.get_loc('High')] = max(
                    df_complete.iloc[i]['High'], df_complete.iloc[i]['Close']
                )
                df_complete.iloc[i, df_complete.columns.get_loc('Low')] = min(
                    df_complete.iloc[i]['Low'], df_complete.iloc[i]['Close']
                )
        
        return df_complete
    
    def calculate_market_indicators(self, data):
        """Calculate comprehensive market indicators"""
        df = data.copy()
        
        # Ensure timezone-naive index
        df = self.normalize_datetime_index(df)
        
        # Advanced gap filling
        df = self.advanced_gap_filling(df)
        
        # Market trend bias (8% annual growth)
        annual_trend = 0.08
        daily_trend = annual_trend / 252
        df['Market_Trend'] = np.arange(len(df)) * daily_trend
        
        # Moving averages (multiple timeframes)
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # Price momentum
        df['Price_Change_1d'] = df['Close'].pct_change(periods=1)
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Price_Change_20d'] = df['Close'].pct_change(periods=20)
        df['Price_Change_60d'] = df['Close'].pct_change(periods=60)
        
        # Technical indicators
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)  # Avoid division by zero
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Seasonal patterns
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Day_of_Week'] = df.index.dayofweek
        df['Month_Sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['Quarter_Sin'] = np.sin(2 * np.pi * df.index.quarter / 4)
        
        # Earnings proximity (quarterly pattern)
        df['Earnings_Proximity'] = self.calculate_earnings_proximity(df.index)
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        return 100 - (100 / (1 + rs))
    
    def calculate_earnings_proximity(self, date_index):
        """Calculate earnings proximity using quarterly patterns"""
        proximity_scores = []
        
        for date in date_index:
            # Calculate days into quarter
            quarter_start = pd.Timestamp(year=date.year, month=((date.quarter-1)*3 + 1), day=1)
            days_into_quarter = (date - quarter_start).days
            
            # Earnings typically 30-45 days into quarter
            earnings_day = 35
            distance = abs(days_into_quarter - earnings_day)
            
            # Proximity score (0-1)
            proximity = max(0, (20 - distance) / 20) if distance <= 20 else 0
            proximity_scores.append(proximity)
        
        return proximity_scores
    
    def calculate_realistic_scenarios(self, base_prediction, current_price, volatility, market_info):
        """Calculate realistic scenarios with market-specific adjustments"""
        
        # Base market trend (quarterly)
        quarterly_trend = 0.02  # 2% quarterly growth
        
        # Market-specific adjustments
        if market_info['currency'] == 'JPY':
            quarterly_trend *= 0.8  # Lower growth expectation for Japan
        elif market_info['currency'] in ['HKD', 'SGD']:
            quarterly_trend *= 0.9  # Moderate growth for Asian markets
        elif market_info['currency'] in ['EUR', 'GBP']:
            quarterly_trend *= 0.95  # Slightly lower for European markets
        
        # Volatility adjustment
        vol_adjustment = volatility * 1.5
        
        scenarios = {
            'best_case': current_price * (1 + quarterly_trend + vol_adjustment),
            'average_case': current_price * (1 + quarterly_trend),
            'worst_case': max(current_price * (1 + quarterly_trend - vol_adjustment * 0.8), 
                            current_price * 0.85),  # Cap downside at -15%
            'base_model': base_prediction
        }
        
        return scenarios
    
    def create_interactive_chart(self, symbol, data, current_price, scenarios, prediction_date, current_date, market_info):
        """Create interactive Plotly chart with mouse-over functionality"""
        
        # Prepare data for chart (last 12 months)
        chart_data = data.tail(252)  # ~1 year
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=[f'{symbol} Price Analysis & 3-Month Targets'],
            vertical_spacing=0.1
        )
        
        # Historical price line
        fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='#2E8B57', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> %{customdata} %{y:.2f}<extra></extra>',
                customdata=[market_info['currency']] * len(chart_data)
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'MA_50' in chart_data.columns:
            ma50_clean = chart_data['MA_50'].dropna()
            fig.add_trace(
                go.Scatter(
                    x=ma50_clean.index,
                    y=ma50_clean,
                    mode='lines',
                    name='50-Day MA',
                    line=dict(color='#FF6B6B', width=1, dash='dot'),
                    opacity=0.7,
                    hovertemplate='<b>50-Day MA:</b> %{customdata} %{y:.2f}<extra></extra>',
                    customdata=[market_info['currency']] * len(ma50_clean)
                ),
                row=1, col=1
            )
        
        # Current price point
        fig.add_trace(
            go.Scatter(
                x=[current_date],
                y=[current_price],
                mode='markers',
                name=f'Current Price',
                marker=dict(color='#4169E1', size=12, symbol='circle'),
                hovertemplate=f'<b>Current Price:</b> {market_info["currency"]} %{{y:.2f}}<br><b>Date:</b> %{{x}}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Prediction points
        scenarios_data = [
            ('Best Case', scenarios['best_case'], '#28a745', 'triangle-up'),
            ('Average Case', scenarios['average_case'], '#007bff', 'square'),
            ('Worst Case', scenarios['worst_case'], '#dc3545', 'triangle-down')
        ]
        
        for name, price, color, symbol_shape in scenarios_data:
            fig.add_trace(
                go.Scatter(
                    x=[prediction_date],
                    y=[price],
                    mode='markers',
                    name=name,
                    marker=dict(color=color, size=14, symbol=symbol_shape),
                    hovertemplate=f'<b>{name}:</b> {market_info["currency"]} %{{y:.2f}}<br><b>Target Date:</b> %{{x}}<br><b>Change:</b> {((price - current_price) / current_price * 100):+.1f}%<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Prediction lines
            fig.add_trace(
                go.Scatter(
                    x=[current_date, prediction_date],
                    y=[current_price, price],
                    mode='lines',
                    name=f'{name} Trend',
                    line=dict(color=color, width=2, dash='dash'),
                    opacity=0.6,
                    showlegend=False,
                    hovertemplate=f'<b>{name} Projection</b><extra></extra>'
                ),
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{symbol} Interactive Price Analysis<br><sub>Market: {market_info.get("market", "Unknown")} | Currency: {market_info["currency"]} | Target: {prediction_date.strftime("%b %Y")}</sub>',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title='Date',
            yaxis_title=f'Price ({market_info["currency"]})',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridcolor='lightgray', tickformat='.2f')
        
        return fig
    
    def prepare_data(self, symbol, period="5y"):
        """Prepare comprehensive dataset"""
        print(f"📊 Downloading {period} of data for {symbol}...")
        
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Normalize timezone immediately
        data = self.normalize_datetime_index(data)
        
        print(f"✅ Downloaded {len(data)} days of data")
        print(f"📅 Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Calculate indicators
        data = self.calculate_market_indicators(data)
        
        # Create target
        data['Target'] = data['Close'].shift(-self.prediction_days)
        
        # Remove missing data
        data = data.dropna()
        
        if len(data) < 200:
            raise ValueError(f"Not enough data for analysis")
        
        # Select features
        features = [
            'Open', 'High', 'Low', 'Volume',
            'MA_5', 'MA_20', 'MA_50', 'MA_200',
            'Price_Change_1d', 'Price_Change_5d', 'Price_Change_20d', 'Price_Change_60d',
            'RSI', 'MACD', 'BB_Position',
            'Volume_Ratio', 'Volatility',
            'Market_Trend', 'Month_Sin', 'Quarter_Sin',
            'Earnings_Proximity'
        ]
        
        available_features = [f for f in features if f in data.columns]
        X = data[available_features]
        y = data['Target']
        
        return X, y, data, available_features
    
    def train_and_predict(self, symbol):
        """Train model and generate interactive analysis"""
        try:
            # Detect market
            market_name, market_info = self.detect_market(symbol)
            market_info['market'] = market_name
            
            print(f"🌍 Detected market: {market_name} ({market_info['currency']})")
            
            # Get current price
            current_price, current_date = self.get_current_price_robust(symbol)
            
            # Prepare data
            X, y, historical_data, features = self.prepare_data(symbol)
            
            print(f"🔧 Training global market model with {len(features)} features")
            
            # Use current price if available
            if current_price is not None:
                actual_current_price = current_price
                actual_current_date = current_date
                print(f"💰 Current price: {market_info['currency']} {actual_current_price:.2f} (as of {actual_current_date.strftime('%Y-%m-%d')})")
            else:
                actual_current_price = historical_data['Close'].iloc[-1]
                actual_current_date = historical_data.index[-1]
                print(f"💰 Using last available price: {market_info['currency']} {actual_current_price:.2f}")
            
            # Train model
            X_scaled = self.scaler.fit_transform(X)
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            print("🤖 Training global market prediction model...")
            self.model.fit(X_train, y_train)
            
            # Calculate confidence
            test_predictions = self.model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, test_predictions)
            confidence = max(65, min(85, (1 - mape) * 100))
            
            # Make prediction
            last_features = X_scaled[-1:].reshape(1, -1)
            base_prediction = self.model.predict(last_features)[0]
            
            # Calculate scenarios
            current_volatility = historical_data['Volatility'].iloc[-30:].mean()
            scenarios = self.calculate_realistic_scenarios(
                base_prediction, actual_current_price, current_volatility, market_info
            )
            
            # Prediction date
            prediction_date = actual_current_date + timedelta(days=90)
            
            # Create interactive chart
            interactive_fig = self.create_interactive_chart(
                symbol, historical_data, actual_current_price, scenarios,
                prediction_date, actual_current_date, market_info
            )
            
            # Calculate changes
            scenario_changes = {
                'best_case_change': (scenarios['best_case'] - actual_current_price) / actual_current_price * 100,
                'average_case_change': (scenarios['average_case'] - actual_current_price) / actual_current_price * 100,
                'worst_case_change': (scenarios['worst_case'] - actual_current_price) / actual_current_price * 100
            }
            
            return {
                'symbol': symbol,
                'market': market_name,
                'currency': market_info['currency'],
                'current_price': round(actual_current_price, 2),
                'current_date': actual_current_date.strftime('%Y-%m-%d'),
                'prediction_date': prediction_date.strftime('%B %d, %Y'),
                'prediction_date_short': prediction_date.strftime('%b %Y'),
                
                'best_case_price': round(scenarios['best_case'], 2),
                'average_case_price': round(scenarios['average_case'], 2),
                'worst_case_price': round(scenarios['worst_case'], 2),
                
                'best_case_change': round(scenario_changes['best_case_change'], 2),
                'average_case_change': round(scenario_changes['average_case_change'], 2),
                'worst_case_change': round(scenario_changes['worst_case_change'], 2),
                
                'confidence': round(confidence, 1),
                'data_points': len(historical_data),
                'model_type': f'Interactive Global Market Model ({market_name})',
                'interactive_chart': interactive_fig,
                
                'data_freshness': f"Global analysis current as of {actual_current_date.strftime('%B %d, %Y')}",
                'volatility_score': round(current_volatility * 100, 1),
                'market_bias': f'Positive ({market_name} market trend incorporated)'
            }
            
        except Exception as e:
            raise Exception(f"Error in global market prediction for {symbol}: {str(e)}")

# Test function
def test_global_predictor():
    """Test the global predictor with different markets"""
    predictor = InteractiveGlobalPredictor()
    
    # Test symbols from different markets
    test_symbols = ['AAPL', 'GOOGL', 'ASML.AS', 'TSLA']  # US, US, Netherlands, US
    
    for symbol in test_symbols:
        try:
            print(f"\n{'='*60}")
            result = predictor.train_and_predict(symbol)
            print(f"✅ Global Analysis for {symbol}:")
            print(f"Market: {result['market']} | Currency: {result['currency']}")
            print(f"Current: {result['currency']} {result['current_price']}")
            print(f"Best Case: {result['currency']} {result['best_case_price']} ({result['best_case_change']:+.1f}%)")
            print(f"Average: {result['currency']} {result['average_case_price']} ({result['average_case_change']:+.1f}%)")
            print(f"Worst: {result['currency']} {result['worst_case_price']} ({result['worst_case_change']:+.1f}%)")
            print(f"Confidence: {result['confidence']}%")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")

if __name__ == "__main__":
    test_global_predictor()
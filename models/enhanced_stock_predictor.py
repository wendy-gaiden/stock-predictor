import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedStockPredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.prediction_days = 65
        
        # Popular stocks for autocomplete
        self.popular_stocks = {
            # US Tech Giants
            'AAPL': 'Apple Inc.',
            'GOOGL': 'Alphabet Inc. (Google)',
            'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc.',
            'META': 'Meta Platforms Inc. (Facebook)',
            'TSLA': 'Tesla Inc.',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc.',
            
            # US Traditional
            'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson',
            'V': 'Visa Inc.',
            'WMT': 'Walmart Inc.',
            'PG': 'Procter & Gamble',
            'MA': 'Mastercard Inc.',
            'UNH': 'UnitedHealth Group',
            'HD': 'Home Depot Inc.',
            'DIS': 'Walt Disney Company',
            'BAC': 'Bank of America',
            
            # ETFs
            'SPY': 'SPDR S&P 500 ETF',
            'QQQ': 'Invesco QQQ Trust',
            'VTI': 'Vanguard Total Stock Market',
            
            # International (examples)
            'ASML.AS': 'ASML Holding (Netherlands)',
            'SAP.DE': 'SAP SE (Germany)',
            'NESN.SW': 'Nestlé SA (Switzerland)',
            '7203.T': 'Toyota Motor (Japan)',
            '0700.HK': 'Tencent Holdings (Hong Kong)',
            'SHOP.TO': 'Shopify Inc. (Canada)'
        }
        
        # Global market info
        self.global_markets = {
            'US': {'suffix': '', 'currency': 'USD'},
            'Canada': {'suffix': '.TO', 'currency': 'CAD'},
            'UK': {'suffix': '.L', 'currency': 'GBP'},
            'Germany': {'suffix': '.DE', 'currency': 'EUR'},
            'Netherlands': {'suffix': '.AS', 'currency': 'EUR'},
            'Switzerland': {'suffix': '.SW', 'currency': 'CHF'},
            'Japan': {'suffix': '.T', 'currency': 'JPY'},
            'Hong Kong': {'suffix': '.HK', 'currency': 'HKD'}
        }
    
    def get_stock_suggestions(self, query):
        """Get stock suggestions based on partial input"""
        query = query.upper()
        suggestions = []
        
        for symbol, name in self.popular_stocks.items():
            if query in symbol or query in name.upper():
                suggestions.append({
                    'symbol': symbol,
                    'name': name,
                    'display': f"{symbol} - {name}"
                })
        
        return suggestions[:10]  # Return top 10 matches
    
    def detect_market(self, symbol):
        """Detect market and currency"""
        symbol_upper = symbol.upper()
        
        for market, info in self.global_markets.items():
            if symbol_upper.endswith(info['suffix']) and info['suffix']:
                return market, info['currency']
        
        return 'US', 'USD'
    
    def normalize_datetime_index(self, df):
        """Ensure DataFrame index is timezone-naive"""
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        # Also ensure the index is datetime type
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df
    
    def get_current_price_robust(self, symbol):
        """Get current price with enhanced methods"""
        try:
            stock = yf.Ticker(symbol)
            
            # First try to get the most recent price from info
            info = stock.info
            current_date = datetime.now()
            
            # Check for real-time price first
            for price_key in ['currentPrice', 'regularMarketPrice', 'ask', 'bid', 'previousClose']:
                if price_key in info and info[price_key] and info[price_key] > 0:
                    price = info[price_key]
                    print(f"Got {price_key}: {price}")
                    return price, current_date
            
            # If no real-time price, get historical data
            for period in ["1d", "5d", "1mo", "3mo"]:
                try:
                    data = stock.history(period=period, interval="1d")
                    if not data.empty:
                        data = self.normalize_datetime_index(data)
                        valid_prices = data['Close'].dropna()
                        if not valid_prices.empty:
                            # Return the most recent available price
                            return valid_prices.iloc[-1], valid_prices.index[-1]
                except:
                    continue
                    
            raise ValueError("No price data available")
            
        except Exception as e:
            print(f"Warning: Could not get current price for {symbol}: {e}")
            return None, None
    
    def calculate_indicators(self, data):
        """Calculate technical indicators with better NaN handling"""
        # CRITICAL: Work on a copy to preserve original data
        df = data.copy(deep=True)
        
        # Ensure we have clean data
        df = df.dropna(subset=['Close'])
        
        # Store original OHLC values to ensure they're not modified
        original_ohlc = df[['Open', 'High', 'Low', 'Close']].copy()
        
        # Moving averages with minimum periods to reduce NaN
        df['MA_20'] = df['Close'].rolling(window=20, min_periods=10).mean()
        df['MA_50'] = df['Close'].rolling(window=50, min_periods=25).mean()
        df['MA_200'] = df['Close'].rolling(window=200, min_periods=100).mean()
        
        # Price changes
        df['Price_Change_1d'] = df['Close'].pct_change(1).fillna(0)
        df['Price_Change_5d'] = df['Close'].pct_change(5).fillna(0)
        df['Price_Change_20d'] = df['Close'].pct_change(20).fillna(0)
        
        # Technical indicators
        df['RSI'] = self.calculate_rsi(df['Close']).fillna(50)  # Default RSI to neutral
        
        # MACD with fillna
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = (ema_12 - ema_26).fillna(0)
        
        # Volatility with minimum periods
        df['Volatility'] = df['Close'].pct_change().rolling(window=20, min_periods=5).std().fillna(0)
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=5).mean()
        df['Volume_Ratio'] = (df['Volume'] / (df['Volume_MA'] + 1e-10)).fillna(1)
        
        # Time features (these should never be NaN)
        df['Days_Index'] = np.arange(len(df))
        df['Days_Normalized'] = df['Days_Index'] / len(df)
        
        # Seasonal (these should never be NaN)
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['DayOfWeek'] = df.index.dayofweek
        
        # Price position indicators with proper NaN handling
        df['Price_to_MA20'] = np.where(
            df['MA_20'].notna() & (df['MA_20'] > 0),
            (df['Close'] / df['MA_20'] - 1),
            0
        )
        
        df['Price_to_MA50'] = np.where(
            df['MA_50'].notna() & (df['MA_50'] > 0),
            (df['Close'] / df['MA_50'] - 1),
            0
        )
        
        df['Price_to_MA200'] = np.where(
            df['MA_200'].notna() & (df['MA_200'] > 0),
            (df['Close'] / df['MA_200'] - 1),
            0
        )
        
        # High/Low indicators
        df['High_Low_Ratio'] = np.where(
            df['Close'] > 0,
            (df['High'] - df['Low']) / df['Close'],
            0
        )
        
        df['Close_to_High'] = np.where(
            (df['High'] - df['Low']) > 0,
            (df['Close'] - df['Low']) / (df['High'] - df['Low']),
            0.5
        )
        
        # Final cleanup: replace any remaining inf or -inf with 0
        df = df.replace([np.inf, -np.inf], 0)
        
        # CRITICAL: Restore original OHLC values to ensure they haven't been modified
        df[['Open', 'High', 'Low', 'Close']] = original_ohlc
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def prepare_data(self, symbol, period="5y"):
        """Prepare data for analysis - FIXED to handle NaN values"""
        print(f"📊 Downloading {period} of data for {symbol}...")
        
        stock = yf.Ticker(symbol)
        
        # Get data with standard method
        data = stock.history(period=period, interval="1d", auto_adjust=True)
        
        if data.empty:
            # Try alternative method
            print("🔄 Trying alternative download method...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1825)  # 5 years
            data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
            
            # Handle multi-level columns from yf.download
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Normalize timezone
        data = self.normalize_datetime_index(data)
        
        print(f"✅ Downloaded {len(data)} days of data")
        print(f"📅 Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"💰 Price check - First: ${data['Close'].iloc[0]:.2f}, Mid: ${data['Close'].iloc[len(data)//2]:.2f}, Last: ${data['Close'].iloc[-1]:.2f}")
        print(f"📊 Price range: ${data['Close'].min():.2f} to ${data['Close'].max():.2f}")
        
        # IMPORTANT: Keep a copy of ALL data for charting before we process it
        full_data_for_chart = data.copy()
        
        # Calculate indicators on the full dataset
        data_with_indicators = self.calculate_indicators(data)
        full_data_for_chart = self.calculate_indicators(full_data_for_chart)
        
        # Create target variable (this will create NaN for last 65 days)
        data_with_indicators['Target'] = data_with_indicators['Close'].shift(-self.prediction_days)
        
        # For training, we need to drop NaN targets
        training_data = data_with_indicators.dropna(subset=['Target'])
        
        print(f"📊 Training data: {len(training_data)} days (excludes last {self.prediction_days} days)")
        
        # Select features for training
        features = [
            'Open', 'High', 'Low', 'Volume',
            'MA_20', 'MA_50', 'MA_200',
            'Price_Change_1d', 'Price_Change_5d', 'Price_Change_20d',
            'RSI', 'MACD', 'Volatility', 'Volume_Ratio',
            'Days_Normalized', 'Month', 'Quarter', 'DayOfWeek',
            'Price_to_MA20', 'Price_to_MA50',
            'High_Low_Ratio', 'Close_to_High'
        ]
        
        # Only use features that exist
        available_features = [f for f in features if f in training_data.columns]
        
        # Create feature matrix from training data only
        X = training_data[available_features].copy()
        y = training_data['Target'].copy()
        
        # CRITICAL: Handle NaN values in features
        # Option 1: Drop rows with any NaN values
        before_drop = len(X)
        X = X.dropna()
        y = y[X.index]  # Keep y aligned with X
        after_drop = len(X)
        
        if before_drop != after_drop:
            print(f"⚠️ Dropped {before_drop - after_drop} rows with NaN values")
        
        # Option 2: Fill remaining NaN values with forward fill then backward fill
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # Option 3: For any remaining NaN values, fill with 0
        X = X.fillna(0)
        
        # Verify no NaN values remain
        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            print(f"⚠️ Warning: NaN values found in columns: {nan_cols}")
            # Force fill with 0
            X = X.fillna(0)
        
        # Verify we still have enough training data
        min_required_data = 100
        if len(X) < min_required_data:
            raise ValueError(f"Not enough data for analysis after cleaning. Got {len(X)} days, need at least {min_required_data}")
        
        print(f"🔧 Using {len(available_features)} features for training")
        print(f"✅ Clean training data: {len(X)} samples")
        
        # Return full data for charting (includes recent data)
        return X, y, full_data_for_chart, available_features
    
    def create_clean_chart(self, symbol, data, current_price, scenarios, prediction_date, current_date, currency):
        """Create clean, professional chart without data corruption"""
        
        # CRITICAL: Get the raw historical data again to ensure we have uncorrupted prices
        print(f"📊 Preparing chart for {symbol}...")
        
        # Get last 6 months of data
        six_months_ago = current_date - timedelta(days=180)
        
        # Ensure timezone compatibility
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            # If data has timezone, make six_months_ago timezone aware too
            six_months_ago = pd.Timestamp(six_months_ago).tz_localize(data.index.tz)
        
        # IMPORTANT: Use the original Close prices, not any processed version
        # Extract only the date range we need
        mask = data.index >= six_months_ago
        chart_dates = data.index[mask]
        chart_prices = data.loc[mask, 'Close'].values
        
        # Verify the data
        print(f"   Chart data: {len(chart_prices)} points from {chart_dates[0].strftime('%Y-%m-%d')} to {chart_dates[-1].strftime('%Y-%m-%d')}")
        print(f"   Price range: ${chart_prices.min():.2f} to ${chart_prices.max():.2f}")
        
        fig = go.Figure()
        
        # Historical price line - using raw close prices
        fig.add_trace(go.Scatter(
            x=chart_dates,
            y=chart_prices,
            mode='lines',
            name='Price History',
            line=dict(color='#1f77b4', width=2),
            connectgaps=True,
            hovertemplate=f'<b>Date:</b> %{{x|%b %d, %Y}}<br><b>Price:</b> {currency} %{{y:.2f}}<extra></extra>'
        ))
        
        # Moving average (if available and enough data)
        if 'MA_50' in data.columns and len(chart_dates) > 50:
            ma50_values = data.loc[mask, 'MA_50'].values
            # Only plot MA where it's not NaN
            ma_mask = ~np.isnan(ma50_values)
            if np.any(ma_mask):
                fig.add_trace(go.Scatter(
                    x=chart_dates[ma_mask],
                    y=ma50_values[ma_mask],
                    mode='lines',
                    name='50-Day Avg',
                    line=dict(color='#ff7f0e', width=1, dash='dot'),
                    opacity=0.6,
                    connectgaps=True,
                    hovertemplate=f'<b>50-Day MA:</b> {currency} %{{y:.2f}}<extra></extra>'
                ))
        
        # Check for gap between last data and current price
        last_data_date = chart_dates[-1]
        last_data_price = chart_prices[-1]
        days_gap = (current_date - last_data_date).days
        
        if days_gap > 1:
            # Connect last data point to current price with dotted line
            fig.add_trace(go.Scatter(
                x=[last_data_date, current_date],
                y=[last_data_price, current_price],
                mode='lines+markers',
                name='Gap Period',
                line=dict(color='gray', width=2, dash='dot'),
                marker=dict(size=6),
                hovertemplate=f'<b>Gap: {days_gap} days</b><br>%{{y:.2f}}<extra></extra>'
            ))
            
            # Add gap annotation
            fig.add_annotation(
                x=last_data_date + timedelta(days=days_gap/2),
                y=(last_data_price + current_price) / 2,
                text=f"Data Gap<br>({days_gap} days)",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="gray",
                ax=0,
                ay=-40,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1
            )
        
        # Current price marker
        fig.add_trace(go.Scatter(
            x=[current_date],
            y=[current_price],
            mode='markers+text',
            name='Current',
            marker=dict(color='#2ca02c', size=14, symbol='circle'),
            text=[f'{currency} {current_price:.2f}'],
            textposition='top center',
            textfont=dict(color='#2ca02c', size=12, family='Arial Black'),
            hovertemplate=f'<b>Current Price:</b> {currency} %{{y:.2f}}<br><b>Date:</b> %{{x|%b %d, %Y}}<extra></extra>'
        ))
        
        # Prediction scenarios - individual traces for cleaner legend
        scenarios_data = [
            ('Best Case', scenarios['best_case'], '#28a745', 'triangle-up'),
            ('Average', scenarios['average_case'], '#007bff', 'square'),
            ('Worst Case', scenarios['worst_case'], '#dc3545', 'triangle-down')
        ]
        
        for name, price, color, symbol in scenarios_data:
            change_pct = (price - current_price) / current_price * 100
            
            # Scenario marker
            fig.add_trace(go.Scatter(
                x=[prediction_date],
                y=[price],
                mode='markers+text',
                name=name,
                marker=dict(color=color, size=12, symbol=symbol),
                text=[f'{currency} {price:.2f}'],
                textposition='middle right' if name == 'Average' else 'top center',
                textfont=dict(size=10, color=color),
                hovertemplate=f'<b>{name}:</b> {currency} %{{y:.2f}}<br><b>Change:</b> {change_pct:+.1f}%<extra></extra>'
            ))
            
            # Projection line
            fig.add_trace(go.Scatter(
                x=[current_date, prediction_date],
                y=[current_price, price],
                mode='lines',
                line=dict(color=color, width=1, dash='dash'),
                opacity=0.4,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Calculate appropriate y-axis range based on actual data
        all_prices = list(chart_prices) + [current_price] + list(scenarios.values())
        y_min = min(all_prices) * 0.95
        y_max = max(all_prices) * 1.05
        
        print(f"   Y-axis range: ${y_min:.2f} to ${y_max:.2f}")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{symbol} Price Analysis & 3-Month Targets<br><sub>Current: {currency} {current_price:.2f} | Target Date: {prediction_date.strftime("%b %Y")}</sub>',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title='Date',
            yaxis_title=f'Price ({currency})',
            template='plotly_white',
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                font=dict(size=11),
                itemsizing='constant'
            ),
            margin=dict(l=60, r=30, t=80, b=60),
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                tickformat='%b %Y',
                rangeslider=dict(visible=False),
                type='date'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                tickformat='.2f',
                range=[y_min, y_max],
                autorange=False  # Force use our calculated range
            )
        )
        
        return fig
    
    def calculate_scenarios(self, base_prediction, current_price, volatility, currency):
        """Calculate realistic scenarios"""
        quarterly_trend = 0.02  # 2% quarterly growth
        
        # Currency-specific adjustments
        if currency == 'JPY':
            quarterly_trend *= 0.8
        elif currency in ['EUR', 'GBP', 'CHF']:
            quarterly_trend *= 0.95
        
        vol_adj = volatility * 1.2
        
        return {
            'best_case': current_price * (1 + quarterly_trend + vol_adj),
            'average_case': current_price * (1 + quarterly_trend),
            'worst_case': max(current_price * (1 + quarterly_trend - vol_adj * 0.8), 
                            current_price * 0.85),
            'base_model': base_prediction
        }
    
    def train_and_predict(self, symbol):
        """Main prediction function - FIXED to show all recent data"""
        try:
            # Detect market and currency
            market, currency = self.detect_market(symbol)
            
            # Get current price
            current_price, current_date = self.get_current_price_robust(symbol)
            
            # Prepare data - now returns full data for charting
            X, y, full_historical_data, features = self.prepare_data(symbol)
            
            print(f"🔧 Training model with {len(features)} features")
            print(f"📊 Full data for chart: {len(full_historical_data)} days up to {full_historical_data.index[-1].strftime('%Y-%m-%d')}")
            
            if current_price is not None:
                actual_current_price = current_price
                actual_current_date = current_date
                print(f"💰 Current price: {currency} {actual_current_price:.2f}")
            else:
                # Use the last available price from full data
                actual_current_price = full_historical_data['Close'].iloc[-1]
                actual_current_date = full_historical_data.index[-1]
                print(f"💰 Using last available: {currency} {actual_current_price:.2f} from {actual_current_date.strftime('%Y-%m-%d')}")
            
            # Train model
            X_scaled = self.scaler.fit_transform(X)
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            self.model.fit(X_train, y_train)
            
            # Calculate confidence
            test_predictions = self.model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, test_predictions)
            confidence = max(65, min(85, (1 - mape) * 100))
            
            # Make prediction using the most recent data we have with all features
            # Get the last row from X that has all features
            last_features = X_scaled[-1:].reshape(1, -1)
            base_prediction = self.model.predict(last_features)[0]
            
            # Calculate scenarios
            current_volatility = full_historical_data['Volatility'].iloc[-30:].mean()
            scenarios = self.calculate_scenarios(
                base_prediction, actual_current_price, current_volatility, currency
            )
            
            # Prediction date
            prediction_date = actual_current_date + timedelta(days=90)
            
            # Create clean chart using FULL historical data
            clean_chart = self.create_clean_chart(
                symbol, full_historical_data, actual_current_price, scenarios,
                prediction_date, actual_current_date, currency
            )
            
            # Calculate changes
            scenario_changes = {
                'best_case_change': (scenarios['best_case'] - actual_current_price) / actual_current_price * 100,
                'average_case_change': (scenarios['average_case'] - actual_current_price) / actual_current_price * 100,
                'worst_case_change': (scenarios['worst_case'] - actual_current_price) / actual_current_price * 100
            }
            
            # Get company name
            company_name = self.popular_stocks.get(symbol, symbol)
            
            return {
                'symbol': symbol,
                'company_name': company_name,
                'market': market,
                'currency': currency,
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
                'data_points': len(full_historical_data),
                'training_points': len(X),
                'model_type': f'Enhanced Global Model ({market})',
                'interactive_chart': clean_chart,
                
                'data_freshness': f"Analysis current as of {actual_current_date.strftime('%B %d, %Y')}",
                'volatility_score': round(current_volatility * 100, 1)
            }
            
        except Exception as e:
            raise Exception(f"Error analyzing {symbol}: {str(e)}")
    
    def diagnose_chart_data(self, symbol="NVDA"):
        """Diagnose chart data issues"""
        print(f"\n🔍 Diagnosing chart data for {symbol}...")
        
        try:
            # Get fresh data
            stock = yf.Ticker(symbol)
            
            # Test 1: Get raw data
            print("\nTest 1: Raw yfinance data (1 year)")
            raw_data = stock.history(period="1y")
            print(f"✓ Raw data points: {len(raw_data)}")
            print(f"✓ Date range: {raw_data.index[0]} to {raw_data.index[-1]}")
            print(f"✓ Price range: ${raw_data['Close'].min():.2f} - ${raw_data['Close'].max():.2f}")
            print("\nSample of raw prices (every 20 days):")
            for i in range(0, len(raw_data), 20):
                print(f"  {raw_data.index[i].strftime('%Y-%m-%d')}: ${raw_data['Close'].iloc[i]:.2f}")
            
            # Test 2: Process through our pipeline
            print("\n\nTest 2: After calculate_indicators")
            raw_data = self.normalize_datetime_index(raw_data)  # Normalize timezone first
            processed_data = self.calculate_indicators(raw_data.copy())
            print(f"✓ Processed data points: {len(processed_data)}")
            print(f"✓ Close price range: ${processed_data['Close'].min():.2f} - ${processed_data['Close'].max():.2f}")
            
            # Check if Close prices were modified
            if not raw_data['Close'].equals(processed_data['Close'][:len(raw_data)]):
                print("❌ ERROR: Close prices were modified during processing!")
                # Find differences
                for i in range(min(len(raw_data), len(processed_data))):
                    if abs(raw_data['Close'].iloc[i] - processed_data['Close'].iloc[i]) > 0.01:
                        print(f"  Difference at {raw_data.index[i]}: Raw=${raw_data['Close'].iloc[i]:.2f}, Processed=${processed_data['Close'].iloc[i]:.2f}")
                        if i > 5:
                            break
            else:
                print("✅ Close prices preserved correctly")
            
            # Test 3: Check 6-month extraction
            print("\n\nTest 3: 6-month data extraction for chart")
            current_date = datetime.now()
            six_months_ago = current_date - timedelta(days=180)
            
            # Method used in create_clean_chart
            mask = processed_data.index >= six_months_ago
            chart_dates = processed_data.index[mask]
            chart_prices = processed_data.loc[mask, 'Close'].values
            
            print(f"✓ Chart data points: {len(chart_prices)}")
            print(f"✓ Chart date range: {chart_dates[0]} to {chart_dates[-1]}")
            print(f"✓ Chart price range: ${chart_prices.min():.2f} - ${chart_prices.max():.2f}")
            print("\nFirst 5 chart prices:")
            for i in range(min(5, len(chart_prices))):
                print(f"  {chart_dates[i].strftime('%Y-%m-%d')}: ${chart_prices[i]:.2f}")
            print("\nLast 5 chart prices:")
            for i in range(max(0, len(chart_prices)-5), len(chart_prices)):
                print(f"  {chart_dates[i].strftime('%Y-%m-%d')}: ${chart_prices[i]:.2f}")
            
            # Test 4: Check current price
            print("\n\nTest 4: Current price check")
            current_price, current_date_from_api = self.get_current_price_robust(symbol)
            print(f"✓ Current price from API: ${current_price:.2f}")
            print(f"✓ Last price in data: ${processed_data['Close'].iloc[-1]:.2f}")
            print(f"✓ Price difference: ${abs(current_price - processed_data['Close'].iloc[-1]):.2f}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

# Test function
def test_enhanced_predictor():
    """Test the enhanced predictor"""
    predictor = EnhancedStockPredictor()
    
    # Test autocomplete
    print("🔍 Testing autocomplete for 'app':")
    suggestions = predictor.get_stock_suggestions('app')
    for suggestion in suggestions:
        print(f"  {suggestion['display']}")
    
    # Test prediction
    try:
        result = predictor.train_and_predict("AAPL")
        print(f"\n✅ Enhanced Analysis for AAPL:")
        print(f"Company: {result['company_name']}")
        print(f"Current: {result['currency']} {result['current_price']}")
        print(f"Best: {result['currency']} {result['best_case_price']} ({result['best_case_change']:+.1f}%)")
        print(f"Average: {result['currency']} {result['average_case_price']} ({result['average_case_change']:+.1f}%)")
        print(f"Worst: {result['currency']} {result['worst_case_price']} ({result['worst_case_change']:+.1f}%)")
        print(f"Confidence: {result['confidence']}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Run diagnostic for NVDA
    predictor = EnhancedStockPredictor()
    predictor.diagnose_chart_data("NVDA")
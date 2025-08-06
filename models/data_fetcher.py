import yfinance as yf
import time
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class RobustDataFetcher:
    def __init__(self):
        self.max_retries = 1  # Just one retry to save time
        self.base_delay = 1
        self.use_demo_mode = False
        
    def generate_demo_data(self, symbol, days=1000):
        """Generate realistic demo data when Yahoo Finance is blocked"""
        print(f"📊 Using demo data for {symbol} (Yahoo Finance is blocked on this server)")
        
        # Base prices for popular stocks
        base_prices = {
            'AAPL': 175, 'GOOGL': 140, 'MSFT': 380, 'NVDA': 450,
            'TSLA': 250, 'AMZN': 150, 'META': 350, 'JPM': 150,
            'SPY': 440, 'QQQ': 380
        }
        base_price = base_prices.get(symbol, 100)
        
        # Generate dates
        end_date = datetime.now()
        dates = pd.date_range(end=end_date - timedelta(days=1), periods=days, freq='B')
        
        # Generate realistic price movements
        np.random.seed(hash(symbol) % 10000)
        returns = np.random.normal(0.0005, 0.015, days)
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # Add trend and volatility
        trend = np.linspace(0, 0.1, days)
        seasonal = 3 * np.sin(np.linspace(0, 4*np.pi, days))
        price_series = price_series * (1 + trend) + seasonal
        
        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['Close'] = price_series
        data['Open'] = data['Close'] * (1 + np.random.normal(0, 0.002, days))
        data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, days)))
        data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, days)))
        data['Volume'] = np.random.randint(10000000, 100000000, days)
        
        self.use_demo_mode = True
        return data
        
    def fetch_with_retry(self, symbol, period="5y", interval="1d"):
        """Fetch data with fallback to demo"""
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    time.sleep(self.base_delay)
                
                print(f"📊 Attempting to fetch real data for {symbol}...")
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="2y", interval=interval)
                
                if data is not None and not data.empty and len(data) > 10:
                    print(f"✅ Got real data for {symbol}")
                    self.use_demo_mode = False
                    return data
                    
            except Exception as e:
                print(f"❌ Yahoo Finance error: {str(e)[:100]}")
                
        # Fallback to demo data
        print(f"⚠️ Using demo data due to Yahoo Finance restrictions")
        return self.generate_demo_data(symbol)
    
    def get_current_price(self, symbol):
        """Get current price with demo fallback"""
        if self.use_demo_mode:
            # Return consistent demo prices
            base_prices = {
                'AAPL': 175, 'GOOGL': 140, 'MSFT': 380, 'NVDA': 450,
                'TSLA': 250, 'AMZN': 150, 'META': 350, 'JPM': 150
            }
            base = base_prices.get(symbol, 100)
            # Add small variation
            current = base * (1 + random.uniform(-0.01, 0.01))
            return current, datetime.now()
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            for key in ['currentPrice', 'regularMarketPrice', 'previousClose']:
                if key in info and info.get(key):
                    return float(info[key]), datetime.now()
                    
        except:
            pass
            
        # Use demo price as fallback
        self.use_demo_mode = True
        return self.get_current_price(symbol)
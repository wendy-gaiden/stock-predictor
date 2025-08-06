import requests
import pandas as pd
from datetime import datetime
import time
import os

class AlphaVantageDataFetcher:
    def __init__(self):
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', 'demo')
        self.base_url = 'https://www.alphavantage.co/query'
        print(f"📊 Using Alpha Vantage API (key: {'✓' if self.api_key != 'demo' else '✗'})")
        
    def fetch_daily_data(self, symbol):
        """Fetch daily data from Alpha Vantage"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full',
            'datatype': 'json'
        }
        
        try:
            print(f"🔄 Fetching data from Alpha Vantage for {symbol}...")
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            if 'Error Message' in data:
                raise ValueError(f"Invalid symbol: {symbol}")
            
            if 'Note' in data:
                raise ValueError("API call limit reached (5/min). Please wait 60 seconds.")
                
            if 'Time Series (Daily)' not in data:
                raise ValueError(f"No data available. Response: {data}")
                
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns to match yfinance format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            print(f"✅ Successfully fetched {len(df)} days of data from Alpha Vantage")
            return df
            
        except Exception as e:
            print(f"❌ Alpha Vantage error: {e}")
            return None
    
    def get_current_price(self, symbol):
        """Get current price from Alpha Vantage"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'Global Quote' in data and '05. price' in data['Global Quote']:
                price = float(data['Global Quote']['05. price'])
                date_str = data['Global Quote']['07. latest trading day']
                date = pd.to_datetime(date_str)
                print(f"✅ Current price from Alpha Vantage: ${price}")
                return price, date
                
        except Exception as e:
            print(f"❌ Error getting quote: {e}")
            
        return None, None

# Create a wrapper that falls back to Alpha Vantage
class RobustDataFetcher:
    def __init__(self):
        self.alpha_vantage = AlphaVantageDataFetcher()
        self.max_retries = 3
        self.base_delay = 2
        
    def fetch_with_retry(self, symbol, period="5y", interval="1d"):
        """Try yfinance first, then Alpha Vantage"""
        import yfinance as yf
        
        # First try yfinance (faster if it works)
        print(f"📊 Trying yfinance for {symbol}...")
        try:
            data = yf.download(symbol, period="2y", progress=False, show_errors=False)
            if not data.empty:
                print("✅ Got data from yfinance")
                return data
        except Exception as e:
            print(f"❌ yfinance failed: {e}")
        
        # Fall back to Alpha Vantage
        print("🔄 Falling back to Alpha Vantage...")
        time.sleep(1)  # Respect rate limits
        return self.alpha_vantage.fetch_daily_data(symbol)
    
    def get_current_price(self, symbol):
        """Get current price from Alpha Vantage"""
        return self.alpha_vantage.get_current_price(symbol)

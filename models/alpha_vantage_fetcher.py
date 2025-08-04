# models/alpha_vantage_fetcher.py
import requests
import pandas as pd
from datetime import datetime
import time
import os

class AlphaVantageDataFetcher:
    def __init__(self):
        # You'll set this as an environment variable
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', 'demo')
        self.base_url = 'https://www.alphavantage.co/query'
        
    def fetch_daily_data(self, symbol):
        """Fetch daily data from Alpha Vantage"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full',  # Get full historical data
            'datatype': 'json'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'Error Message' in data:
                raise ValueError(f"Invalid symbol: {symbol}")
            
            if 'Note' in data:
                # API call frequency limit
                raise ValueError("API call limit reached. Please try again later.")
                
            if 'Time Series (Daily)' not in data:
                raise ValueError("No data available")
                
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            return df
            
        except Exception as e:
            print(f"Error fetching from Alpha Vantage: {e}")
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
                return price, datetime.now()
                
        except Exception as e:
            print(f"Error getting quote: {e}")
            
        return None, None
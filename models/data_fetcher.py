import yfinance as yf
import time
import random
from datetime import datetime, timedelta
import pandas as pd

class RobustDataFetcher:
    def __init__(self):
        self.max_retries = 3
        self.base_delay = 5  # Increased delay
        
    def fetch_with_retry(self, symbol, period="5y", interval="1d"):
        """Fetch data with retry logic and delays"""
        for attempt in range(self.max_retries):
            try:
                # Add longer delay between attempts
                if attempt > 0:
                    delay = self.base_delay * (attempt + 1) + random.uniform(0, 3)
                    print(f"Waiting {delay:.1f}s before retry {attempt + 1}...")
                    time.sleep(delay)
                
                # Use yfinance with specific parameters
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval, prepost=False)
                
                if data is not None and not data.empty:
                    print(f"✅ Successfully fetched data for {symbol}")
                    return data
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                continue
        
        # If all retries failed, return None
        return None
    
    def get_current_price(self, symbol):
        """Get current price with error handling"""
        try:
            time.sleep(2)  # Add delay
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different price fields
            for key in ['currentPrice', 'regularMarketPrice', 'previousClose']:
                if key in info and info[key]:
                    return info[key], datetime.now()
            
            # Fallback to history
            hist = ticker.history(period="5d")
            if not hist.empty:
                return hist['Close'].iloc[-1], hist.index[-1]
                
        except Exception as e:
            print(f"Error getting current price: {e}")
            
        return None, None
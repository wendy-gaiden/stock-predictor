import yfinance as yf
import time
import random
from datetime import datetime, timedelta
import pandas as pd

class RobustDataFetcher:
    def __init__(self):
        self.max_retries = 3
        self.base_delay = 2
        
    def fetch_with_retry(self, symbol, period="5y", interval="1d"):
        """Fetch data with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Add random delay to avoid being detected as bot
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                if attempt > 0:
                    print(f"Retry {attempt + 1}/{self.max_retries} after {delay:.1f}s delay...")
                    time.sleep(delay)
                
                # Try different download methods
                if attempt == 0:
                    # Method 1: Direct download
                    data = yf.download(
                        symbol, 
                        period=period, 
                        interval=interval,
                        progress=False,
                        show_errors=False,
                        threads=False,
                        timeout=10
                    )
                elif attempt == 1:
                    # Method 2: Using Ticker object
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period, interval=interval)
                else:
                    # Method 3: With specific dates
                    end = datetime.now()
                    start = end - timedelta(days=1825)  # 5 years
                    data = yf.download(
                        symbol,
                        start=start,
                        end=end,
                        interval=interval,
                        progress=False,
                        show_errors=False,
                        threads=False
                    )
                
                # Check if we got data
                if data is not None and not data.empty:
                    print(f"✅ Successfully fetched data for {symbol}")
                    return data
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if "429" in str(e):
                    # Rate limited - wait longer
                    time.sleep(10)
                continue
        
        return None
    
    def get_current_price(self, symbol):
        """Get current price with fallback methods"""
        try:
            # Add delay to respect rate limits
            time.sleep(0.5)
            
            ticker = yf.Ticker(symbol)
            
            # Try fast_info first (lighter weight)
            try:
                fast_info = ticker.fast_info
                if hasattr(fast_info, 'last_price') and fast_info.last_price:
                    return fast_info.last_price, datetime.now()
            except:
                pass
            
            # Try regular info
            info = ticker.info
            for key in ['currentPrice', 'regularMarketPrice', 'previousClose']:
                if key in info and info[key]:
                    return info[key], datetime.now()
            
            # Fallback to recent history
            hist = ticker.history(period="5d")
            if not hist.empty:
                return hist['Close'].iloc[-1], hist.index[-1]
                
        except Exception as e:
            print(f"Error getting current price: {e}")
            
        return None, None

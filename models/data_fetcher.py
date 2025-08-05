import yfinance as yf
import time
import random
from datetime import datetime, timedelta
import pandas as pd

class RobustDataFetcher:
    def __init__(self):
        self.max_retries = 3
        self.base_delay = 5  # 5 second base delay
        
    def fetch_with_retry(self, symbol, period="5y", interval="1d"):
        """Fetch data with retry logic and delays"""
        for attempt in range(self.max_retries):
            try:
                # Add delay before each attempt (except first)
                if attempt > 0:
                    delay = self.base_delay * (attempt + 1) + random.uniform(0, 3)
                    print(f"⏳ Waiting {delay:.1f}s before retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(delay)
                else:
                    # Even on first attempt, add a small delay
                    time.sleep(random.uniform(1, 2))
                
                print(f"📊 Fetching data for {symbol} (attempt {attempt + 1}/{self.max_retries})...")
                
                # Use yfinance Ticker object
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    period=period, 
                    interval=interval, 
                    prepost=False,
                    repair=True
                )
                
                if data is not None and not data.empty:
                    print(f"✅ Successfully fetched {len(data)} days of data for {symbol}")
                    return data
                else:
                    print(f"⚠️ Empty data returned for {symbol}")
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if "429" in str(e) or "Too Many Requests" in str(e):
                    # If rate limited, wait even longer
                    extra_delay = 10 + random.uniform(0, 5)
                    print(f"🚫 Rate limited! Waiting additional {extra_delay:.1f}s...")
                    time.sleep(extra_delay)
                continue
        
        # If all retries failed
        print(f"❌ All attempts failed for {symbol}")
        return None
    
    def get_current_price(self, symbol):
        """Get current price with error handling and delays"""
        try:
            # Add random delay to avoid rate limits
            delay = random.uniform(2, 4)
            print(f"⏳ Waiting {delay:.1f}s before fetching current price...")
            time.sleep(delay)
            
            ticker = yf.Ticker(symbol)
            
            # Try to get price from info (less rate-limited)
            try:
                info = ticker.info
                price_keys = ['currentPrice', 'regularMarketPrice', 'ask', 'bid', 'previousClose']
                
                for key in price_keys:
                    if key in info and info.get(key):
                        price = float(info[key])
                        if price > 0:
                            print(f"✅ Got {key}: ${price:.2f}")
                            return price, datetime.now()
            except:
                print("⚠️ Could not get price from info, trying history...")
            
            # Fallback to recent history
            time.sleep(2)  # Another small delay
            hist = ticker.history(period="5d", interval="1d")
            if not hist.empty:
                last_price = float(hist['Close'].iloc[-1])
                last_date = hist.index[-1]
                print(f"✅ Got price from history: ${last_price:.2f}")
                return last_price, last_date
                
        except Exception as e:
            print(f"❌ Error getting current price: {e}")
            
        return None, None
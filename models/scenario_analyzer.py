import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for web
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class InvestmentScenarioAnalyzer:
    def __init__(self):
        self.default_investment = 200000
        
    def get_historical_returns(self, symbol, years_back=10):
        """Get historical annual returns for the stock"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365 + 100)
            
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No historical data found for {symbol}")
            
            # Calculate annual returns
            data['Year'] = data.index.year
            annual_data = data.groupby('Year')['Close'].agg(['first', 'last'])
            annual_returns = ((annual_data['last'] - annual_data['first']) / annual_data['first'] * 100)
            
            # Remove incomplete years
            current_year = datetime.now().year
            if datetime.now().month < 12:
                annual_returns = annual_returns[annual_returns.index < current_year]
            
            return annual_returns.dropna()
            
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            # Fallback data for demo
            return pd.Series([15, 22, -8, 12, 18, -3, 25, 8, 14, 19])
    
    def calculate_scenarios(self, symbol, investment_amount=None, projection_years=3):
        """Calculate best, average, and worst-case scenarios"""
        if investment_amount is None:
            investment_amount = self.default_investment
            
        # Get historical returns
        historical_returns = self.get_historical_returns(symbol)
        
        if len(historical_returns) < 3:
            raise ValueError(f"Not enough historical data for {symbol}")
        
        # Calculate statistics
        returns_pct = historical_returns / 100
        
        # Define scenarios based on historical percentiles
        best_case_return = np.percentile(returns_pct, 85)
        average_case_return = returns_pct.mean()
        worst_case_return = np.percentile(returns_pct, 15)
        
        # Calculate projections
        years = np.arange(0, projection_years + 1)
        
        best_case_values = investment_amount * (1 + best_case_return) ** years
        average_case_values = investment_amount * (1 + average_case_return) ** years
        worst_case_values = investment_amount * (1 + worst_case_return) ** years
        
        # Calculate total returns
        best_case_total_return = (best_case_values[-1] - investment_amount) / investment_amount * 100
        average_case_total_return = (average_case_values[-1] - investment_amount) / investment_amount * 100
        worst_case_total_return = (worst_case_values[-1] - investment_amount) / investment_amount * 100
        
        return {
            'symbol': symbol,
            'years': years.tolist(),
            'best_case': {
                'values': best_case_values.tolist(),
                'annual_return': best_case_return * 100,
                'total_return': best_case_total_return,
                'final_value': best_case_values[-1]
            },
            'average_case': {
                'values': average_case_values.tolist(),
                'annual_return': average_case_return * 100,
                'total_return': average_case_total_return,
                'final_value': average_case_values[-1]
            },
            'worst_case': {
                'values': worst_case_values.tolist(),
                'annual_return': worst_case_return * 100,
                'total_return': worst_case_total_return,
                'final_value': worst_case_values[-1]
            },
            'historical_stats': {
                'years_analyzed': len(historical_returns),
                'best_year': historical_returns.max(),
                'worst_year': historical_returns.min(),
                'volatility': returns_pct.std() * 100,
                'historical_returns': historical_returns.tolist()
            },
            'investment_amount': investment_amount,
            'projection_years': projection_years
        }
    
    def create_scenario_chart(self, symbol, scenarios):
        """Create scenario visualization chart"""
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        years = scenarios['years']
        
        # Plot scenarios
        ax.plot(years, scenarios['best_case']['values'], 
                label=f'Best Case ({scenarios["best_case"]["annual_return"]:.1f}% p.a.)', 
                color='#2E8B57', marker='o', linewidth=3, markersize=8)
        
        ax.plot(years, scenarios['average_case']['values'], 
                label=f'Average Case ({scenarios["average_case"]["annual_return"]:.1f}% p.a.)', 
                color='#4169E1', marker='s', linewidth=3, markersize=8)
        
        ax.plot(years, scenarios['worst_case']['values'], 
                label=f'Worst Case ({scenarios["worst_case"]["annual_return"]:.1f}% p.a.)', 
                color='#DC143C', marker='^', linewidth=3, markersize=8)
        
        # Add initial investment line
        ax.axhline(y=scenarios['investment_amount'], color='gray', 
                   linestyle='--', alpha=0.7, label='Initial Investment')
        
        # Formatting
        ax.set_title(f'{symbol} Investment Projection: ${scenarios["investment_amount"]:,.0f} Over {scenarios["projection_years"]} Years\n'
                     f'Based on {scenarios["historical_stats"]["years_analyzed"]} Years of Historical Data', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Years', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Value (USD)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(years)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        return fig

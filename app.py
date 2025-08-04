from flask import Flask, render_template, request, jsonify
import sys
import os
import json
import plotly

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.enhanced_stock_predictor import EnhancedStockPredictor as StockPredictor

app = Flask(__name__, 
            template_folder=os.path.join(project_root, 'templates'),
            static_folder=os.path.join(project_root, 'static'))

predictor = StockPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search_stocks', methods=['GET'])
def search_stocks():
    """API endpoint for stock symbol autocomplete"""
    query = request.args.get('q', '')
    if len(query) < 1:
        return jsonify([])
    
    suggestions = predictor.get_stock_suggestions(query)
    return jsonify(suggestions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Please enter a stock symbol'})
        
        print(f"\n🔍 DEBUG: Analyzing {symbol}")
        
        # Get prediction with clean chart
        result = predictor.train_and_predict(symbol)
        
        # Debug: Check what data is in the result
        if 'interactive_chart' in result:
            chart = result['interactive_chart']
            print(f"📊 DEBUG: Chart has {len(chart.data)} traces")
            
            # Check the first trace (should be price history)
            if len(chart.data) > 0:
                price_trace = chart.data[0]
                y_values = price_trace.y
                if hasattr(y_values, '__len__'):
                    print(f"📊 DEBUG: Price trace has {len(y_values)} points")
                    print(f"📊 DEBUG: Price range: ${min(y_values):.2f} - ${max(y_values):.2f}")
                    print(f"📊 DEBUG: First few prices: {list(y_values[:5])}")
        
        # Convert Plotly figure to JSON
        chart_json = None
        if 'interactive_chart' in result:
            chart_json = json.dumps(result['interactive_chart'], cls=plotly.utils.PlotlyJSONEncoder)
            del result['interactive_chart']
        
        if chart_json:
            result['chart_json'] = chart_json
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        print(f"❌ DEBUG: Error - {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Enhanced Stock Analysis App...")
    print("📱 Features: Stock Search, Clean Charts, Advanced Gap Filling")
    print("🌐 Open your browser and go to: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
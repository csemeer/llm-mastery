"""
REST API for Financial LLM

Provides HTTP endpoints for:
- Model predictions
- Trading signals
- Portfolio recommendations
- Model status and health checks

Usage:
    python scripts/deploy_api.py --checkpoint path/to/model.pt --port 8000

API Endpoints:
    GET  /health              - Health check
    POST /predict/trading     - Get trading signal
    POST /predict/risk        - Get risk assessment
    POST /predict/portfolio   - Get portfolio recommendation
    GET  /model/info          - Model information
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_llm.models.financial_llm import FinancialLLM
from financial_llm.data_processors.market_data import MarketDataFetcher, TechnicalIndicators


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model instance
model = None
device = None
model_info = {}


def load_model(checkpoint_path: str):
    """Load model from checkpoint"""
    global model, device, model_info

    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = device_type

    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model (in production, save config in checkpoint)
    model = FinancialLLM(
        vocab_size=50000,
        d_model=256,
        num_heads=8,
        num_text_layers=4,
        num_ts_layers=3,
        num_fusion_layers=2,
        ts_input_dim=7,
        num_indicators=20,
        max_seq_len_ts=60,
        dropout=0.0  # No dropout for inference
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Store model info
    model_info = {
        'loaded_at': datetime.now().isoformat(),
        'checkpoint': checkpoint_path,
        'device': str(device),
        'parameters': sum(p.numel() for p in model.parameters()),
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_loss': checkpoint.get('val_loss', 'unknown'),
        'val_acc': checkpoint.get('val_acc', 'unknown')
    }

    print(f"✓ Model loaded successfully")
    print(f"  Device: {device}")
    print(f"  Parameters: {model_info['parameters']:,}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    return jsonify(model_info)


@app.route('/predict/trading', methods=['POST'])
def predict_trading():
    """
    Predict trading signal

    Request body:
    {
        "ticker": "AAPL",
        "lookback": 60  # optional, default 60
    }

    OR provide raw OHLCV data:
    {
        "ohlcv": [[open, high, low, close, volume], ...],
        "indicators": [[ind1, ind2, ...], ...]  # optional
    }

    Response:
    {
        "action": "BUY|SELL|HOLD",
        "action_probs": [buy_prob, sell_prob, hold_prob],
        "price_prediction": predicted_return,
        "confidence": confidence_score,
        "timestamp": "2024-01-01T00:00:00"
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json

        # Option 1: Fetch data from ticker
        if 'ticker' in data:
            ticker = data['ticker']
            lookback = data.get('lookback', 60)

            # Fetch recent data
            fetcher = MarketDataFetcher()
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback + 30)  # Extra buffer

            df = fetcher.fetch_stock_data(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            df = TechnicalIndicators.add_indicators(df)

            # Get most recent lookback periods
            ohlcv_data = df[['open', 'high', 'low', 'close', 'volume']].values[-lookback:]

            indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            if indicator_cols:
                indicators_data = df[indicator_cols].values[-lookback:]
            else:
                indicators_data = None

        # Option 2: Use provided OHLCV data
        elif 'ohlcv' in data:
            ohlcv_data = np.array(data['ohlcv'])
            indicators_data = np.array(data['indicators']) if 'indicators' in data else None

        else:
            return jsonify({'error': 'Must provide either "ticker" or "ohlcv" data'}), 400

        # Normalize
        ohlcv_mean = ohlcv_data.mean(axis=0)
        ohlcv_std = ohlcv_data.std(axis=0) + 1e-8
        ohlcv_norm = (ohlcv_data - ohlcv_mean) / ohlcv_std

        if indicators_data is not None:
            indicators_mean = indicators_data.mean(axis=0)
            indicators_std = indicators_data.std(axis=0) + 1e-8
            indicators_norm = (indicators_data - indicators_mean) / indicators_std
        else:
            indicators_norm = None

        # Convert to tensors
        ohlcv_tensor = torch.tensor(ohlcv_norm, dtype=torch.float32).unsqueeze(0).to(device)
        indicators_tensor = torch.tensor(indicators_norm, dtype=torch.float32).unsqueeze(0).to(device) if indicators_norm is not None else None

        # Get prediction
        with torch.no_grad():
            predictions = model.predict_trading_action(
                ohlcv=ohlcv_tensor,
                indicators=indicators_tensor
            )

        action_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
        action = action_map[predictions['action'].item()]
        action_probs = predictions['action_probs'][0].cpu().numpy().tolist()
        price_pred = predictions['price_prediction'][0].item() if 'price_prediction' in predictions else None

        confidence = max(action_probs)

        response = {
            'action': action,
            'action_probs': {
                'buy': action_probs[0],
                'sell': action_probs[1],
                'hold': action_probs[2]
            },
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }

        if price_pred is not None:
            response['price_prediction'] = price_pred

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/risk', methods=['POST'])
def predict_risk():
    """
    Predict risk assessment

    Similar to /predict/trading but returns risk scores
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Implementation similar to predict_trading but with task='risk'
    return jsonify({'error': 'Risk prediction not yet implemented'}), 501


@app.route('/predict/portfolio', methods=['POST'])
def predict_portfolio():
    """
    Get portfolio recommendations

    Request body:
    {
        "tickers": ["AAPL", "GOOGL", "MSFT"],
        "lookback": 60
    }

    Response:
    {
        "weights": {"AAPL": 0.4, "GOOGL": 0.35, "MSFT": 0.25},
        "timestamp": "2024-01-01T00:00:00"
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Implementation for portfolio optimization
    return jsonify({'error': 'Portfolio prediction not yet implemented'}), 501


@app.route('/batch/predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple stocks

    Request body:
    {
        "tickers": ["AAPL", "GOOGL", "MSFT"],
        "lookback": 60
    }

    Response:
    {
        "predictions": {
            "AAPL": {"action": "BUY", "confidence": 0.85, ...},
            "GOOGL": {"action": "HOLD", "confidence": 0.60, ...},
            ...
        }
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        tickers = data.get('tickers', [])
        lookback = data.get('lookback', 60)

        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400

        predictions = {}

        for ticker in tickers:
            try:
                # Make individual prediction for each ticker
                result = predict_trading_for_ticker(ticker, lookback)
                predictions[ticker] = result
            except Exception as e:
                predictions[ticker] = {'error': str(e)}

        return jsonify({
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def predict_trading_for_ticker(ticker: str, lookback: int):
    """Helper function for batch predictions"""
    fetcher = MarketDataFetcher()
    from datetime import timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback + 30)

    df = fetcher.fetch_stock_data(
        ticker,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    df = TechnicalIndicators.add_indicators(df)

    ohlcv_data = df[['open', 'high', 'low', 'close', 'volume']].values[-lookback:]

    indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    if indicator_cols:
        indicators_data = df[indicator_cols].values[-lookback:]
    else:
        indicators_data = None

    # Normalize
    ohlcv_mean = ohlcv_data.mean(axis=0)
    ohlcv_std = ohlcv_data.std(axis=0) + 1e-8
    ohlcv_norm = (ohlcv_data - ohlcv_mean) / ohlcv_std

    if indicators_data is not None:
        indicators_mean = indicators_data.mean(axis=0)
        indicators_std = indicators_data.std(axis=0) + 1e-8
        indicators_norm = (indicators_data - indicators_mean) / indicators_std
    else:
        indicators_norm = None

    # Convert to tensors
    ohlcv_tensor = torch.tensor(ohlcv_norm, dtype=torch.float32).unsqueeze(0).to(device)
    indicators_tensor = torch.tensor(indicators_norm, dtype=torch.float32).unsqueeze(0).to(device) if indicators_norm is not None else None

    # Get prediction
    with torch.no_grad():
        predictions = model.predict_trading_action(
            ohlcv=ohlcv_tensor,
            indicators=indicators_tensor
        )

    action_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
    action = action_map[predictions['action'].item()]
    action_probs = predictions['action_probs'][0].cpu().numpy().tolist()

    return {
        'action': action,
        'action_probs': {
            'buy': action_probs[0],
            'sell': action_probs[1],
            'hold': action_probs[2]
        },
        'confidence': max(action_probs),
        'current_price': float(df['close'].iloc[-1])
    }


def main(args):
    print("=" * 60)
    print("Financial LLM API Server")
    print("=" * 60)

    # Load model
    load_model(args.checkpoint)

    print(f"\nStarting server on port {args.port}...")
    print("\nAvailable endpoints:")
    print("  GET  /health                - Health check")
    print("  GET  /model/info            - Model information")
    print("  POST /predict/trading       - Trading signal prediction")
    print("  POST /predict/risk          - Risk assessment")
    print("  POST /predict/portfolio     - Portfolio recommendation")
    print("  POST /batch/predict         - Batch predictions")
    print("\nExample request:")
    print('  curl -X POST http://localhost:8000/predict/trading \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"ticker": "AAPL", "lookback": 60}\'')
    print("\n" + "=" * 60)

    # Run server
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy Financial LLM as REST API')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port number')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')

    args = parser.parse_args()

    # Check dependencies
    try:
        from flask import Flask
        from flask_cors import CORS
    except ImportError:
        print("Error: Flask not installed")
        print("Install with: pip install flask flask-cors")
        sys.exit(1)

    main(args)

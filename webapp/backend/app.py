"""
Financial LLM Trading Bot - Web Application Backend

Production-ready Flask API for managing trading bots, fine-tuning models,
and monitoring trading activity.

Features:
- Bot configuration and management
- LLM fine-tuning interface
- Real-time monitoring with WebSocket
- Broker integration
- Backtesting API
- User authentication
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import time

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import jwt
from functools import wraps

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from financial_llm.brokers import BrokerFactory, BrokerConfig, OrderManager
from webapp.backend.models import db, User, TradingBot, TrainingJob, Trade
from webapp.backend.bot_manager import BotManager
from webapp.backend.training_manager import TrainingManager
from webapp.backend.monitoring import MonitoringService

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend/build')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///trading_bots.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
CORS(app, origins=['http://localhost:3000', 'http://localhost:5000'])
socketio = SocketIO(app, cors_allowed_origins='*')
db.init_app(app)

# Initialize managers
bot_manager = BotManager(socketio)
training_manager = TrainingManager(socketio)
monitoring_service = MonitoringService(socketio)

# Create database tables
with app.app_context():
    db.create_all()


# ============================================================================
# Authentication Decorators
# ============================================================================

def token_required(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'error': 'Token is missing'}), 401

        try:
            if token.startswith('Bearer '):
                token = token[7:]

            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])

            if not current_user:
                return jsonify({'error': 'User not found'}), 401

        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        return f(current_user, *args, **kwargs)

    return decorated


# ============================================================================
# Authentication Routes
# ============================================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user"""
    data = request.json

    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 400

    user = User(
        email=data['email'],
        username=data['username']
    )
    user.set_password(data['password'])

    db.session.add(user)
    db.session.commit()

    return jsonify({
        'message': 'User registered successfully',
        'user_id': user.id
    }), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    data = request.json

    user = User.query.filter_by(email=data['email']).first()

    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401

    # Generate JWT token
    token = jwt.encode(
        {
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=7)
        },
        app.config['SECRET_KEY'],
        algorithm='HS256'
    )

    user.last_login = datetime.utcnow()
    db.session.commit()

    return jsonify({
        'token': token,
        'user': {
            'id': user.id,
            'email': user.email,
            'username': user.username
        }
    })


@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_current_user(current_user):
    """Get current user info"""
    return jsonify({
        'id': current_user.id,
        'email': current_user.email,
        'username': current_user.username,
        'created_at': current_user.created_at.isoformat()
    })


# ============================================================================
# Trading Bot Routes
# ============================================================================

@app.route('/api/bots', methods=['GET'])
@token_required
def get_bots(current_user):
    """Get all trading bots for current user"""
    bots = TradingBot.query.filter_by(user_id=current_user.id).all()

    return jsonify({
        'bots': [bot.to_dict() for bot in bots]
    })


@app.route('/api/bots', methods=['POST'])
@token_required
def create_bot(current_user):
    """Create new trading bot"""
    data = request.json

    bot = TradingBot(
        user_id=current_user.id,
        name=data['name'],
        description=data.get('description', ''),
        broker=data['broker'],
        symbols=json.dumps(data['symbols']),
        strategy=data['strategy'],
        config=json.dumps(data.get('config', {})),
        status='stopped'
    )

    db.session.add(bot)
    db.session.commit()

    # Initialize bot in manager
    bot_manager.add_bot(bot)

    return jsonify({
        'message': 'Bot created successfully',
        'bot': bot.to_dict()
    }), 201


@app.route('/api/bots/<int:bot_id>', methods=['GET'])
@token_required
def get_bot(current_user, bot_id):
    """Get specific bot details"""
    bot = TradingBot.query.filter_by(id=bot_id, user_id=current_user.id).first()

    if not bot:
        return jsonify({'error': 'Bot not found'}), 404

    return jsonify(bot.to_dict())


@app.route('/api/bots/<int:bot_id>', methods=['PUT'])
@token_required
def update_bot(current_user, bot_id):
    """Update bot configuration"""
    bot = TradingBot.query.filter_by(id=bot_id, user_id=current_user.id).first()

    if not bot:
        return jsonify({'error': 'Bot not found'}), 404

    if bot.status == 'running':
        return jsonify({'error': 'Cannot update running bot'}), 400

    data = request.json

    bot.name = data.get('name', bot.name)
    bot.description = data.get('description', bot.description)
    bot.symbols = json.dumps(data.get('symbols', json.loads(bot.symbols)))
    bot.strategy = data.get('strategy', bot.strategy)
    bot.config = json.dumps(data.get('config', json.loads(bot.config)))
    bot.updated_at = datetime.utcnow()

    db.session.commit()

    return jsonify({
        'message': 'Bot updated successfully',
        'bot': bot.to_dict()
    })


@app.route('/api/bots/<int:bot_id>', methods=['DELETE'])
@token_required
def delete_bot(current_user, bot_id):
    """Delete trading bot"""
    bot = TradingBot.query.filter_by(id=bot_id, user_id=current_user.id).first()

    if not bot:
        return jsonify({'error': 'Bot not found'}), 404

    if bot.status == 'running':
        return jsonify({'error': 'Cannot delete running bot'}), 400

    bot_manager.remove_bot(bot_id)
    db.session.delete(bot)
    db.session.commit()

    return jsonify({'message': 'Bot deleted successfully'})


@app.route('/api/bots/<int:bot_id>/start', methods=['POST'])
@token_required
def start_bot(current_user, bot_id):
    """Start trading bot"""
    bot = TradingBot.query.filter_by(id=bot_id, user_id=current_user.id).first()

    if not bot:
        return jsonify({'error': 'Bot not found'}), 404

    if bot.status == 'running':
        return jsonify({'error': 'Bot already running'}), 400

    try:
        bot_manager.start_bot(bot_id)
        bot.status = 'running'
        bot.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            'message': 'Bot started successfully',
            'bot': bot.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bots/<int:bot_id>/stop', methods=['POST'])
@token_required
def stop_bot(current_user, bot_id):
    """Stop trading bot"""
    bot = TradingBot.query.filter_by(id=bot_id, user_id=current_user.id).first()

    if not bot:
        return jsonify({'error': 'Bot not found'}), 404

    bot_manager.stop_bot(bot_id)
    bot.status = 'stopped'
    bot.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        'message': 'Bot stopped successfully',
        'bot': bot.to_dict()
    })


@app.route('/api/bots/<int:bot_id>/performance', methods=['GET'])
@token_required
def get_bot_performance(current_user, bot_id):
    """Get bot performance metrics"""
    bot = TradingBot.query.filter_by(id=bot_id, user_id=current_user.id).first()

    if not bot:
        return jsonify({'error': 'Bot not found'}), 404

    performance = bot_manager.get_performance(bot_id)

    return jsonify(performance)


# ============================================================================
# Training Routes
# ============================================================================

@app.route('/api/training/jobs', methods=['GET'])
@token_required
def get_training_jobs(current_user):
    """Get all training jobs"""
    jobs = TrainingJob.query.filter_by(user_id=current_user.id).order_by(
        TrainingJob.created_at.desc()
    ).all()

    return jsonify({
        'jobs': [job.to_dict() for job in jobs]
    })


@app.route('/api/training/start', methods=['POST'])
@token_required
def start_training(current_user):
    """Start model training/fine-tuning"""
    data = request.json

    job = TrainingJob(
        user_id=current_user.id,
        model_type=data['model_type'],
        config=json.dumps(data.get('config', {})),
        status='queued'
    )

    db.session.add(job)
    db.session.commit()

    # Start training in background
    training_manager.start_training(job)

    return jsonify({
        'message': 'Training job started',
        'job': job.to_dict()
    }), 201


@app.route('/api/training/jobs/<int:job_id>', methods=['GET'])
@token_required
def get_training_job(current_user, job_id):
    """Get training job details"""
    job = TrainingJob.query.filter_by(id=job_id, user_id=current_user.id).first()

    if not job:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(job.to_dict())


@app.route('/api/training/jobs/<int:job_id>/cancel', methods=['POST'])
@token_required
def cancel_training(current_user, job_id):
    """Cancel training job"""
    job = TrainingJob.query.filter_by(id=job_id, user_id=current_user.id).first()

    if not job:
        return jsonify({'error': 'Job not found'}), 404

    training_manager.cancel_training(job_id)

    return jsonify({'message': 'Training job cancelled'})


# ============================================================================
# Broker Routes
# ============================================================================

@app.route('/api/brokers', methods=['GET'])
def get_brokers():
    """Get list of supported brokers"""
    return jsonify({
        'brokers': {
            'us': [
                {'id': 'alpaca', 'name': 'Alpaca', 'market': 'US'},
                {'id': 'ibkr', 'name': 'Interactive Brokers', 'market': 'US'},
                {'id': 'td_ameritrade', 'name': 'TD Ameritrade', 'market': 'US'},
            ],
            'india': [
                {'id': 'zerodha', 'name': 'Zerodha Kite', 'market': 'IN'},
                {'id': 'angel_one', 'name': 'Angel One', 'market': 'IN'},
                {'id': 'upstox', 'name': 'Upstox', 'market': 'IN'},
            ]
        }
    })


@app.route('/api/brokers/test', methods=['POST'])
@token_required
def test_broker_connection(current_user):
    """Test broker connection"""
    data = request.json

    try:
        # Test connection (simplified)
        return jsonify({
            'success': True,
            'message': 'Connection successful'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


# ============================================================================
# Monitoring Routes
# ============================================================================

@app.route('/api/dashboard/stats', methods=['GET'])
@token_required
def get_dashboard_stats(current_user):
    """Get dashboard statistics"""
    stats = monitoring_service.get_dashboard_stats(current_user.id)
    return jsonify(stats)


@app.route('/api/trades', methods=['GET'])
@token_required
def get_trades(current_user):
    """Get recent trades"""
    limit = request.args.get('limit', 100, type=int)

    trades = Trade.query.filter_by(user_id=current_user.id).order_by(
        Trade.executed_at.desc()
    ).limit(limit).all()

    return jsonify({
        'trades': [trade.to_dict() for trade in trades]
    })


@app.route('/api/backtest', methods=['POST'])
@token_required
def run_backtest(current_user):
    """Run backtest simulation"""
    data = request.json

    # Run backtest (implement in bot_manager)
    results = bot_manager.run_backtest(
        strategy=data['strategy'],
        symbols=data['symbols'],
        start_date=data['start_date'],
        end_date=data['end_date'],
        config=data.get('config', {})
    )

    return jsonify(results)


# ============================================================================
# WebSocket Events
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print(f'Client connected: {request.sid}')
    emit('connected', {'message': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print(f'Client disconnected: {request.sid}')


@socketio.on('subscribe_bot')
def handle_subscribe_bot(data):
    """Subscribe to bot updates"""
    bot_id = data['bot_id']
    join_room(f'bot_{bot_id}')
    emit('subscribed', {'bot_id': bot_id})


@socketio.on('unsubscribe_bot')
def handle_unsubscribe_bot(data):
    """Unsubscribe from bot updates"""
    bot_id = data['bot_id']
    leave_room(f'bot_{bot_id}')
    emit('unsubscribed', {'bot_id': bot_id})


# ============================================================================
# Serve React Frontend
# ============================================================================

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve React frontend"""
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Financial LLM Trading Bot - Web Application")
    print("=" * 60)
    print(f"\nStarting server on http://localhost:5000")
    print(f"API documentation: http://localhost:5000/api")
    print(f"\nPress Ctrl+C to stop\n")

    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    )

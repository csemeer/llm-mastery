"""
Database Models

SQLAlchemy models for the trading bot application.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()


class User(db.Model):
    """User model"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)

    # Relationships
    bots = db.relationship('TradingBot', backref='user', lazy=True, cascade='all, delete-orphan')
    training_jobs = db.relationship('TrainingJob', backref='user', lazy=True, cascade='all, delete-orphan')
    trades = db.relationship('Trade', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None
        }


class TradingBot(db.Model):
    """Trading bot configuration"""
    __tablename__ = 'trading_bots'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    broker = db.Column(db.String(50), nullable=False)
    symbols = db.Column(db.Text, nullable=False)  # JSON array
    strategy = db.Column(db.String(50), nullable=False)
    config = db.Column(db.Text)  # JSON configuration
    status = db.Column(db.String(20), default='stopped')  # stopped, running, paused, error
    model_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_run = db.Column(db.DateTime)

    # Performance metrics
    total_trades = db.Column(db.Integer, default=0)
    winning_trades = db.Column(db.Integer, default=0)
    total_pnl = db.Column(db.Float, default=0.0)

    # Relationships
    trades = db.relationship('Trade', backref='bot', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'broker': self.broker,
            'symbols': json.loads(self.symbols),
            'strategy': self.strategy,
            'config': json.loads(self.config) if self.config else {},
            'status': self.status,
            'model_path': self.model_path,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'performance': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
                'total_pnl': self.total_pnl
            }
        }


class TrainingJob(db.Model):
    """Model training/fine-tuning job"""
    __tablename__ = 'training_jobs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # base, fine_tune
    config = db.Column(db.Text)  # JSON configuration
    status = db.Column(db.String(20), default='queued')  # queued, running, completed, failed, cancelled
    progress = db.Column(db.Float, default=0.0)
    current_epoch = db.Column(db.Integer, default=0)
    total_epochs = db.Column(db.Integer)
    loss = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    model_path = db.Column(db.String(255))
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'model_type': self.model_type,
            'config': json.loads(self.config) if self.config else {},
            'status': self.status,
            'progress': self.progress,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'loss': self.loss,
            'accuracy': self.accuracy,
            'model_path': self.model_path,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class Trade(db.Model):
    """Trade execution record"""
    __tablename__ = 'trades'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    bot_id = db.Column(db.Integer, db.ForeignKey('trading_bots.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False, index=True)
    side = db.Column(db.String(10), nullable=False)  # buy, sell
    quantity = db.Column(db.Float, nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    pnl = db.Column(db.Float)
    pnl_percent = db.Column(db.Float)
    status = db.Column(db.String(20), default='open')  # open, closed
    strategy = db.Column(db.String(50))
    executed_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    closed_at = db.Column(db.DateTime)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'bot_id': self.bot_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'status': self.status,
            'strategy': self.strategy,
            'executed_at': self.executed_at.isoformat(),
            'closed_at': self.closed_at.isoformat() if self.closed_at else None
        }

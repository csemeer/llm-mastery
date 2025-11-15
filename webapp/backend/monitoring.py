"""
Monitoring Service

Real-time monitoring and dashboard statistics.
"""

from datetime import datetime, timedelta
from sqlalchemy import func
from models import db, TradingBot, Trade, TrainingJob


class MonitoringService:
    """Monitoring and statistics service"""

    def __init__(self, socketio):
        self.socketio = socketio

    def get_dashboard_stats(self, user_id: int) -> dict:
        """Get dashboard statistics for user"""

        # Bot statistics
        total_bots = TradingBot.query.filter_by(user_id=user_id).count()
        running_bots = TradingBot.query.filter_by(user_id=user_id, status='running').count()

        # Trade statistics
        total_trades = Trade.query.filter_by(user_id=user_id).count()
        winning_trades = Trade.query.filter_by(user_id=user_id).filter(
            Trade.pnl > 0
        ).count()

        # PnL statistics
        total_pnl = db.session.query(func.sum(Trade.pnl)).filter_by(
            user_id=user_id
        ).scalar() or 0.0

        # Today's PnL
        today = datetime.utcnow().date()
        today_pnl = db.session.query(func.sum(Trade.pnl)).filter(
            Trade.user_id == user_id,
            func.date(Trade.executed_at) == today
        ).scalar() or 0.0

        # Recent trades
        recent_trades = Trade.query.filter_by(user_id=user_id).order_by(
            Trade.executed_at.desc()
        ).limit(10).all()

        # Training job statistics
        total_training_jobs = TrainingJob.query.filter_by(user_id=user_id).count()
        running_training_jobs = TrainingJob.query.filter_by(
            user_id=user_id,
            status='running'
        ).count()

        # Active bots list
        active_bots = TradingBot.query.filter_by(
            user_id=user_id,
            status='running'
        ).all()

        return {
            'bots': {
                'total': total_bots,
                'running': running_bots,
                'active_list': [bot.to_dict() for bot in active_bots]
            },
            'trades': {
                'total': total_trades,
                'winning': winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'recent': [trade.to_dict() for trade in recent_trades]
            },
            'performance': {
                'total_pnl': total_pnl,
                'today_pnl': today_pnl,
                'pnl_change': self._calculate_pnl_change(user_id)
            },
            'training': {
                'total_jobs': total_training_jobs,
                'running_jobs': running_training_jobs
            }
        }

    def _calculate_pnl_change(self, user_id: int) -> float:
        """Calculate PnL change percentage"""
        today = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)

        today_pnl = db.session.query(func.sum(Trade.pnl)).filter(
            Trade.user_id == user_id,
            func.date(Trade.executed_at) == today
        ).scalar() or 0.0

        yesterday_pnl = db.session.query(func.sum(Trade.pnl)).filter(
            Trade.user_id == user_id,
            func.date(Trade.executed_at) == yesterday
        ).scalar() or 1.0

        if yesterday_pnl == 0:
            return 100.0 if today_pnl > 0 else 0.0

        return ((today_pnl - yesterday_pnl) / abs(yesterday_pnl)) * 100

    def get_performance_chart_data(self, user_id: int, days: int = 30) -> dict:
        """Get performance chart data"""
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days)

        # Query daily PnL
        daily_pnl = db.session.query(
            func.date(Trade.executed_at).label('date'),
            func.sum(Trade.pnl).label('pnl')
        ).filter(
            Trade.user_id == user_id,
            func.date(Trade.executed_at) >= start_date
        ).group_by(
            func.date(Trade.executed_at)
        ).all()

        # Format data for charts
        labels = []
        values = []
        cumulative = 0

        for day_pnl in daily_pnl:
            labels.append(day_pnl.date.isoformat())
            cumulative += day_pnl.pnl
            values.append(cumulative)

        return {
            'labels': labels,
            'values': values
        }

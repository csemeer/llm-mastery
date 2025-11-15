"""
Bot Manager

Manages trading bot lifecycle, execution, and monitoring.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
import json

from models import db, TradingBot, Trade


class BotManager:
    """Manages trading bots"""

    def __init__(self, socketio):
        self.socketio = socketio
        self.running_bots: Dict[int, threading.Thread] = {}
        self.bot_stop_flags: Dict[int, threading.Event] = {}

    def add_bot(self, bot: TradingBot):
        """Add bot to manager"""
        if bot.id not in self.running_bots:
            self.bot_stop_flags[bot.id] = threading.Event()

    def remove_bot(self, bot_id: int):
        """Remove bot from manager"""
        if bot_id in self.running_bots:
            self.stop_bot(bot_id)

        if bot_id in self.bot_stop_flags:
            del self.bot_stop_flags[bot_id]

    def start_bot(self, bot_id: int):
        """Start trading bot"""
        if bot_id in self.running_bots and self.running_bots[bot_id].is_alive():
            raise ValueError("Bot is already running")

        # Get bot from database
        bot = TradingBot.query.get(bot_id)
        if not bot:
            raise ValueError("Bot not found")

        # Create stop flag if not exists
        if bot_id not in self.bot_stop_flags:
            self.bot_stop_flags[bot_id] = threading.Event()

        # Clear stop flag
        self.bot_stop_flags[bot_id].clear()

        # Start bot thread
        thread = threading.Thread(
            target=self._run_bot,
            args=(bot_id,),
            daemon=True
        )
        thread.start()

        self.running_bots[bot_id] = thread

        # Emit status update
        self.socketio.emit('bot_status', {
            'bot_id': bot_id,
            'status': 'running'
        }, room=f'bot_{bot_id}')

    def stop_bot(self, bot_id: int):
        """Stop trading bot"""
        if bot_id in self.bot_stop_flags:
            self.bot_stop_flags[bot_id].set()

        if bot_id in self.running_bots:
            thread = self.running_bots[bot_id]
            thread.join(timeout=5)
            del self.running_bots[bot_id]

        # Emit status update
        self.socketio.emit('bot_status', {
            'bot_id': bot_id,
            'status': 'stopped'
        }, room=f'bot_{bot_id}')

    def _run_bot(self, bot_id: int):
        """Main bot execution loop"""
        print(f"Starting bot {bot_id}")

        bot = TradingBot.query.get(bot_id)
        stop_flag = self.bot_stop_flags[bot_id]

        try:
            while not stop_flag.is_set():
                # Update last run time
                bot.last_run = datetime.utcnow()
                db.session.commit()

                # Execute trading logic (simplified)
                self._execute_trading_logic(bot)

                # Emit heartbeat
                self.socketio.emit('bot_heartbeat', {
                    'bot_id': bot_id,
                    'timestamp': datetime.utcnow().isoformat()
                }, room=f'bot_{bot_id}')

                # Sleep before next iteration (e.g., 60 seconds)
                stop_flag.wait(timeout=60)

        except Exception as e:
            print(f"Error in bot {bot_id}: {str(e)}")
            bot.status = 'error'
            db.session.commit()

            self.socketio.emit('bot_error', {
                'bot_id': bot_id,
                'error': str(e)
            }, room=f'bot_{bot_id}')

        finally:
            print(f"Stopped bot {bot_id}")

    def _execute_trading_logic(self, bot: TradingBot):
        """Execute trading logic for bot"""
        # This is a simplified version
        # In production, this would:
        # 1. Fetch market data
        # 2. Run model predictions
        # 3. Execute trades based on strategy
        # 4. Update database with trade records

        symbols = json.loads(bot.symbols)

        # Simulate trading logic
        for symbol in symbols[:1]:  # Process one symbol per iteration
            # Generate signal (simplified)
            signal = self._generate_signal(bot, symbol)

            if signal:
                # Execute trade
                trade = self._execute_trade(bot, symbol, signal)

                # Emit trade update
                if trade:
                    self.socketio.emit('new_trade', {
                        'bot_id': bot.id,
                        'trade': trade.to_dict()
                    }, room=f'bot_{bot.id}')

    def _generate_signal(self, bot: TradingBot, symbol: str) -> Optional[Dict]:
        """Generate trading signal"""
        # Simplified signal generation
        # In production, this would use the Financial LLM model
        return None

    def _execute_trade(self, bot: TradingBot, symbol: str, signal: Dict) -> Optional[Trade]:
        """Execute trade based on signal"""
        # Simplified trade execution
        # In production, this would interface with broker API
        return None

    def get_performance(self, bot_id: int) -> Dict:
        """Get bot performance metrics"""
        bot = TradingBot.query.get(bot_id)

        if not bot:
            return {}

        # Calculate performance metrics
        trades = Trade.query.filter_by(bot_id=bot_id).all()

        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
        total_pnl = sum([t.pnl for t in trades if t.pnl])

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'recent_trades': [t.to_dict() for t in trades[-10:]]
        }

    def run_backtest(self, strategy: str, symbols: list, start_date: str,
                     end_date: str, config: dict) -> Dict:
        """Run backtest simulation"""
        # Simplified backtest
        # In production, this would:
        # 1. Load historical data
        # 2. Run strategy on historical data
        # 3. Calculate performance metrics
        # 4. Return results with charts

        return {
            'summary': {
                'total_return': 15.5,
                'sharpe_ratio': 1.2,
                'max_drawdown': -8.3,
                'win_rate': 0.58,
                'total_trades': 125
            },
            'equity_curve': [],
            'trades': []
        }

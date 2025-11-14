"""
Trading Strategy Examples using Financial LLM

Demonstrates various trading strategies:
1. Simple buy/hold/sell based on model predictions
2. Momentum trading
3. Mean reversion
4. Multi-stock portfolio
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_llm.models.financial_llm import FinancialLLM
from financial_llm.data_processors.market_data import MarketDataFetcher, TechnicalIndicators


class TradingStrategy:
    """
    Base class for trading strategies
    """
    def __init__(self, model: FinancialLLM, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict(self, ohlcv, indicators=None):
        """
        Get model prediction for current market state
        """
        with torch.no_grad():
            ohlcv_tensor = torch.tensor(ohlcv, dtype=torch.float32).unsqueeze(0).to(self.device)

            if indicators is not None:
                indicators_tensor = torch.tensor(indicators, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                indicators_tensor = None

            predictions = self.model.predict_trading_action(
                ohlcv=ohlcv_tensor,
                indicators=indicators_tensor
            )

        return predictions

    def generate_signal(self, data, idx):
        """
        Generate trading signal for given data point
        Returns: 'BUY', 'SELL', or 'HOLD'
        """
        raise NotImplementedError


class SimpleLLMStrategy(TradingStrategy):
    """
    Simple strategy that follows model predictions directly
    """
    def __init__(self, model, device='cpu', confidence_threshold=0.5):
        super().__init__(model, device)
        self.confidence_threshold = confidence_threshold

    def generate_signal(self, data, idx, lookback=60):
        """
        Generate signal based on model prediction
        """
        if idx < lookback:
            return 'HOLD', 0.0

        # Get recent data
        ohlcv = data.iloc[idx-lookback:idx][['open', 'high', 'low', 'close', 'volume']].values

        # Normalize
        mean = ohlcv.mean(axis=0)
        std = ohlcv.std(axis=0) + 1e-8
        ohlcv_norm = (ohlcv - mean) / std

        # Get indicators if available
        indicator_cols = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        if indicator_cols:
            indicators = data.iloc[idx-lookback:idx][indicator_cols].values
            indicators_mean = indicators.mean(axis=0)
            indicators_std = indicators.std(axis=0) + 1e-8
            indicators_norm = (indicators - indicators_mean) / indicators_std
        else:
            indicators_norm = None

        # Get prediction
        predictions = self.predict(ohlcv_norm, indicators_norm)

        action = predictions['action'].item()
        confidence = predictions['action_probs'][0][action].item()

        # Only act if confidence is high enough
        if confidence < self.confidence_threshold:
            return 'HOLD', confidence

        action_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
        return action_map[action], confidence


class MomentumLLMStrategy(TradingStrategy):
    """
    Momentum strategy enhanced with LLM predictions
    """
    def __init__(self, model, device='cpu', momentum_window=20, llm_weight=0.6):
        super().__init__(model, device)
        self.momentum_window = momentum_window
        self.llm_weight = llm_weight

    def generate_signal(self, data, idx, lookback=60):
        """
        Combine momentum indicator with LLM prediction
        """
        if idx < max(lookback, self.momentum_window):
            return 'HOLD', 0.0

        # Calculate momentum
        close_prices = data['close'].values
        momentum = (close_prices[idx] - close_prices[idx - self.momentum_window]) / close_prices[idx - self.momentum_window]

        # Get LLM prediction
        ohlcv = data.iloc[idx-lookback:idx][['open', 'high', 'low', 'close', 'volume']].values
        mean = ohlcv.mean(axis=0)
        std = ohlcv.std(axis=0) + 1e-8
        ohlcv_norm = (ohlcv - mean) / std

        indicator_cols = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        if indicator_cols:
            indicators = data.iloc[idx-lookback:idx][indicator_cols].values
            indicators_mean = indicators.mean(axis=0)
            indicators_std = indicators.std(axis=0) + 1e-8
            indicators_norm = (indicators - indicators_mean) / indicators_std
        else:
            indicators_norm = None

        predictions = self.predict(ohlcv_norm, indicators_norm)
        action = predictions['action'].item()

        # Combine signals
        # Momentum: positive = buy signal, negative = sell signal
        # LLM: 0 = buy, 1 = sell, 2 = hold

        momentum_signal = 1 if momentum > 0 else -1 if momentum < 0 else 0
        llm_signal = 1 if action == 0 else -1 if action == 1 else 0

        combined_signal = self.llm_weight * llm_signal + (1 - self.llm_weight) * momentum_signal

        if combined_signal > 0.3:
            signal = 'BUY'
        elif combined_signal < -0.3:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        confidence = abs(combined_signal)

        return signal, confidence


class MeanReversionLLMStrategy(TradingStrategy):
    """
    Mean reversion strategy with LLM filtering
    """
    def __init__(self, model, device='cpu', lookback=20, num_std=2.0):
        super().__init__(model, device)
        self.lookback = lookback
        self.num_std = num_std

    def generate_signal(self, data, idx, model_lookback=60):
        """
        Buy when price is below mean, sell when above (if LLM agrees)
        """
        if idx < max(model_lookback, self.lookback):
            return 'HOLD', 0.0

        # Calculate mean and std
        close_prices = data['close'].values
        recent_prices = close_prices[idx - self.lookback:idx]
        mean_price = recent_prices.mean()
        std_price = recent_prices.std()

        current_price = close_prices[idx]
        z_score = (current_price - mean_price) / (std_price + 1e-8)

        # Get LLM confirmation
        ohlcv = data.iloc[idx-model_lookback:idx][['open', 'high', 'low', 'close', 'volume']].values
        mean = ohlcv.mean(axis=0)
        std = ohlcv.std(axis=0) + 1e-8
        ohlcv_norm = (ohlcv - mean) / std

        indicator_cols = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        if indicator_cols:
            indicators = data.iloc[idx-model_lookback:idx][indicator_cols].values
            indicators_mean = indicators.mean(axis=0)
            indicators_std = indicators.std(axis=0) + 1e-8
            indicators_norm = (indicators - indicators_mean) / indicators_std
        else:
            indicators_norm = None

        predictions = self.predict(ohlcv_norm, indicators_norm)
        llm_action = predictions['action'].item()

        # Mean reversion logic
        if z_score < -self.num_std and llm_action == 0:  # Price low and LLM says buy
            return 'BUY', abs(z_score) / self.num_std
        elif z_score > self.num_std and llm_action == 1:  # Price high and LLM says sell
            return 'SELL', abs(z_score) / self.num_std
        else:
            return 'HOLD', 0.0


def simulate_strategy(strategy: TradingStrategy, data: pd.DataFrame, initial_capital: float = 100000):
    """
    Simulate trading strategy on historical data

    Returns:
        results: Dictionary with performance metrics
    """
    print("Simulating strategy...")
    print("-" * 60)

    capital = initial_capital
    position = 0  # Number of shares held
    position_value = 0
    trades = []
    portfolio_values = []

    for idx in range(len(data)):
        current_price = data['close'].iloc[idx]
        current_date = data.index[idx] if hasattr(data.index[idx], 'strftime') else idx

        # Generate signal
        signal, confidence = strategy.generate_signal(data, idx)

        # Execute trade
        if signal == 'BUY' and capital > current_price:
            # Buy as many shares as possible
            shares_to_buy = int(capital * 0.95 / current_price)  # Use 95% of capital

            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                capital -= cost
                position += shares_to_buy

                trades.append({
                    'date': current_date,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'confidence': confidence
                })

        elif signal == 'SELL' and position > 0:
            # Sell all shares
            proceeds = position * current_price
            capital += proceeds

            trades.append({
                'date': current_date,
                'action': 'SELL',
                'price': current_price,
                'shares': position,
                'confidence': confidence
            })

            position = 0

        # Calculate portfolio value
        position_value = position * current_price
        total_value = capital + position_value
        portfolio_values.append(total_value)

    # Final liquidation
    if position > 0:
        final_price = data['close'].iloc[-1]
        capital += position * final_price
        position = 0

    final_value = capital

    # Calculate metrics
    total_return = (final_value - initial_capital) / initial_capital
    num_trades = len(trades)

    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']

    print(f"\nSimulation Results:")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Final Value: ${final_value:,.2f}")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Number of Trades: {num_trades}")
    print(f"    - Buys: {len(buy_trades)}")
    print(f"    - Sells: {len(sell_trades)}")

    # Calculate Sharpe ratio (simplified)
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        sharpe_ratio = 0

    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'num_trades': num_trades,
        'trades': trades,
        'portfolio_values': portfolio_values,
        'sharpe_ratio': sharpe_ratio
    }


def main(args):
    print("=" * 60)
    print("Financial LLM Trading Strategy Simulator")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train a model first using train_financial_llm.py")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create model (simplified - in production, save config in checkpoint)
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
        dropout=0.1
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded")

    # Fetch data
    print(f"\nFetching {args.ticker} data...")
    fetcher = MarketDataFetcher()

    try:
        data = fetcher.fetch_stock_data(args.ticker, args.start_date, args.end_date)
        data = TechnicalIndicators.add_indicators(data)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    print(f"✓ Fetched {len(data)} records")

    # Create strategy
    print(f"\nCreating {args.strategy} strategy...")

    if args.strategy == 'simple':
        strategy = SimpleLLMStrategy(model, device, confidence_threshold=args.confidence)
    elif args.strategy == 'momentum':
        strategy = MomentumLLMStrategy(model, device, momentum_window=args.momentum_window)
    elif args.strategy == 'mean_reversion':
        strategy = MeanReversionLLMStrategy(model, device, lookback=args.lookback)
    else:
        print(f"Unknown strategy: {args.strategy}")
        return

    print("✓ Strategy created")

    # Run simulation
    results = simulate_strategy(strategy, data, initial_capital=args.initial_capital)

    # Show sample trades
    print(f"\nSample Trades (first 5):")
    print("-" * 60)
    for trade in results['trades'][:5]:
        print(f"  {trade['date']}: {trade['action']} {trade['shares']} shares @ ${trade['price']:.2f} (confidence: {trade['confidence']:.2f})")

    if len(results['trades']) > 5:
        print(f"  ... and {len(results['trades']) - 5} more trades")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading Strategy Simulator')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Stock ticker')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-01-01',
                       help='End date (YYYY-MM-DD)')

    parser.add_argument('--strategy', type=str, default='simple',
                       choices=['simple', 'momentum', 'mean_reversion'],
                       help='Trading strategy')

    parser.add_argument('--initial-capital', type=float, default=100000,
                       help='Initial capital')

    # Strategy-specific parameters
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for simple strategy')
    parser.add_argument('--momentum-window', type=int, default=20,
                       help='Momentum window for momentum strategy')
    parser.add_argument('--lookback', type=int, default=20,
                       help='Lookback period for mean reversion')

    args = parser.parse_args()

    main(args)

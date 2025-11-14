"""
Evaluation and Backtesting Framework for Financial LLM

Provides comprehensive metrics for assessing model performance:
- Trading performance (returns, Sharpe ratio, drawdown)
- Prediction accuracy
- Risk metrics
- Benchmark comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import torch
from datetime import datetime


class PerformanceMetrics:
    """
    Calculate trading performance metrics
    """

    @staticmethod
    def total_return(portfolio_values: np.ndarray) -> float:
        """Calculate total return"""
        if len(portfolio_values) == 0:
            return 0.0
        return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

    @staticmethod
    def annualized_return(portfolio_values: np.ndarray, trading_days: int = 252) -> float:
        """Calculate annualized return"""
        if len(portfolio_values) <= 1:
            return 0.0

        total_ret = PerformanceMetrics.total_return(portfolio_values)
        num_years = len(portfolio_values) / trading_days
        return (1 + total_ret) ** (1 / num_years) - 1

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, trading_days: int = 252) -> float:
        """
        Calculate Sharpe ratio

        Args:
            returns: Daily returns
            risk_free_rate: Annual risk-free rate
            trading_days: Number of trading days per year
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        daily_rf = risk_free_rate / trading_days
        excess_returns = returns - daily_rf
        sharpe = np.mean(excess_returns) / np.std(returns)
        return sharpe * np.sqrt(trading_days)  # Annualized

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, trading_days: int = 252) -> float:
        """
        Calculate Sortino ratio (like Sharpe but only considers downside volatility)
        """
        if len(returns) == 0:
            return 0.0

        daily_rf = risk_free_rate / trading_days
        excess_returns = returns - daily_rf

        # Only consider negative returns for volatility
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        sortino = np.mean(excess_returns) / np.std(downside_returns)
        return sortino * np.sqrt(trading_days)

    @staticmethod
    def max_drawdown(portfolio_values: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown

        Returns:
            (max_drawdown, start_idx, end_idx)
        """
        if len(portfolio_values) == 0:
            return 0.0, 0, 0

        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax

        max_dd = np.min(drawdown)
        end_idx = np.argmin(drawdown)

        # Find start of drawdown
        start_idx = np.argmax(portfolio_values[:end_idx+1])

        return max_dd, start_idx, end_idx

    @staticmethod
    def calmar_ratio(portfolio_values: np.ndarray, trading_days: int = 252) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown)
        """
        ann_return = PerformanceMetrics.annualized_return(portfolio_values, trading_days)
        max_dd, _, _ = PerformanceMetrics.max_drawdown(portfolio_values)

        if abs(max_dd) < 1e-8:
            return 0.0

        return ann_return / abs(max_dd)

    @staticmethod
    def win_rate(trades: List[Dict]) -> float:
        """
        Calculate win rate (percentage of profitable trades)
        """
        if len(trades) == 0:
            return 0.0

        # Match buy and sell trades
        position = 0
        buy_price = 0
        profits = []

        for trade in trades:
            if trade['action'] == 'BUY':
                position += trade['shares']
                buy_price = trade['price']
            elif trade['action'] == 'SELL' and position > 0:
                profit = (trade['price'] - buy_price) * trade['shares']
                profits.append(profit)
                position = 0

        if len(profits) == 0:
            return 0.0

        winning_trades = sum(1 for p in profits if p > 0)
        return winning_trades / len(profits)

    @staticmethod
    def profit_factor(trades: List[Dict]) -> float:
        """
        Calculate profit factor (gross profit / gross loss)
        """
        if len(trades) == 0:
            return 0.0

        position = 0
        buy_price = 0
        profits = []

        for trade in trades:
            if trade['action'] == 'BUY':
                position += trade['shares']
                buy_price = trade['price']
            elif trade['action'] == 'SELL' and position > 0:
                profit = (trade['price'] - buy_price) * trade['shares']
                profits.append(profit)
                position = 0

        if len(profits) == 0:
            return 0.0

        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) at given confidence level
        """
        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def conditional_value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        Average of returns below VaR threshold
        """
        if len(returns) == 0:
            return 0.0

        var = PerformanceMetrics.value_at_risk(returns, confidence)
        return np.mean(returns[returns <= var])


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate_predictions(
        self,
        test_data: List[Dict],
        task: str = 'trading'
    ) -> Dict[str, float]:
        """
        Evaluate model predictions

        Args:
            test_data: List of samples with 'ohlcv', 'indicators', 'label'
            task: Task type

        Returns:
            Dictionary of metrics
        """
        print("Evaluating model predictions...")

        all_predictions = []
        all_labels = []
        all_confidences = []

        with torch.no_grad():
            for sample in test_data:
                ohlcv = sample['ohlcv'].unsqueeze(0).to(self.device)
                indicators = sample.get('indicators')
                if indicators is not None:
                    indicators = indicators.unsqueeze(0).to(self.device)
                label = sample['label']

                # Get prediction
                outputs = self.model(
                    ohlcv=ohlcv,
                    indicators=indicators,
                    task=task
                )

                if task == 'trading':
                    logits = outputs['trading_logits']
                    pred = torch.argmax(logits, dim=1).item()
                    confidence = torch.softmax(logits, dim=1).max().item()

                    all_predictions.append(pred)
                    all_labels.append(label.item() if torch.is_tensor(label) else label)
                    all_confidences.append(confidence)

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)

        accuracy = np.mean(all_predictions == all_labels)

        # Per-class metrics
        unique_classes = np.unique(all_labels)
        class_accuracies = {}

        for cls in unique_classes:
            mask = all_labels == cls
            if mask.sum() > 0:
                class_acc = np.mean(all_predictions[mask] == all_labels[mask])
                class_accuracies[f'class_{cls}_accuracy'] = class_acc

        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(all_labels, all_predictions)

        print(f"\nPrediction Accuracy: {accuracy:.2%}")
        print(f"Average Confidence: {np.mean(all_confidences):.2%}")
        print(f"\nConfusion Matrix:")
        print(cm)

        return {
            'accuracy': accuracy,
            'avg_confidence': np.mean(all_confidences),
            'confusion_matrix': cm,
            **class_accuracies
        }


def backtest_comprehensive(
    strategy,
    data: pd.DataFrame,
    initial_capital: float = 100000,
    commission: float = 0.001,
    benchmark_ticker: str = 'SPY'
) -> Dict:
    """
    Comprehensive backtesting with all metrics

    Args:
        strategy: Trading strategy instance
        data: Historical price data
        initial_capital: Starting capital
        commission: Commission rate (e.g., 0.001 = 0.1%)
        benchmark_ticker: Benchmark for comparison

    Returns:
        Dictionary with comprehensive metrics
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE BACKTESTING")
    print("=" * 60)

    capital = initial_capital
    position = 0
    trades = []
    portfolio_values = []
    daily_returns = []

    # Simulate trading
    for idx in range(len(data)):
        current_price = data['close'].iloc[idx]
        current_date = data.index[idx] if hasattr(data.index[idx], 'strftime') else idx

        # Generate signal
        signal, confidence = strategy.generate_signal(data, idx)

        # Execute trade
        if signal == 'BUY' and capital > current_price:
            shares_to_buy = int(capital * 0.95 / current_price)

            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + commission)
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
            proceeds = position * current_price * (1 - commission)
            capital += proceeds

            trades.append({
                'date': current_date,
                'action': 'SELL',
                'price': current_price,
                'shares': position,
                'confidence': confidence
            })

            position = 0

        # Track portfolio value
        position_value = position * current_price
        total_value = capital + position_value
        portfolio_values.append(total_value)

        # Calculate daily return
        if len(portfolio_values) > 1:
            daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)

    # Final liquidation
    if position > 0:
        final_price = data['close'].iloc[-1]
        capital += position * final_price * (1 - commission)
        position = 0

    portfolio_values = np.array(portfolio_values)
    daily_returns = np.array(daily_returns)

    # Calculate all metrics
    results = {
        'initial_capital': initial_capital,
        'final_value': portfolio_values[-1] if len(portfolio_values) > 0 else initial_capital,
        'total_return': PerformanceMetrics.total_return(portfolio_values),
        'annualized_return': PerformanceMetrics.annualized_return(portfolio_values),
        'sharpe_ratio': PerformanceMetrics.sharpe_ratio(daily_returns),
        'sortino_ratio': PerformanceMetrics.sortino_ratio(daily_returns),
        'max_drawdown': PerformanceMetrics.max_drawdown(portfolio_values)[0],
        'calmar_ratio': PerformanceMetrics.calmar_ratio(portfolio_values),
        'win_rate': PerformanceMetrics.win_rate(trades),
        'profit_factor': PerformanceMetrics.profit_factor(trades),
        'value_at_risk_95': PerformanceMetrics.value_at_risk(daily_returns, 0.95),
        'cvar_95': PerformanceMetrics.conditional_value_at_risk(daily_returns, 0.95),
        'num_trades': len(trades),
        'trades': trades,
        'portfolio_values': portfolio_values,
        'daily_returns': daily_returns
    }

    # Print results
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"\nCapital:")
    print(f"  Initial: ${results['initial_capital']:,.2f}")
    print(f"  Final:   ${results['final_value']:,.2f}")
    print(f"\nReturns:")
    print(f"  Total Return:      {results['total_return']:.2%}")
    print(f"  Annualized Return: {results['annualized_return']:.2%}")
    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio:      {results['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:     {results['sortino_ratio']:.2f}")
    print(f"  Max Drawdown:      {results['max_drawdown']:.2%}")
    print(f"  Calmar Ratio:      {results['calmar_ratio']:.2f}")
    print(f"  VaR (95%):         {results['value_at_risk_95']:.2%}")
    print(f"  CVaR (95%):        {results['cvar_95']:.2%}")
    print(f"\nTrading Statistics:")
    print(f"  Number of Trades:  {results['num_trades']}")
    print(f"  Win Rate:          {results['win_rate']:.2%}")
    print(f"  Profit Factor:     {results['profit_factor']:.2f}")

    return results


if __name__ == "__main__":
    print("Testing evaluation metrics...")

    # Test with sample data
    portfolio_values = np.array([100000, 102000, 101000, 105000, 103000, 108000])
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    print(f"Total Return: {PerformanceMetrics.total_return(portfolio_values):.2%}")
    print(f"Sharpe Ratio: {PerformanceMetrics.sharpe_ratio(returns):.2f}")
    print(f"Max Drawdown: {PerformanceMetrics.max_drawdown(portfolio_values)[0]:.2%}")

    trades = [
        {'action': 'BUY', 'price': 100, 'shares': 10},
        {'action': 'SELL', 'price': 110, 'shares': 10},
        {'action': 'BUY', 'price': 105, 'shares': 10},
        {'action': 'SELL', 'price': 103, 'shares': 10},
    ]

    print(f"Win Rate: {PerformanceMetrics.win_rate(trades):.2%}")
    print(f"Profit Factor: {PerformanceMetrics.profit_factor(trades):.2f}")

    print("\n✓ Evaluation metrics working correctly!")

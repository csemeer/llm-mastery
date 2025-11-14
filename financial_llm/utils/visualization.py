"""
Visualization Tools for Financial LLM

Provides visualization for:
- Portfolio performance
- Predictions vs actual
- Attention weights
- Feature importance
- Risk metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("plotly not available. Install with: pip install plotly")


class PerformanceVisualizer:
    """
    Visualize trading performance and backtesting results
    """

    @staticmethod
    def plot_portfolio_value(
        portfolio_values: np.ndarray,
        dates: Optional[List] = None,
        benchmark_values: Optional[np.ndarray] = None,
        title: str = "Portfolio Value Over Time",
        save_path: Optional[str] = None
    ):
        """
        Plot portfolio value over time with optional benchmark
        """
        plt.figure(figsize=(14, 7))

        x = dates if dates is not None else range(len(portfolio_values))

        plt.plot(x, portfolio_values, label='Strategy', linewidth=2, color='#2E86AB')

        if benchmark_values is not None:
            plt.plot(x, benchmark_values, label='Benchmark', linewidth=2,
                    color='#A23B72', linestyle='--', alpha=0.7)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date' if dates else 'Time', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        if dates:
            plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_drawdown(
        portfolio_values: np.ndarray,
        dates: Optional[List] = None,
        title: str = "Drawdown Analysis",
        save_path: Optional[str] = None
    ):
        """
        Plot drawdown over time
        """
        # Calculate drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax * 100  # Percentage

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        x = dates if dates is not None else range(len(portfolio_values))

        # Portfolio value
        ax1.plot(x, portfolio_values, linewidth=2, color='#2E86AB')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2.fill_between(x, drawdown, 0, color='#E63946', alpha=0.3)
        ax2.plot(x, drawdown, color='#E63946', linewidth=2)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date' if dates else 'Time', fontsize=12)
        ax2.grid(True, alpha=0.3)

        if dates:
            plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_returns_distribution(
        daily_returns: np.ndarray,
        title: str = "Returns Distribution",
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of daily returns
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(daily_returns * 100, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Return')
        axes[0].set_xlabel('Daily Return (%)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Returns Histogram', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(daily_returns, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_trade_analysis(
        trades: List[Dict],
        title: str = "Trade Analysis",
        save_path: Optional[str] = None
    ):
        """
        Analyze trades: entry/exit points, profits, etc.
        """
        if len(trades) == 0:
            print("No trades to visualize")
            return

        # Extract trade data
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Trades over time
        ax = axes[0, 0]
        if buy_trades:
            buy_dates = [t['date'] for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            ax.scatter(buy_dates, buy_prices, color='green', marker='^',
                      s=100, label='Buy', alpha=0.7)

        if sell_trades:
            sell_dates = [t['date'] for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            ax.scatter(sell_dates, sell_prices, color='red', marker='v',
                      s=100, label='Sell', alpha=0.7)

        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.set_title('Trade Entry/Exit Points', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Calculate profits
        profits = []
        position = 0
        buy_price = 0

        for trade in trades:
            if trade['action'] == 'BUY':
                position = trade['shares']
                buy_price = trade['price']
            elif trade['action'] == 'SELL' and position > 0:
                profit = (trade['price'] - buy_price) * position
                profits.append(profit)
                position = 0

        if profits:
            # Profit distribution
            ax = axes[0, 1]
            ax.hist(profits, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Profit ($)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Profit Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Cumulative profit
            ax = axes[1, 0]
            cumulative_profits = np.cumsum(profits)
            ax.plot(cumulative_profits, linewidth=2, color='#2E86AB')
            ax.axhline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Trade Number', fontsize=11)
            ax.set_ylabel('Cumulative Profit ($)', fontsize=11)
            ax.set_title('Cumulative Profit', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Win/Loss analysis
            ax = axes[1, 1]
            wins = sum(1 for p in profits if p > 0)
            losses = sum(1 for p in profits if p < 0)
            ax.bar(['Wins', 'Losses'], [wins, losses],
                  color=['green', 'red'], alpha=0.7)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title(f'Win Rate: {wins/(wins+losses)*100:.1f}%',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_metrics_comparison(
        results: Dict[str, Dict],
        title: str = "Strategy Comparison",
        save_path: Optional[str] = None
    ):
        """
        Compare multiple strategies side by side

        Args:
            results: Dictionary mapping strategy_name -> metrics_dict
        """
        metrics_to_plot = [
            'total_return', 'sharpe_ratio', 'max_drawdown',
            'win_rate', 'profit_factor'
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        strategy_names = list(results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_names)))

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            values = [results[strategy].get(metric, 0) for strategy in strategy_names]

            bars = ax.bar(strategy_names, values, color=colors, alpha=0.7, edgecolor='black')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)

            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Remove empty subplot
        fig.delaxes(axes[-1])

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()


class ModelInsightsVisualizer:
    """
    Visualize model internals and predictions
    """

    @staticmethod
    def plot_prediction_confidence(
        predictions: List[int],
        confidences: List[float],
        labels: List[int],
        title: str = "Prediction Confidence Analysis",
        save_path: Optional[str] = None
    ):
        """
        Visualize prediction confidence and accuracy
        """
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        labels = np.array(labels)

        correct = predictions == labels

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confidence distribution by correctness
        ax = axes[0]
        ax.hist(confidences[correct], bins=30, alpha=0.7, label='Correct',
               color='green', edgecolor='black')
        ax.hist(confidences[~correct], bins=30, alpha=0.7, label='Incorrect',
               color='red', edgecolor='black')
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy by confidence bins
        ax = axes[1]
        bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_centers = []

        for i in range(len(bins)-1):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() > 0:
                acc = correct[mask].mean()
                bin_accuracies.append(acc)
                bin_centers.append((bins[i] + bins[i+1]) / 2)

        ax.plot(bin_centers, bin_accuracies, marker='o', linewidth=2,
               markersize=8, color='#2E86AB')
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration', alpha=0.7)
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Calibration Plot', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: List[str] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix heatmap
        """
        if class_names is None:
            class_names = ['Buy', 'Sell', 'Hold']

        plt.figure(figsize=(10, 8))

        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})

        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()


def create_interactive_dashboard(
    portfolio_values: np.ndarray,
    dates: List,
    trades: List[Dict],
    daily_returns: np.ndarray,
    price_data: pd.DataFrame
):
    """
    Create interactive dashboard with plotly

    Requires: pip install plotly
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Portfolio Value', 'Price & Trades',
            'Daily Returns', 'Cumulative Returns',
            'Drawdown', 'Volume'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # Portfolio value
    fig.add_trace(
        go.Scatter(x=dates, y=portfolio_values, name='Portfolio',
                  line=dict(color='#2E86AB', width=2)),
        row=1, col=1
    )

    # Price with trades
    fig.add_trace(
        go.Scatter(x=price_data.index, y=price_data['close'],
                  name='Price', line=dict(color='gray', width=1)),
        row=1, col=2
    )

    # Add buy/sell markers
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']

    if buy_trades:
        fig.add_trace(
            go.Scatter(x=[t['date'] for t in buy_trades],
                      y=[t['price'] for t in buy_trades],
                      mode='markers', name='Buy',
                      marker=dict(symbol='triangle-up', size=10, color='green')),
            row=1, col=2
        )

    if sell_trades:
        fig.add_trace(
            go.Scatter(x=[t['date'] for t in sell_trades],
                      y=[t['price'] for t in sell_trades],
                      mode='markers', name='Sell',
                      marker=dict(symbol='triangle-down', size=10, color='red')),
            row=1, col=2
        )

    # Daily returns
    fig.add_trace(
        go.Bar(x=dates[1:], y=daily_returns*100, name='Daily Returns',
              marker=dict(color=['red' if r < 0 else 'green' for r in daily_returns])),
        row=2, col=1
    )

    # Cumulative returns
    cumulative_returns = np.cumprod(1 + daily_returns) - 1
    fig.add_trace(
        go.Scatter(x=dates[1:], y=cumulative_returns*100,
                  name='Cumulative Returns',
                  line=dict(color='#2E86AB', width=2)),
        row=2, col=2
    )

    # Drawdown
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax * 100
    fig.add_trace(
        go.Scatter(x=dates, y=drawdown, name='Drawdown',
                  fill='tozeroy', line=dict(color='red', width=1)),
        row=3, col=1
    )

    # Volume
    fig.add_trace(
        go.Bar(x=price_data.index, y=price_data['volume'],
              name='Volume', marker=dict(color='lightblue')),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text="Trading Strategy Dashboard",
        title_font_size=20
    )

    fig.show()


if __name__ == "__main__":
    print("Testing visualization tools...")

    # Generate sample data
    np.random.seed(42)
    portfolio_values = 100000 * np.cumprod(1 + np.random.randn(252) * 0.01)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Test portfolio plot
    PerformanceVisualizer.plot_portfolio_value(portfolio_values)

    # Test drawdown plot
    PerformanceVisualizer.plot_drawdown(portfolio_values)

    # Test returns distribution
    PerformanceVisualizer.plot_returns_distribution(daily_returns)

    print("\n✓ Visualization tools working!")

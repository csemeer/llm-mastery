"""
Market Data Processor

Fetches and processes stock market data from various sources
- Yahoo Finance for historical OHLCV
- Technical indicators calculation
- Real-time data support
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings

try:
    import yfinance as yf
except ImportError:
    yf = None
    warnings.warn("yfinance not installed. Install with: pip install yfinance")

try:
    import ta  # Technical analysis library
except ImportError:
    ta = None
    warnings.warn("ta not installed. Install with: pip install ta")


class MarketDataFetcher:
    """
    Fetch market data from various sources
    """
    def __init__(self, cache_dir: str = './data/market_cache'):
        self.cache_dir = cache_dir

    def fetch_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch stock OHLCV data

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        if yf is None:
            raise ImportError("yfinance required. Install with: pip install yfinance")

        print(f"Fetching {ticker} from {start_date} to {end_date}...")

        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for {ticker}")

        # Standardize column names
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        df = df[['open', 'high', 'low', 'close', 'volume']]

        print(f"Fetched {len(df)} records")

        return df

    def fetch_multiple_stocks(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks

        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        data = {}

        for ticker in tickers:
            try:
                df = self.fetch_stock_data(ticker, start_date, end_date, interval)
                data[ticker] = df
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")

        return data


class TechnicalIndicators:
    """
    Calculate technical indicators from OHLCV data
    """
    @staticmethod
    def add_indicators(df: pd.DataFrame, indicators: List[str] = None) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV DataFrame

        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicators to add (None = all)

        Returns:
            DataFrame with additional indicator columns
        """
        if ta is None:
            warnings.warn("ta library not available. Using simple indicators only.")
            return TechnicalIndicators._add_simple_indicators(df)

        df = df.copy()

        if indicators is None:
            indicators = [
                'sma', 'ema', 'rsi', 'macd', 'bbands', 'atr',
                'obv', 'adx', 'stoch', 'cci'
            ]

        # Simple Moving Averages
        if 'sma' in indicators:
            df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
            df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)

        # Exponential Moving Averages
        if 'ema' in indicators:
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)

        # RSI (Relative Strength Index)
        if 'rsi' in indicators:
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        # MACD
        if 'macd' in indicators:
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

        # Bollinger Bands
        if 'bbands' in indicators:
            bbands = ta.volatility.BollingerBands(df['close'])
            df['bb_high'] = bbands.bollinger_hband()
            df['bb_mid'] = bbands.bollinger_mavg()
            df['bb_low'] = bbands.bollinger_lband()

        # Average True Range (volatility)
        if 'atr' in indicators:
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])

        # On-Balance Volume
        if 'obv' in indicators:
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

        # ADX (trend strength)
        if 'adx' in indicators:
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])

        # Stochastic Oscillator
        if 'stoch' in indicators:
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

        # Commodity Channel Index
        if 'cci' in indicators:
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])

        # Fill NaN values with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    @staticmethod
    def _add_simple_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add simple indicators without ta library
        """
        df = df.copy()

        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()

        # Returns
        df['returns'] = df['close'].pct_change()

        # Volatility (rolling std of returns)
        df['volatility'] = df['returns'].rolling(window=20).std()

        # Volume change
        df['volume_change'] = df['volume'].pct_change()

        df = df.fillna(method='ffill').fillna(method='bfill')

        return df


class MarketDataset(Dataset):
    """
    PyTorch Dataset for market data with multi-modal support
    """
    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int = 60,
        pred_horizon: int = 1,
        include_indicators: bool = True,
        indicators: List[str] = None,
        task: str = 'classification'  # 'classification' or 'regression'
    ):
        """
        Args:
            data: DataFrame with OHLCV and optional indicators
            seq_len: Length of input sequence
            pred_horizon: How many steps ahead to predict
            include_indicators: Whether to include technical indicators
            indicators: List of specific indicators to include
            task: 'classification' (up/down) or 'regression' (predict return)
        """
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.task = task

        # Add indicators if requested
        if include_indicators and 'sma_5' not in data.columns:
            data = TechnicalIndicators.add_indicators(data, indicators)

        self.data = data

        # Separate OHLCV and indicators
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        self.ohlcv_data = data[ohlcv_cols].values

        if include_indicators:
            indicator_cols = [col for col in data.columns if col not in ohlcv_cols]
            if indicator_cols:
                self.indicator_data = data[indicator_cols].values
            else:
                self.indicator_data = None
        else:
            self.indicator_data = None

        # Normalize data
        self.ohlcv_mean = np.mean(self.ohlcv_data, axis=0, keepdims=True)
        self.ohlcv_std = np.std(self.ohlcv_data, axis=0, keepdims=True) + 1e-8

        if self.indicator_data is not None:
            self.indicator_mean = np.mean(self.indicator_data, axis=0, keepdims=True)
            self.indicator_std = np.std(self.indicator_data, axis=0, keepdims=True) + 1e-8

        # Create labels
        self.labels = self._create_labels()

    def _create_labels(self):
        """
        Create labels for prediction

        For classification: 0 = down, 1 = neutral, 2 = up
        For regression: percentage return
        """
        close_prices = self.data['close'].values

        if self.task == 'classification':
            # Calculate future returns
            returns = np.zeros(len(close_prices))
            returns[:-self.pred_horizon] = (
                close_prices[self.pred_horizon:] - close_prices[:-self.pred_horizon]
            ) / close_prices[:-self.pred_horizon]

            # Classify: up (>0.5%), down (<-0.5%), neutral
            labels = np.ones(len(returns), dtype=np.long)  # Default: neutral
            labels[returns > 0.005] = 2  # Up
            labels[returns < -0.005] = 0  # Down

        else:  # regression
            returns = np.zeros(len(close_prices))
            returns[:-self.pred_horizon] = (
                close_prices[self.pred_horizon:] - close_prices[:-self.pred_horizon]
            ) / close_prices[:-self.pred_horizon]
            labels = returns

        return labels

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_horizon + 1

    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with:
            - ohlcv: Normalized OHLCV sequence
            - indicators: Normalized indicator sequence (if available)
            - label: Target label
        """
        # Get sequence
        ohlcv_seq = self.ohlcv_data[idx:idx + self.seq_len]
        ohlcv_seq = (ohlcv_seq - self.ohlcv_mean) / self.ohlcv_std

        sample = {
            'ohlcv': torch.tensor(ohlcv_seq, dtype=torch.float32),
            'label': torch.tensor(self.labels[idx + self.seq_len - 1], dtype=torch.long if self.task == 'classification' else torch.float32)
        }

        if self.indicator_data is not None:
            indicator_seq = self.indicator_data[idx:idx + self.seq_len]
            indicator_seq = (indicator_seq - self.indicator_mean) / self.indicator_std
            sample['indicators'] = torch.tensor(indicator_seq, dtype=torch.float32)

        return sample


def create_market_datasets(
    ticker: str,
    start_date: str,
    end_date: str,
    seq_len: int = 60,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    **kwargs
) -> Tuple[MarketDataset, MarketDataset, MarketDataset]:
    """
    Convenience function to create train/val/test datasets

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date
        seq_len: Sequence length
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        **kwargs: Additional arguments for MarketDataset

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Fetch data
    fetcher = MarketDataFetcher()
    data = fetcher.fetch_stock_data(ticker, start_date, end_date)

    # Split
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    # Create datasets
    train_dataset = MarketDataset(train_data, seq_len=seq_len, **kwargs)
    val_dataset = MarketDataset(val_data, seq_len=seq_len, **kwargs)
    test_dataset = MarketDataset(test_data, seq_len=seq_len, **kwargs)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    print("Testing Market Data Processor...\n")

    # Note: This requires yfinance and internet connection
    try:
        # Fetch sample data
        fetcher = MarketDataFetcher()

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        print(f"Fetching AAPL data from {start_date} to {end_date}...")
        data = fetcher.fetch_stock_data('AAPL', start_date, end_date)

        print(f"✓ Fetched {len(data)} records")
        print(f"✓ Columns: {list(data.columns)}")

        # Add indicators
        print("\nAdding technical indicators...")
        data_with_indicators = TechnicalIndicators.add_indicators(data)
        print(f"✓ Now have {len(data_with_indicators.columns)} columns")

        # Create dataset
        print("\nCreating dataset...")
        dataset = MarketDataset(data_with_indicators, seq_len=60, task='classification')
        print(f"✓ Dataset size: {len(dataset)}")

        # Get sample
        sample = dataset[0]
        print(f"✓ Sample OHLCV shape: {sample['ohlcv'].shape}")
        if 'indicators' in sample:
            print(f"✓ Sample indicators shape: {sample['indicators'].shape}")
        print(f"✓ Sample label: {sample['label']}")

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"Test requires internet and yfinance: {e}")
        print("Install with: pip install yfinance ta")

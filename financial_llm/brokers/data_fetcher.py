"""
Unified Data Fetcher

Provides a unified interface for fetching market data from multiple brokers.
Automatically handles failover and data aggregation.
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd
from .base import BaseBroker, Quote
from .factory import BrokerFactory


class UnifiedDataFetcher:
    """
    Unified market data fetcher.

    Fetches data from multiple broker sources with automatic failover.
    Useful for redundancy and data quality.
    """

    def __init__(self, brokers: List[BaseBroker], primary_broker: Optional[str] = None):
        """
        Initialize unified data fetcher.

        Args:
            brokers: List of connected broker instances
            primary_broker: Name of primary broker (uses first if not specified)
        """
        self.brokers = brokers
        self.primary_broker_index = 0

        if primary_broker:
            for i, broker in enumerate(brokers):
                if broker.__class__.__name__.lower().startswith(primary_broker.lower()):
                    self.primary_broker_index = i
                    break

    @property
    def primary_broker(self) -> BaseBroker:
        """Get primary broker"""
        return self.brokers[self.primary_broker_index]

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback: Optional[int] = None,
        use_fallback: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data with automatic fallback.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            start_date: Start date
            end_date: End date
            lookback: Number of bars to look back
            use_fallback: Whether to use fallback brokers on failure

        Returns:
            DataFrame with OHLCV data
        """
        # Try primary broker first
        try:
            data = self.primary_broker.get_historical_data(
                symbol, timeframe, start_date, end_date, lookback
            )
            if not data.empty:
                return data
        except Exception as e:
            print(f"⚠ Primary broker failed: {str(e)}")

        # Try fallback brokers
        if use_fallback:
            for i, broker in enumerate(self.brokers):
                if i == self.primary_broker_index:
                    continue

                try:
                    data = broker.get_historical_data(
                        symbol, timeframe, start_date, end_date, lookback
                    )
                    if not data.empty:
                        print(f"✓ Using fallback broker: {broker.__class__.__name__}")
                        return data
                except Exception as e:
                    print(f"⚠ Fallback broker {broker.__class__.__name__} failed: {str(e)}")
                    continue

        # All brokers failed
        raise RuntimeError(f"Failed to fetch data for {symbol} from all brokers")

    def get_quote(
        self,
        symbol: str,
        use_fallback: bool = True
    ) -> Quote:
        """
        Get real-time quote with automatic fallback.

        Args:
            symbol: Trading symbol
            use_fallback: Whether to use fallback brokers on failure

        Returns:
            Quote object
        """
        # Try primary broker first
        try:
            return self.primary_broker.get_quote(symbol)
        except Exception as e:
            print(f"⚠ Primary broker failed: {str(e)}")

        # Try fallback brokers
        if use_fallback:
            for i, broker in enumerate(self.brokers):
                if i == self.primary_broker_index:
                    continue

                try:
                    quote = broker.get_quote(symbol)
                    print(f"✓ Using fallback broker: {broker.__class__.__name__}")
                    return quote
                except Exception as e:
                    continue

        raise RuntimeError(f"Failed to get quote for {symbol} from all brokers")

    def get_quotes(
        self,
        symbols: List[str],
        use_fallback: bool = True
    ) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of trading symbols
            use_fallback: Whether to use fallback brokers on failure

        Returns:
            Dictionary mapping symbol to Quote
        """
        quotes = {}

        for symbol in symbols:
            try:
                quotes[symbol] = self.get_quote(symbol, use_fallback)
            except Exception as e:
                print(f"⚠ Failed to get quote for {symbol}: {str(e)}")
                continue

        return quotes

    def aggregate_data_from_multiple_sources(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from all brokers and return aggregated results.

        Useful for data quality comparison and validation.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            start_date: Start date
            end_date: End date
            lookback: Number of bars

        Returns:
            Dictionary mapping broker name to DataFrame
        """
        results = {}

        for broker in self.brokers:
            try:
                data = broker.get_historical_data(
                    symbol, timeframe, start_date, end_date, lookback
                )
                if not data.empty:
                    results[broker.__class__.__name__] = data
            except Exception as e:
                print(f"⚠ {broker.__class__.__name__} failed: {str(e)}")
                continue

        return results

    def set_primary_broker(self, broker_index: int):
        """Set primary broker by index"""
        if 0 <= broker_index < len(self.brokers):
            self.primary_broker_index = broker_index
        else:
            raise ValueError(f"Invalid broker index: {broker_index}")

    def add_broker(self, broker: BaseBroker):
        """Add a broker to the pool"""
        self.brokers.append(broker)

    def remove_broker(self, broker_index: int):
        """Remove a broker from the pool"""
        if broker_index == self.primary_broker_index:
            raise ValueError("Cannot remove primary broker")

        if 0 <= broker_index < len(self.brokers):
            self.brokers.pop(broker_index)
        else:
            raise ValueError(f"Invalid broker index: {broker_index}")

    def get_broker_status(self) -> List[Dict[str, Any]]:
        """Get status of all brokers"""
        status = []
        for i, broker in enumerate(self.brokers):
            status.append({
                'index': i,
                'name': broker.__class__.__name__,
                'is_primary': i == self.primary_broker_index,
                'is_connected': broker.is_connected,
                'is_paper_trading': broker.is_paper_trading,
                'capabilities': [cap.value for cap in broker.get_capabilities()]
            })
        return status

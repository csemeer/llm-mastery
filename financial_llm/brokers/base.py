"""
Base Broker Interface

Defines the abstract base class that all broker implementations must inherit from.
Provides standardized methods for market data and order execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd


class BrokerCapability(Enum):
    """Broker capabilities enumeration"""
    MARKET_DATA = "market_data"
    HISTORICAL_DATA = "historical_data"
    REAL_TIME_QUOTES = "real_time_quotes"
    REAL_TIME_BARS = "real_time_bars"
    ORDER_EXECUTION = "order_execution"
    PAPER_TRADING = "paper_trading"
    OPTIONS_TRADING = "options_trading"
    FUTURES_TRADING = "futures_trading"
    CRYPTO_TRADING = "crypto_trading"
    MARGIN_TRADING = "margin_trading"
    BRACKET_ORDERS = "bracket_orders"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Position:
    """Represents a position held at the broker"""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    side: str  # 'long' or 'short'
    exchange: Optional[str] = None

    @property
    def is_long(self) -> bool:
        return self.side.lower() == 'long'

    @property
    def is_short(self) -> bool:
        return self.side.lower() == 'short'


@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    symbol: str
    quantity: float
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    commission: float = 0.0
    exchange: Optional[str] = None
    client_order_id: Optional[str] = None

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_pending(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]


@dataclass
class AccountInfo:
    """Broker account information"""
    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    equity: float
    currency: str = "USD"
    margin_used: float = 0.0
    maintenance_margin: float = 0.0
    last_updated: Optional[datetime] = None

    @property
    def margin_available(self) -> float:
        return self.buying_power - self.margin_used


@dataclass
class Quote:
    """Real-time quote data"""
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_price: float
    last_size: int
    volume: int
    timestamp: datetime
    exchange: Optional[str] = None

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price


class BaseBroker(ABC):
    """
    Abstract base class for all broker implementations.

    All broker integrations must implement this interface to ensure
    compatibility with the trading system.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize broker with configuration.

        Args:
            config: Dictionary containing broker-specific configuration
        """
        self.config = config
        self.is_connected = False
        self.is_paper_trading = config.get('paper_trading', True)

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to broker.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close connection to broker.

        Returns:
            True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[BrokerCapability]:
        """
        Get list of supported broker capabilities.

        Returns:
            List of BrokerCapability enums
        """
        pass

    # Market Data Methods

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'RELIANCE')
            timeframe: Bar timeframe ('1Min', '5Min', '1Hour', '1Day')
            start_date: Start date for data
            end_date: End date for data
            lookback: Number of bars to look back (alternative to start_date)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Quote object with current bid/ask/last prices
        """
        pass

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get real-time quotes for multiple symbols.

        Args:
            symbols: List of trading symbols

        Returns:
            Dictionary mapping symbol to Quote object
        """
        return {symbol: self.get_quote(symbol) for symbol in symbols}

    # Account Methods

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            AccountInfo object with account details
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get all current positions.

        Returns:
            List of Position objects
        """
        pass

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position object if exists, None otherwise
        """
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    # Order Methods

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        quantity: float,
        side: OrderSide,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        **kwargs
    ) -> Order:
        """
        Place a trading order.

        Args:
            symbol: Trading symbol
            quantity: Number of shares/contracts
            side: OrderSide.BUY or OrderSide.SELL
            order_type: Type of order (MARKET, LIMIT, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force ('day', 'gtc', 'ioc', 'fok')
            **kwargs: Broker-specific parameters

        Returns:
            Order object with order details
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful, False otherwise
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Order:
        """
        Get order details.

        Args:
            order_id: Order ID

        Returns:
            Order object
        """
        pass

    @abstractmethod
    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        limit: int = 100
    ) -> List[Order]:
        """
        Get list of orders.

        Args:
            status: Filter by order status (None for all)
            limit: Maximum number of orders to return

        Returns:
            List of Order objects
        """
        pass

    # Helper Methods

    def validate_connection(self) -> bool:
        """
        Validate broker connection is active.

        Returns:
            True if connected and validated, False otherwise
        """
        if not self.is_connected:
            return False

        try:
            # Try to get account info as a connectivity test
            self.get_account()
            return True
        except Exception:
            self.is_connected = False
            return False

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
        return False

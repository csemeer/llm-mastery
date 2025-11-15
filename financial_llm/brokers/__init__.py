"""
Broker Integration Module for US and Indian Markets

This module provides unified interfaces for connecting to major US and Indian brokers
for market data streaming and live trading execution.

Supported US Brokers:
    - Alpaca
    - Interactive Brokers (IBKR)
    - TD Ameritrade

Supported Indian Brokers:
    - Zerodha Kite
    - Angel One
    - Upstox
    - ICICI Direct

Usage:
    from financial_llm.brokers import BrokerFactory, BrokerConfig

    # Load broker configuration
    config = BrokerConfig.from_yaml('configs/broker_config.yaml')

    # Create broker instance
    broker = BrokerFactory.create('alpaca', config.get_broker_config('alpaca'))

    # Fetch market data
    data = broker.get_historical_data('AAPL', '1Day', lookback=60)

    # Place order
    order = broker.place_order(
        symbol='AAPL',
        quantity=10,
        side='buy',
        order_type='market'
    )
"""

from .base import BaseBroker, BrokerCapability, OrderSide, OrderType, OrderStatus
from .factory import BrokerFactory
from .config import BrokerConfig, BrokerCredentials
from .data_fetcher import UnifiedDataFetcher
from .order_manager import OrderManager

__all__ = [
    'BaseBroker',
    'BrokerCapability',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'BrokerFactory',
    'BrokerConfig',
    'BrokerCredentials',
    'UnifiedDataFetcher',
    'OrderManager',
]

__version__ = '1.0.0'

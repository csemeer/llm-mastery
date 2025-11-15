"""
Broker Factory

Creates broker instances based on configuration.
"""

from typing import Dict, Any, Optional
from .base import BaseBroker
from .us.alpaca_broker import AlpacaBroker
from .us.ibkr_broker import InteractiveBrokersBroker
from .us.td_ameritrade_broker import TDAmeritradeBroker
from .india.zerodha_broker import ZerodhaBroker
from .india.angel_one_broker import AngelOneBroker
from .india.upstox_broker import UpstoxBroker
from .india.icici_direct_broker import ICICIDirectBroker


class BrokerFactory:
    """
    Factory for creating broker instances.

    Automatically instantiates the correct broker class based on type.
    """

    # Broker type mapping
    BROKER_CLASSES = {
        # US Brokers
        'alpaca': AlpacaBroker,
        'ibkr': InteractiveBrokersBroker,
        'interactive_brokers': InteractiveBrokersBroker,
        'td_ameritrade': TDAmeritradeBroker,
        'tda': TDAmeritradeBroker,

        # Indian Brokers
        'zerodha': ZerodhaBroker,
        'kite': ZerodhaBroker,
        'angel_one': AngelOneBroker,
        'angel': AngelOneBroker,
        'upstox': UpstoxBroker,
        'icici_direct': ICICIDirectBroker,
        'icici': ICICIDirectBroker,
    }

    @classmethod
    def create(
        cls,
        broker_type: str,
        config: Dict[str, Any],
        auto_connect: bool = True
    ) -> BaseBroker:
        """
        Create a broker instance.

        Args:
            broker_type: Type of broker (alpaca, zerodha, etc.)
            config: Broker configuration dictionary
            auto_connect: Whether to automatically connect

        Returns:
            BaseBroker instance

        Raises:
            ValueError: If broker type is not supported
        """
        broker_type = broker_type.lower()

        if broker_type not in cls.BROKER_CLASSES:
            supported = ', '.join(cls.BROKER_CLASSES.keys())
            raise ValueError(
                f"Unsupported broker type: {broker_type}. "
                f"Supported brokers: {supported}"
            )

        # Create broker instance
        broker_class = cls.BROKER_CLASSES[broker_type]
        broker = broker_class(config)

        # Auto-connect if requested
        if auto_connect:
            success = broker.connect()
            if not success:
                print(f"⚠ Warning: Failed to connect to {broker_type}")

        return broker

    @classmethod
    def create_from_config(
        cls,
        broker_name: str,
        broker_config_manager,
        auto_connect: bool = True
    ) -> BaseBroker:
        """
        Create broker from BrokerConfig manager.

        Args:
            broker_name: Name of the broker in configuration
            broker_config_manager: BrokerConfig instance
            auto_connect: Whether to automatically connect

        Returns:
            BaseBroker instance
        """
        config = broker_config_manager.get_broker_config(broker_name)
        broker_type = config.get('type')

        if not broker_type:
            raise ValueError(f"Broker '{broker_name}' missing 'type' in configuration")

        return cls.create(broker_type, config, auto_connect)

    @classmethod
    def get_supported_brokers(cls) -> Dict[str, str]:
        """
        Get list of supported brokers.

        Returns:
            Dictionary mapping broker type to class name
        """
        return {
            broker_type: broker_class.__name__
            for broker_type, broker_class in cls.BROKER_CLASSES.items()
        }

    @classmethod
    def is_broker_supported(cls, broker_type: str) -> bool:
        """Check if a broker type is supported"""
        return broker_type.lower() in cls.BROKER_CLASSES

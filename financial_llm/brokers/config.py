"""
Broker Configuration Management

Handles broker credentials, configuration loading/saving, and encryption
for secure storage of API keys and secrets.
"""

import os
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any, List
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64
import getpass


@dataclass
class BrokerCredentials:
    """Broker API credentials"""
    broker_name: str
    api_key: str
    api_secret: str
    account_id: Optional[str] = None
    api_endpoint: Optional[str] = None
    paper_trading: bool = True
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BrokerCredentials':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class MarketConfig:
    """Market-specific configuration"""
    market: str  # 'US' or 'IN'
    timezone: str
    trading_hours_start: str  # HH:MM format
    trading_hours_end: str
    currency: str
    default_exchange: Optional[str] = None

    @classmethod
    def us_market(cls) -> 'MarketConfig':
        """Default US market configuration"""
        return cls(
            market='US',
            timezone='America/New_York',
            trading_hours_start='09:30',
            trading_hours_end='16:00',
            currency='USD',
            default_exchange='NASDAQ'
        )

    @classmethod
    def indian_market(cls) -> 'MarketConfig':
        """Default Indian market configuration"""
        return cls(
            market='IN',
            timezone='Asia/Kolkata',
            trading_hours_start='09:15',
            trading_hours_end='15:30',
            currency='INR',
            default_exchange='NSE'
        )


class CredentialManager:
    """
    Secure credential management with encryption.

    Credentials are encrypted at rest using Fernet symmetric encryption
    with a key derived from a user password.
    """

    def __init__(self, credentials_file: str = ".broker_credentials.enc"):
        """
        Initialize credential manager.

        Args:
            credentials_file: Path to encrypted credentials file
        """
        self.credentials_file = Path(credentials_file)
        self._fernet: Optional[Fernet] = None
        self._credentials: Dict[str, BrokerCredentials] = {}

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def _get_fernet(self, password: str, salt: bytes) -> Fernet:
        """Get Fernet cipher instance"""
        key = self._derive_key(password, salt)
        return Fernet(key)

    def save_credentials(
        self,
        broker_name: str,
        credentials: BrokerCredentials,
        password: Optional[str] = None
    ):
        """
        Save encrypted credentials for a broker.

        Args:
            broker_name: Name of the broker
            credentials: BrokerCredentials object
            password: Encryption password (will prompt if not provided)
        """
        self._credentials[broker_name] = credentials

        if password is None:
            password = getpass.getpass("Enter password to encrypt credentials: ")

        # Generate random salt
        salt = os.urandom(16)

        # Encrypt credentials
        fernet = self._get_fernet(password, salt)
        credentials_json = json.dumps(
            {k: v.to_dict() for k, v in self._credentials.items()}
        )
        encrypted_data = fernet.encrypt(credentials_json.encode())

        # Save salt + encrypted data
        with open(self.credentials_file, 'wb') as f:
            f.write(salt + encrypted_data)

        print(f"✓ Credentials for {broker_name} saved securely")

    def load_credentials(
        self,
        password: Optional[str] = None
    ) -> Dict[str, BrokerCredentials]:
        """
        Load and decrypt all credentials.

        Args:
            password: Decryption password (will prompt if not provided)

        Returns:
            Dictionary mapping broker name to BrokerCredentials
        """
        if not self.credentials_file.exists():
            return {}

        if password is None:
            password = getpass.getpass("Enter password to decrypt credentials: ")

        # Read salt + encrypted data
        with open(self.credentials_file, 'rb') as f:
            data = f.read()

        salt = data[:16]
        encrypted_data = data[16:]

        # Decrypt
        try:
            fernet = self._get_fernet(password, salt)
            decrypted_json = fernet.decrypt(encrypted_data).decode()
            credentials_dict = json.loads(decrypted_json)

            self._credentials = {
                k: BrokerCredentials.from_dict(v)
                for k, v in credentials_dict.items()
            }
            return self._credentials
        except Exception as e:
            raise ValueError(f"Failed to decrypt credentials: {str(e)}")

    def get_credentials(self, broker_name: str) -> Optional[BrokerCredentials]:
        """Get credentials for a specific broker"""
        return self._credentials.get(broker_name)

    def delete_credentials(self, broker_name: str):
        """Delete credentials for a broker"""
        if broker_name in self._credentials:
            del self._credentials[broker_name]
            print(f"✓ Credentials for {broker_name} deleted")

    def list_brokers(self) -> List[str]:
        """List all configured brokers"""
        return list(self._credentials.keys())


class BrokerConfig:
    """
    Broker configuration management.

    Handles loading/saving broker configurations from YAML files
    and integrates with CredentialManager for secure credential storage.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize broker configuration.

        Args:
            config_file: Path to broker configuration YAML file
        """
        self.config_file = config_file
        self.brokers: Dict[str, Dict[str, Any]] = {}
        self.markets: Dict[str, MarketConfig] = {
            'US': MarketConfig.us_market(),
            'IN': MarketConfig.indian_market()
        }
        self.credential_manager = CredentialManager()

        if config_file and Path(config_file).exists():
            self.load_from_yaml(config_file)

    def load_from_yaml(self, config_file: str):
        """
        Load broker configuration from YAML file.

        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = config_file

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Load broker configurations
        self.brokers = config.get('brokers', {})

        # Load market configurations
        if 'markets' in config:
            for market_name, market_data in config['markets'].items():
                self.markets[market_name] = MarketConfig(**market_data)

        print(f"✓ Loaded configuration for {len(self.brokers)} brokers")

    def save_to_yaml(self, config_file: Optional[str] = None):
        """
        Save broker configuration to YAML file.

        Args:
            config_file: Path to save configuration (uses self.config_file if None)
        """
        config_file = config_file or self.config_file
        if not config_file:
            raise ValueError("No configuration file specified")

        config = {
            'brokers': self.brokers,
            'markets': {
                name: asdict(market)
                for name, market in self.markets.items()
            }
        }

        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Configuration saved to {config_file}")

    def get_broker_config(
        self,
        broker_name: str,
        include_credentials: bool = True
    ) -> Dict[str, Any]:
        """
        Get complete configuration for a broker.

        Args:
            broker_name: Name of the broker
            include_credentials: Whether to include credentials

        Returns:
            Dictionary with broker configuration
        """
        if broker_name not in self.brokers:
            raise ValueError(f"Broker '{broker_name}' not configured")

        config = self.brokers[broker_name].copy()

        if include_credentials:
            credentials = self.credential_manager.get_credentials(broker_name)
            if credentials:
                config.update(credentials.to_dict())

        return config

    def add_broker(
        self,
        broker_name: str,
        broker_type: str,
        market: str,
        enabled: bool = True,
        **kwargs
    ):
        """
        Add a broker configuration.

        Args:
            broker_name: Unique name for this broker instance
            broker_type: Type of broker (alpaca, zerodha, etc.)
            market: Market (US or IN)
            enabled: Whether broker is enabled
            **kwargs: Additional broker-specific configuration
        """
        self.brokers[broker_name] = {
            'type': broker_type,
            'market': market,
            'enabled': enabled,
            **kwargs
        }

    def get_enabled_brokers(self, market: Optional[str] = None) -> List[str]:
        """
        Get list of enabled brokers.

        Args:
            market: Filter by market (US or IN), None for all

        Returns:
            List of enabled broker names
        """
        enabled = []
        for name, config in self.brokers.items():
            if config.get('enabled', True):
                if market is None or config.get('market') == market:
                    enabled.append(name)
        return enabled

    def get_market_config(self, market: str) -> MarketConfig:
        """Get market configuration"""
        if market not in self.markets:
            raise ValueError(f"Market '{market}' not configured")
        return self.markets[market]

    @classmethod
    def create_default_config(cls, config_file: str = "configs/broker_config.yaml"):
        """
        Create a default broker configuration file with examples.

        Args:
            config_file: Path to create configuration file
        """
        default_config = {
            'markets': {
                'US': asdict(MarketConfig.us_market()),
                'IN': asdict(MarketConfig.indian_market())
            },
            'brokers': {
                # US Brokers
                'alpaca': {
                    'type': 'alpaca',
                    'market': 'US',
                    'enabled': True,
                    'paper_trading': True,
                    'base_url': 'https://paper-api.alpaca.markets',
                    'data_feed': 'iex',  # 'iex' or 'sip'
                },
                'interactive_brokers': {
                    'type': 'ibkr',
                    'market': 'US',
                    'enabled': False,
                    'paper_trading': True,
                    'host': '127.0.0.1',
                    'port': 7497,  # 7497 for paper, 7496 for live
                    'client_id': 1,
                },
                'td_ameritrade': {
                    'type': 'td_ameritrade',
                    'market': 'US',
                    'enabled': False,
                    'paper_trading': True,
                },

                # Indian Brokers
                'zerodha': {
                    'type': 'zerodha',
                    'market': 'IN',
                    'enabled': False,
                    'paper_trading': True,
                    'exchange': 'NSE',  # NSE or BSE
                },
                'angel_one': {
                    'type': 'angel_one',
                    'market': 'IN',
                    'enabled': False,
                    'paper_trading': True,
                    'exchange': 'NSE',
                },
                'upstox': {
                    'type': 'upstox',
                    'market': 'IN',
                    'enabled': False,
                    'paper_trading': True,
                    'exchange': 'NSE',
                },
                'icici_direct': {
                    'type': 'icici_direct',
                    'market': 'IN',
                    'enabled': False,
                    'paper_trading': True,
                    'exchange': 'NSE',
                },
            }
        }

        # Create directory if needed
        Path(config_file).parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Default broker configuration created at {config_file}")
        return cls(config_file)

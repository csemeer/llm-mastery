#!/usr/bin/env python3
"""
Broker Configuration CLI

Command-line tool for managing broker configurations and credentials.

Usage:
    python broker_cli.py setup              # Create default configuration
    python broker_cli.py add-credentials    # Add broker credentials
    python broker_cli.py test <broker>      # Test broker connection
    python broker_cli.py list               # List configured brokers
    python broker_cli.py quote <symbol>     # Get real-time quote
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from financial_llm.brokers import (
    BrokerConfig, BrokerFactory, BrokerCredentials,
    UnifiedDataFetcher, OrderManager
)


class BrokerCLI:
    """CLI for broker management"""

    def __init__(self):
        self.config_file = "configs/broker_config.yaml"
        self.broker_config = None

    def setup(self):
        """Create default broker configuration"""
        print("Creating default broker configuration...")
        BrokerConfig.create_default_config(self.config_file)
        print(f"\n✓ Configuration created at {self.config_file}")
        print("\nNext steps:")
        print("1. Edit the configuration file to enable desired brokers")
        print("2. Add credentials: python broker_cli.py add-credentials")
        print("3. Test connection: python broker_cli.py test <broker_name>")

    def add_credentials(self, broker_name: Optional[str] = None):
        """Add credentials for a broker"""
        # Load configuration
        if not Path(self.config_file).exists():
            print(f"✗ Configuration file not found: {self.config_file}")
            print("Run: python broker_cli.py setup")
            return

        self.broker_config = BrokerConfig(self.config_file)

        # Select broker
        if not broker_name:
            print("Available brokers:")
            for i, name in enumerate(self.broker_config.brokers.keys(), 1):
                broker_info = self.broker_config.brokers[name]
                print(f"  {i}. {name} ({broker_info['type']}) - {broker_info['market']}")

            choice = input("\nSelect broker number: ")
            broker_name = list(self.broker_config.brokers.keys())[int(choice) - 1]

        broker_info = self.broker_config.brokers.get(broker_name)
        if not broker_info:
            print(f"✗ Broker '{broker_name}' not found in configuration")
            return

        print(f"\n=== Adding credentials for {broker_name} ===")
        print(f"Type: {broker_info['type']}")
        print(f"Market: {broker_info['market']}")
        print()

        # Collect credentials
        api_key = input("API Key: ")
        api_secret = input("API Secret: ")
        account_id = input("Account ID (optional, press Enter to skip): ") or None

        # Additional fields based on broker type
        additional_params = {}
        broker_type = broker_info['type']

        if broker_type == 'alpaca':
            additional_params['base_url'] = broker_info.get('base_url')
            additional_params['data_feed'] = broker_info.get('data_feed', 'iex')

        elif broker_type == 'ibkr':
            additional_params['host'] = broker_info.get('host', '127.0.0.1')
            additional_params['port'] = broker_info.get('port', 7497)
            additional_params['client_id'] = broker_info.get('client_id', 1)

        elif broker_type == 'zerodha':
            access_token = input("Access Token (optional, press Enter to skip): ") or None
            if access_token:
                additional_params['access_token'] = access_token

        # Create credentials object
        credentials = BrokerCredentials(
            broker_name=broker_name,
            api_key=api_key,
            api_secret=api_secret,
            account_id=account_id,
            paper_trading=broker_info.get('paper_trading', True),
            additional_params=additional_params
        )

        # Save credentials
        self.broker_config.credential_manager.save_credentials(
            broker_name,
            credentials
        )

        print(f"\n✓ Credentials saved for {broker_name}")

    def test_connection(self, broker_name: str):
        """Test connection to a broker"""
        # Load configuration
        if not Path(self.config_file).exists():
            print(f"✗ Configuration file not found: {self.config_file}")
            return

        self.broker_config = BrokerConfig(self.config_file)

        # Load credentials
        print("Loading credentials...")
        self.broker_config.credential_manager.load_credentials()

        # Create broker instance
        print(f"\nConnecting to {broker_name}...")
        try:
            broker = BrokerFactory.create_from_config(
                broker_name,
                self.broker_config,
                auto_connect=True
            )

            if broker.is_connected:
                print(f"✓ Connected successfully!\n")

                # Test account info
                print("Fetching account information...")
                account = broker.get_account()
                print(f"\nAccount ID: {account.account_id}")
                print(f"Buying Power: ${account.buying_power:,.2f}")
                print(f"Cash: ${account.cash:,.2f}")
                print(f"Portfolio Value: ${account.portfolio_value:,.2f}")
                print(f"Currency: {account.currency}")

                # Test positions
                print("\nFetching positions...")
                positions = broker.get_positions()
                if positions:
                    print(f"\n{len(positions)} position(s):")
                    for pos in positions:
                        pnl_sign = '+' if pos.unrealized_pnl >= 0 else ''
                        print(f"  {pos.symbol}: {pos.quantity} @ ${pos.avg_entry_price:.2f} "
                              f"(P&L: {pnl_sign}${pos.unrealized_pnl:.2f})")
                else:
                    print("  No positions")

                # Test capabilities
                print("\nBroker capabilities:")
                for cap in broker.get_capabilities():
                    print(f"  ✓ {cap.value}")

                broker.disconnect()

            else:
                print("✗ Failed to connect")

        except Exception as e:
            print(f"✗ Error: {str(e)}")

    def list_brokers(self):
        """List all configured brokers"""
        if not Path(self.config_file).exists():
            print(f"✗ Configuration file not found: {self.config_file}")
            return

        self.broker_config = BrokerConfig(self.config_file)

        print("\n=== Configured Brokers ===\n")
        for name, info in self.broker_config.brokers.items():
            status = "✓ Enabled" if info.get('enabled', True) else "✗ Disabled"
            print(f"{name}")
            print(f"  Type: {info['type']}")
            print(f"  Market: {info['market']}")
            print(f"  Status: {status}")
            print(f"  Paper Trading: {info.get('paper_trading', True)}")
            print()

    def get_quote(self, broker_name: str, symbol: str):
        """Get real-time quote"""
        # Load configuration
        if not Path(self.config_file).exists():
            print(f"✗ Configuration file not found: {self.config_file}")
            return

        self.broker_config = BrokerConfig(self.config_file)
        self.broker_config.credential_manager.load_credentials()

        # Create broker
        print(f"Connecting to {broker_name}...")
        broker = BrokerFactory.create_from_config(
            broker_name,
            self.broker_config,
            auto_connect=True
        )

        if not broker.is_connected:
            print("✗ Failed to connect")
            return

        # Get quote
        print(f"Fetching quote for {symbol}...")
        try:
            quote = broker.get_quote(symbol)

            print(f"\n=== {symbol} Quote ===")
            print(f"Last Price: ${quote.last_price:.2f}")
            print(f"Bid: ${quote.bid_price:.2f} x {quote.bid_size}")
            print(f"Ask: ${quote.ask_price:.2f} x {quote.ask_size}")
            print(f"Spread: ${quote.spread:.2f}")
            print(f"Volume: {quote.volume:,}")
            print(f"Time: {quote.timestamp}")

        except Exception as e:
            print(f"✗ Error: {str(e)}")
        finally:
            broker.disconnect()

    def get_historical_data(self, broker_name: str, symbol: str, timeframe: str = '1Day', lookback: int = 30):
        """Get historical data"""
        # Load configuration
        if not Path(self.config_file).exists():
            print(f"✗ Configuration file not found: {self.config_file}")
            return

        self.broker_config = BrokerConfig(self.config_file)
        self.broker_config.credential_manager.load_credentials()

        # Create broker
        print(f"Connecting to {broker_name}...")
        broker = BrokerFactory.create_from_config(
            broker_name,
            self.broker_config,
            auto_connect=True
        )

        if not broker.is_connected:
            print("✗ Failed to connect")
            return

        # Get historical data
        print(f"Fetching {lookback} {timeframe} bars for {symbol}...")
        try:
            data = broker.get_historical_data(symbol, timeframe, lookback=lookback)

            if not data.empty:
                print(f"\n✓ Received {len(data)} bars")
                print(f"\nLatest 5 bars:")
                print(data.tail(5))

                print(f"\nSummary:")
                print(f"  High: ${data['high'].max():.2f}")
                print(f"  Low: ${data['low'].min():.2f}")
                print(f"  Avg Volume: {data['volume'].mean():,.0f}")
            else:
                print("✗ No data received")

        except Exception as e:
            print(f"✗ Error: {str(e)}")
        finally:
            broker.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description='Broker Configuration CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Setup command
    subparsers.add_parser('setup', help='Create default broker configuration')

    # Add credentials command
    add_creds_parser = subparsers.add_parser('add-credentials', help='Add broker credentials')
    add_creds_parser.add_argument('broker', nargs='?', help='Broker name (optional)')

    # Test connection command
    test_parser = subparsers.add_parser('test', help='Test broker connection')
    test_parser.add_argument('broker', help='Broker name')

    # List brokers command
    subparsers.add_parser('list', help='List configured brokers')

    # Get quote command
    quote_parser = subparsers.add_parser('quote', help='Get real-time quote')
    quote_parser.add_argument('broker', help='Broker name')
    quote_parser.add_argument('symbol', help='Trading symbol')

    # Get historical data command
    hist_parser = subparsers.add_parser('history', help='Get historical data')
    hist_parser.add_argument('broker', help='Broker name')
    hist_parser.add_argument('symbol', help='Trading symbol')
    hist_parser.add_argument('--timeframe', default='1Day', help='Timeframe (default: 1Day)')
    hist_parser.add_argument('--lookback', type=int, default=30, help='Lookback period (default: 30)')

    args = parser.parse_args()

    cli = BrokerCLI()

    if args.command == 'setup':
        cli.setup()
    elif args.command == 'add-credentials':
        cli.add_credentials(args.broker if hasattr(args, 'broker') else None)
    elif args.command == 'test':
        cli.test_connection(args.broker)
    elif args.command == 'list':
        cli.list_brokers()
    elif args.command == 'quote':
        cli.get_quote(args.broker, args.symbol)
    elif args.command == 'history':
        cli.get_historical_data(args.broker, args.symbol, args.timeframe, args.lookback)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

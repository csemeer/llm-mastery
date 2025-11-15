"""
Broker Integration Example

Demonstrates how to use the broker integration module for
US and Indian markets.

This example shows:
1. Setting up broker configuration
2. Connecting to multiple brokers
3. Fetching market data
4. Managing orders and positions
5. Using unified data fetcher and order manager
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from financial_llm.brokers import (
    BrokerConfig, BrokerFactory, BrokerCredentials,
    UnifiedDataFetcher, OrderManager,
    OrderSide, OrderType
)


def example_1_basic_connection():
    """Example 1: Basic broker connection"""
    print("=" * 60)
    print("Example 1: Basic Broker Connection")
    print("=" * 60)

    # Create broker configuration (programmatic)
    config = {
        'api_key': 'YOUR_API_KEY',
        'api_secret': 'YOUR_API_SECRET',
        'paper_trading': True,
        'base_url': 'https://paper-api.alpaca.markets',
        'data_feed': 'iex'
    }

    # Create broker instance
    broker = BrokerFactory.create('alpaca', config, auto_connect=False)

    # Connect manually
    if broker.connect():
        print("\n✓ Connected successfully!")

        # Get account info
        account = broker.get_account()
        print(f"\nAccount Info:")
        print(f"  Buying Power: ${account.buying_power:,.2f}")
        print(f"  Portfolio Value: ${account.portfolio_value:,.2f}")

        # Disconnect
        broker.disconnect()
    else:
        print("\n✗ Failed to connect")


def example_2_market_data():
    """Example 2: Fetching market data"""
    print("\n" + "=" * 60)
    print("Example 2: Fetching Market Data")
    print("=" * 60)

    # Load from configuration file
    broker_config = BrokerConfig('configs/broker_config.yaml')
    broker_config.credential_manager.load_credentials()

    # Create broker
    broker = BrokerFactory.create_from_config('alpaca', broker_config)

    if broker.is_connected:
        # Get historical data
        print("\nFetching historical data for AAPL...")
        data = broker.get_historical_data(
            symbol='AAPL',
            timeframe='1Day',
            lookback=60
        )

        print(f"✓ Received {len(data)} bars")
        print(f"\nLatest 5 bars:")
        print(data.tail())

        # Get real-time quote
        print("\nFetching real-time quote for AAPL...")
        quote = broker.get_quote('AAPL')

        print(f"✓ Last: ${quote.last_price:.2f}")
        print(f"  Bid: ${quote.bid_price:.2f} x {quote.bid_size}")
        print(f"  Ask: ${quote.ask_price:.2f} x {quote.ask_size}")

        broker.disconnect()


def example_3_unified_data_fetcher():
    """Example 3: Using unified data fetcher with multiple brokers"""
    print("\n" + "=" * 60)
    print("Example 3: Unified Data Fetcher")
    print("=" * 60)

    broker_config = BrokerConfig('configs/broker_config.yaml')
    broker_config.credential_manager.load_credentials()

    # Create multiple broker instances
    brokers = []

    # Add Alpaca
    try:
        alpaca = BrokerFactory.create_from_config('alpaca', broker_config)
        if alpaca.is_connected:
            brokers.append(alpaca)
            print("✓ Alpaca connected")
    except Exception as e:
        print(f"⚠ Alpaca failed: {str(e)}")

    # Add more brokers as needed...

    if brokers:
        # Create unified data fetcher
        data_fetcher = UnifiedDataFetcher(brokers, primary_broker='alpaca')

        # Fetch data with automatic fallback
        print("\nFetching data with automatic fallback...")
        data = data_fetcher.get_historical_data(
            symbol='AAPL',
            timeframe='1Day',
            lookback=30,
            use_fallback=True
        )

        print(f"✓ Received {len(data)} bars from {data_fetcher.primary_broker.__class__.__name__}")

        # Get quotes for multiple symbols
        print("\nFetching quotes for multiple symbols...")
        quotes = data_fetcher.get_quotes(['AAPL', 'MSFT', 'GOOGL'])

        for symbol, quote in quotes.items():
            print(f"  {symbol}: ${quote.last_price:.2f}")

        # Cleanup
        for broker in brokers:
            broker.disconnect()


def example_4_order_management():
    """Example 4: Order management"""
    print("\n" + "=" * 60)
    print("Example 4: Order Management")
    print("=" * 60)

    broker_config = BrokerConfig('configs/broker_config.yaml')
    broker_config.credential_manager.load_credentials()

    # Create brokers
    brokers = {}

    try:
        alpaca = BrokerFactory.create_from_config('alpaca', broker_config)
        if alpaca.is_connected:
            brokers['alpaca'] = alpaca
            print("✓ Alpaca connected")
    except Exception as e:
        print(f"⚠ Alpaca failed: {str(e)}")

    if brokers:
        # Create order manager
        order_manager = OrderManager(brokers, default_broker='alpaca')

        # Set risk controls
        order_manager.set_max_position_size('AAPL', 100)  # Max 100 shares
        order_manager.set_max_order_value(10000)  # Max $10,000 per order
        order_manager.set_daily_loss_limit(500)  # Max $500 daily loss

        print("\n✓ Risk controls enabled")

        # Get current positions
        print("\nCurrent positions:")
        positions = order_manager.get_positions()
        if positions:
            for symbol, pos in positions.items():
                print(f"  {symbol}: {pos.quantity} shares @ ${pos.avg_entry_price:.2f}")
        else:
            print("  No positions")

        # Place a limit order (example - not executed)
        print("\nOrder placement example (not executed):")
        print("  order_manager.place_order(")
        print("      symbol='AAPL',")
        print("      quantity=10,")
        print("      side=OrderSide.BUY,")
        print("      order_type=OrderType.LIMIT,")
        print("      limit_price=150.00")
        print("  )")

        # Uncomment to actually place order:
        # order_record = order_manager.place_order(
        #     symbol='AAPL',
        #     quantity=10,
        #     side=OrderSide.BUY,
        #     order_type=OrderType.LIMIT,
        #     limit_price=150.00
        # )
        # print(f"✓ Order placed: {order_record.order.order_id}")

        # Get order history
        print("\nRecent orders:")
        order_history = order_manager.get_order_history()
        if order_history:
            for record in order_history[:5]:
                print(f"  {record.order.symbol}: {record.order.side.value} "
                      f"{record.order.quantity} @ {record.order.status.value}")
        else:
            print("  No order history")

        # Cleanup
        for broker in brokers.values():
            broker.disconnect()


def example_5_zerodha_india():
    """Example 5: Trading with Zerodha (Indian market)"""
    print("\n" + "=" * 60)
    print("Example 5: Zerodha (Indian Market)")
    print("=" * 60)

    broker_config = BrokerConfig('configs/broker_config.yaml')
    broker_config.credential_manager.load_credentials()

    try:
        # Create Zerodha broker
        zerodha = BrokerFactory.create_from_config('zerodha', broker_config)

        if zerodha.is_connected:
            print("✓ Connected to Zerodha")

            # Get Indian market data
            print("\nFetching historical data for RELIANCE (NSE)...")
            data = zerodha.get_historical_data(
                symbol='RELIANCE',
                timeframe='1Day',
                lookback=30
            )

            print(f"✓ Received {len(data)} bars")
            print(f"\nLatest prices:")
            print(data.tail())

            # Get quote
            print("\nFetching real-time quote for RELIANCE...")
            quote = zerodha.get_quote('RELIANCE')
            print(f"✓ Last: ₹{quote.last_price:.2f}")

            # Get account info
            account = zerodha.get_account()
            print(f"\nAccount Info:")
            print(f"  Buying Power: ₹{account.buying_power:,.2f}")
            print(f"  Portfolio Value: ₹{account.portfolio_value:,.2f}")

            zerodha.disconnect()

    except Exception as e:
        print(f"✗ Error: {str(e)}")


def example_6_multi_market():
    """Example 6: Multi-market trading (US + India)"""
    print("\n" + "=" * 60)
    print("Example 6: Multi-Market Trading (US + India)")
    print("=" * 60)

    broker_config = BrokerConfig('configs/broker_config.yaml')
    broker_config.credential_manager.load_credentials()

    # Create brokers for both markets
    brokers = {}

    # US Broker
    try:
        alpaca = BrokerFactory.create_from_config('alpaca', broker_config)
        if alpaca.is_connected:
            brokers['alpaca_us'] = alpaca
            print("✓ US market: Alpaca connected")
    except Exception as e:
        print(f"⚠ Alpaca failed: {str(e)}")

    # Indian Broker
    try:
        zerodha = BrokerFactory.create_from_config('zerodha', broker_config)
        if zerodha.is_connected:
            brokers['zerodha_in'] = zerodha
            print("✓ Indian market: Zerodha connected")
    except Exception as e:
        print(f"⚠ Zerodha failed: {str(e)}")

    if brokers:
        # Create order manager
        order_manager = OrderManager(brokers, default_broker='alpaca_us')

        # Get combined account summary
        print("\nMulti-market account summary:")
        accounts = order_manager.get_account_summary()

        total_value_usd = 0
        for name, account in accounts.items():
            print(f"\n{name}:")
            print(f"  Portfolio Value: {account.currency} {account.portfolio_value:,.2f}")

            # Convert to USD for total (simplified - use actual FX rates)
            if account.currency == 'INR':
                value_usd = account.portfolio_value / 83.0  # Approximate conversion
            else:
                value_usd = account.portfolio_value

            total_value_usd += value_usd

        print(f"\nTotal Portfolio Value (USD): ${total_value_usd:,.2f}")

        # Cleanup
        for broker in brokers.values():
            broker.disconnect()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("BROKER INTEGRATION EXAMPLES")
    print("=" * 60)
    print("\nNote: These examples require proper broker configuration.")
    print("Run 'python scripts/broker_cli.py setup' to create configuration.")
    print()

    # Run examples
    # example_1_basic_connection()
    # example_2_market_data()
    # example_3_unified_data_fetcher()
    # example_4_order_management()
    # example_5_zerodha_india()
    # example_6_multi_market()

    print("\n✓ Examples completed!")
    print("\nTo run individual examples, uncomment the function calls in main()")


if __name__ == '__main__':
    main()

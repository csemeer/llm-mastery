# Broker Integration Guide

Complete guide for integrating US and Indian brokers with the Financial LLM trading system.

## Table of Contents

1. [Overview](#overview)
2. [Supported Brokers](#supported-brokers)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [API Reference](#api-reference)
8. [Security](#security)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The broker integration module provides a unified interface for connecting to major US and Indian brokers. It supports:

- **Market Data**: Real-time quotes and historical OHLCV data
- **Order Execution**: Market, limit, stop, and advanced order types
- **Account Management**: Positions, balances, and portfolio tracking
- **Multi-Broker Support**: Use multiple brokers simultaneously with automatic failover
- **Risk Controls**: Position limits, order value limits, and daily loss limits

### Key Features

✅ **Unified API** - Single interface for all brokers
✅ **Secure Credentials** - Encrypted storage with password protection
✅ **Automatic Failover** - Redundant data sources
✅ **Paper Trading** - Test strategies without risk
✅ **Multi-Market** - Trade US and Indian markets from one system

---

## Supported Brokers

### US Markets 🇺🇸

| Broker | Type | Markets | Features |
|--------|------|---------|----------|
| **Alpaca** | `alpaca` | US stocks, ETFs | ✅ Paper trading, Real-time data, Options |
| **Interactive Brokers** | `ibkr` | Global markets | ✅ Stocks, Options, Futures, Forex |
| **TD Ameritrade** | `td_ameritrade` | US stocks, ETFs | ✅ Options, Real-time data |

### Indian Markets 🇮🇳

| Broker | Type | Markets | Features |
|--------|------|---------|----------|
| **Zerodha Kite** | `zerodha` | NSE, BSE, NFO | ✅ Equity, F&O, Currency, Commodity |
| **Angel One** | `angel_one` | NSE, BSE | ✅ Equity, F&O, SmartAPI |
| **Upstox** | `upstox` | NSE, BSE | ✅ Equity, F&O, API v2 |
| **ICICI Direct** | `icici_direct` | NSE, BSE | ⚠️ Institutional only |

---

## Installation

### 1. Install Dependencies

```bash
# Install broker integration requirements
pip install -r requirements_brokers.txt
```

### 2. Broker-Specific Setup

#### For Alpaca (US)
```bash
pip install alpaca-py
```
- Sign up at [alpaca.markets](https://alpaca.markets)
- Get API keys from dashboard
- Use paper trading for testing

#### For Interactive Brokers (US)
```bash
pip install ib-insync
```
- Download [IB Gateway](https://www.interactivebrokers.com/en/trading/ib-gateway.php) or TWS
- Enable API connections in settings
- Configure port (7497 for paper, 7496 for live)

#### For TD Ameritrade (US)
```bash
pip install tda-api selenium
```
- Sign up at [developer.tdameritrade.com](https://developer.tdameritrade.com)
- Create app and get API key
- Note: TD Ameritrade is merging with Schwab

#### For Zerodha (India)
```bash
pip install kiteconnect
```
- Subscribe to [Kite Connect](https://kite.trade/)
- Get API key and secret
- Complete OAuth flow for access token

#### For Angel One (India)
```bash
pip install smartapi-python
```
- Register at [Angel One SmartAPI](https://smartapi.angelbroking.com/)
- Get API credentials
- Generate TOTP token

#### For Upstox (India)
```bash
pip install upstox-python-sdk
```
- Register at [Upstox Developer](https://upstox.com/developer/)
- Get API credentials
- Complete OAuth flow

---

## Quick Start

### 1. Create Configuration

```bash
python scripts/broker_cli.py setup
```

This creates `configs/broker_config.yaml` with all supported brokers.

### 2. Add Credentials

```bash
python scripts/broker_cli.py add-credentials
```

Follow the prompts to add API keys securely. Credentials are encrypted and stored in `.broker_credentials.enc`.

### 3. Test Connection

```bash
python scripts/broker_cli.py test alpaca
```

### 4. Get Market Data

```bash
# Get real-time quote
python scripts/broker_cli.py quote alpaca AAPL

# Get historical data
python scripts/broker_cli.py history alpaca AAPL --timeframe 1Day --lookback 30
```

---

## Configuration

### Broker Configuration File

Edit `configs/broker_config.yaml` to configure brokers:

```yaml
markets:
  US:
    market: US
    timezone: America/New_York
    trading_hours_start: '09:30'
    trading_hours_end: '16:00'
    currency: USD

  IN:
    market: IN
    timezone: Asia/Kolkata
    trading_hours_start: '09:15'
    trading_hours_end: '15:30'
    currency: INR

brokers:
  alpaca:
    type: alpaca
    market: US
    enabled: true
    paper_trading: true
    base_url: https://paper-api.alpaca.markets
    data_feed: iex  # 'iex' or 'sip'

  zerodha:
    type: zerodha
    market: IN
    enabled: true
    paper_trading: false
    exchange: NSE  # NSE or BSE
```

### Credential Storage

Credentials are stored encrypted in `.broker_credentials.enc`. Never commit this file to version control.

**Add to `.gitignore`:**
```
.broker_credentials.enc
td_token.json
*.token
*.secret
```

---

## Usage Examples

### Example 1: Basic Connection

```python
from financial_llm.brokers import BrokerFactory, BrokerConfig

# Load configuration
config = BrokerConfig('configs/broker_config.yaml')
config.credential_manager.load_credentials()

# Create broker instance
broker = BrokerFactory.create_from_config('alpaca', config)

# Use broker
account = broker.get_account()
print(f"Buying Power: ${account.buying_power:,.2f}")

broker.disconnect()
```

### Example 2: Fetch Market Data

```python
from financial_llm.brokers import BrokerFactory, BrokerConfig

config = BrokerConfig('configs/broker_config.yaml')
config.credential_manager.load_credentials()
broker = BrokerFactory.create_from_config('alpaca', config)

# Historical data
data = broker.get_historical_data(
    symbol='AAPL',
    timeframe='1Day',
    lookback=60
)
print(data.tail())

# Real-time quote
quote = broker.get_quote('AAPL')
print(f"Last: ${quote.last_price:.2f}")
```

### Example 3: Place Orders

```python
from financial_llm.brokers import BrokerFactory, BrokerConfig, OrderSide, OrderType

config = BrokerConfig('configs/broker_config.yaml')
config.credential_manager.load_credentials()
broker = BrokerFactory.create_from_config('alpaca', config)

# Place market order
order = broker.place_order(
    symbol='AAPL',
    quantity=10,
    side=OrderSide.BUY,
    order_type=OrderType.MARKET
)
print(f"Order ID: {order.order_id}")

# Place limit order
order = broker.place_order(
    symbol='AAPL',
    quantity=10,
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    limit_price=150.00
)
```

### Example 4: Unified Data Fetcher

```python
from financial_llm.brokers import BrokerFactory, BrokerConfig, UnifiedDataFetcher

config = BrokerConfig('configs/broker_config.yaml')
config.credential_manager.load_credentials()

# Create multiple brokers
brokers = []
brokers.append(BrokerFactory.create_from_config('alpaca', config))
# Add more brokers...

# Create unified fetcher
fetcher = UnifiedDataFetcher(brokers, primary_broker='alpaca')

# Fetch with automatic fallback
data = fetcher.get_historical_data(
    symbol='AAPL',
    timeframe='1Day',
    lookback=30,
    use_fallback=True  # Automatically tries other brokers if primary fails
)
```

### Example 5: Order Manager

```python
from financial_llm.brokers import BrokerFactory, BrokerConfig, OrderManager, OrderSide, OrderType

config = BrokerConfig('configs/broker_config.yaml')
config.credential_manager.load_credentials()

# Create brokers
brokers = {
    'alpaca': BrokerFactory.create_from_config('alpaca', config),
    'zerodha': BrokerFactory.create_from_config('zerodha', config),
}

# Create order manager
order_mgr = OrderManager(brokers, default_broker='alpaca')

# Set risk controls
order_mgr.set_max_position_size('AAPL', 100)
order_mgr.set_max_order_value(10000)
order_mgr.set_daily_loss_limit(500)

# Place order with risk controls
order_record = order_mgr.place_order(
    symbol='AAPL',
    quantity=10,
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    limit_price=150.00,
    broker_name='alpaca'
)

# Get positions across all brokers
positions = order_mgr.get_positions(aggregate=True)
for symbol, pos in positions.items():
    print(f"{symbol}: {pos.quantity} shares")
```

### Example 6: Multi-Market Trading

```python
from financial_llm.brokers import BrokerFactory, BrokerConfig, OrderManager

config = BrokerConfig('configs/broker_config.yaml')
config.credential_manager.load_credentials()

brokers = {
    'us_alpaca': BrokerFactory.create_from_config('alpaca', config),
    'in_zerodha': BrokerFactory.create_from_config('zerodha', config),
}

order_mgr = OrderManager(brokers)

# Trade US stocks
order_mgr.place_order(
    symbol='AAPL',
    quantity=10,
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    broker_name='us_alpaca'
)

# Trade Indian stocks
order_mgr.place_order(
    symbol='RELIANCE',
    quantity=100,
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    broker_name='in_zerodha'
)

# Get combined portfolio
accounts = order_mgr.get_account_summary()
for broker_name, account in accounts.items():
    print(f"{broker_name}: {account.currency} {account.portfolio_value:,.2f}")
```

---

## API Reference

### BrokerFactory

Create broker instances:

```python
from financial_llm.brokers import BrokerFactory

# Create from config dict
broker = BrokerFactory.create('alpaca', config_dict)

# Create from BrokerConfig
broker = BrokerFactory.create_from_config('alpaca', broker_config)

# Get supported brokers
supported = BrokerFactory.get_supported_brokers()
```

### BaseBroker Interface

All brokers implement this interface:

```python
# Connection
broker.connect() -> bool
broker.disconnect() -> bool
broker.is_connected -> bool

# Market Data
broker.get_historical_data(symbol, timeframe, start_date, end_date, lookback) -> DataFrame
broker.get_quote(symbol) -> Quote
broker.get_quotes(symbols) -> Dict[str, Quote]

# Account
broker.get_account() -> AccountInfo
broker.get_positions() -> List[Position]
broker.get_position(symbol) -> Position

# Orders
broker.place_order(symbol, quantity, side, order_type, ...) -> Order
broker.cancel_order(order_id) -> bool
broker.get_order(order_id) -> Order
broker.get_orders(status, limit) -> List[Order]

# Capabilities
broker.get_capabilities() -> List[BrokerCapability]
```

### UnifiedDataFetcher

Fetch data from multiple brokers with failover:

```python
fetcher = UnifiedDataFetcher(brokers, primary_broker='alpaca')

# Fetch with automatic failover
data = fetcher.get_historical_data(symbol, timeframe, ...)
quote = fetcher.get_quote(symbol)
quotes = fetcher.get_quotes(symbols)

# Get status
status = fetcher.get_broker_status()
```

### OrderManager

Manage orders across brokers with risk controls:

```python
order_mgr = OrderManager(brokers, default_broker='alpaca')

# Place orders
order_record = order_mgr.place_order(symbol, quantity, side, ...)

# Cancel orders
order_mgr.cancel_order(order_id)
order_mgr.cancel_all_orders(symbol)

# Get orders
orders = order_mgr.get_orders(status, symbol, broker_name)
fills = order_mgr.get_fills()
open_orders = order_mgr.get_open_orders()

# Positions
positions = order_mgr.get_positions(aggregate=True)
accounts = order_mgr.get_account_summary()

# Risk controls
order_mgr.set_max_position_size(symbol, size)
order_mgr.set_max_order_value(value)
order_mgr.set_daily_loss_limit(limit)
```

---

## Security

### Best Practices

1. **Never commit credentials** to version control
2. **Use paper trading** for testing
3. **Enable MFA** on broker accounts
4. **Rotate API keys** regularly
5. **Use encrypted credentials** storage
6. **Set risk limits** on all accounts

### Credential Encryption

Credentials are encrypted using Fernet (symmetric encryption) with a password-derived key (PBKDF2):

```python
from financial_llm.brokers import BrokerConfig, BrokerCredentials

config = BrokerConfig()

# Add credentials (will prompt for password)
credentials = BrokerCredentials(
    broker_name='alpaca',
    api_key='YOUR_API_KEY',
    api_secret='YOUR_API_SECRET',
    paper_trading=True
)

config.credential_manager.save_credentials('alpaca', credentials)

# Load credentials (will prompt for password)
config.credential_manager.load_credentials()
```

### Environment Variables

Alternatively, use environment variables:

```bash
export ALPACA_API_KEY="your_key"
export ALPACA_API_SECRET="your_secret"
export ZERODHA_API_KEY="your_key"
export ZERODHA_API_SECRET="your_secret"
```

```python
import os

config = {
    'api_key': os.getenv('ALPACA_API_KEY'),
    'api_secret': os.getenv('ALPACA_API_SECRET'),
    'paper_trading': True
}

broker = BrokerFactory.create('alpaca', config)
```

---

## Troubleshooting

### Connection Issues

**Problem:** Broker connection fails

**Solutions:**
- Verify API credentials are correct
- Check if API endpoint is accessible
- For IBKR, ensure IB Gateway/TWS is running
- For paper trading, use correct paper trading URL
- Check firewall and network settings

### Authentication Errors

**Problem:** Invalid credentials or OAuth errors

**Solutions:**
- Regenerate API keys from broker dashboard
- For Zerodha/Upstox, complete OAuth flow
- Check if API subscription is active
- Verify account has API access enabled

### Data Fetching Errors

**Problem:** No data returned or errors

**Solutions:**
- Verify symbol format (AAPL for US, RELIANCE for India)
- Check market hours and trading calendar
- For Indian stocks, specify exchange (NSE:RELIANCE)
- Verify data subscription includes requested data type
- Use correct timeframe format

### Order Placement Errors

**Problem:** Orders rejected

**Solutions:**
- Check buying power and account balance
- Verify order parameters (price, quantity)
- Check if symbol is tradeable
- Ensure market is open (or use extended hours)
- Review risk controls and position limits

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing dependency | Install broker-specific package |
| `Authentication failed` | Invalid credentials | Check API keys |
| `Insufficient funds` | Not enough buying power | Add funds or reduce order size |
| `Symbol not found` | Invalid symbol | Verify symbol format |
| `Market closed` | Trading outside hours | Wait for market open |

---

## Advanced Topics

### Real-Time Data Streaming

For real-time tick data, use WebSocket connections (broker-specific):

```python
# Alpaca WebSocket example
from alpaca.data.live import StockDataStream

stream = StockDataStream(api_key, secret_key)

async def quote_handler(data):
    print(f"{data.symbol}: ${data.ask_price}")

stream.subscribe_quotes(quote_handler, 'AAPL')
stream.run()
```

### Multi-Threaded Order Execution

Execute orders across multiple brokers in parallel:

```python
import concurrent.futures

def place_order_on_broker(broker_name, symbol, quantity):
    order = order_mgr.place_order(
        symbol=symbol,
        quantity=quantity,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        broker_name=broker_name
    )
    return order

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(place_order_on_broker, 'alpaca', 'AAPL', 10),
        executor.submit(place_order_on_broker, 'zerodha', 'RELIANCE', 100),
    ]

    for future in concurrent.futures.as_completed(futures):
        order = future.result()
        print(f"Order placed: {order.order_id}")
```

### Custom Broker Implementation

Extend `BaseBroker` to add new brokers:

```python
from financial_llm.brokers.base import BaseBroker

class CustomBroker(BaseBroker):
    def connect(self):
        # Implement connection logic
        pass

    def get_historical_data(self, ...):
        # Implement data fetching
        pass

    # Implement all required methods...
```

---

## Support

For issues and questions:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/llm-mastery/issues)
- **Documentation**: [Full documentation](https://docs.yourproject.com)
- **Broker Support**: Contact broker directly for API access issues

---

## License

MIT License - See LICENSE file for details.

---

## Changelog

### Version 1.0.0 (2024)
- ✅ Initial release
- ✅ Support for 7 major brokers (US + India)
- ✅ Unified API interface
- ✅ Encrypted credential storage
- ✅ Multi-broker order management
- ✅ Risk controls
- ✅ Paper trading support

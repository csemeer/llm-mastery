# Broker Integration Module

Production-ready broker integrations for US and Indian markets.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements_brokers.txt

# 2. Create configuration
python scripts/broker_cli.py setup

# 3. Add credentials
python scripts/broker_cli.py add-credentials

# 4. Test connection
python scripts/broker_cli.py test alpaca
```

## Supported Brokers

### US Markets
- ✅ **Alpaca** - Commission-free trading, paper trading
- ✅ **Interactive Brokers** - Global markets, advanced features
- ✅ **TD Ameritrade** - Full-service broker

### Indian Markets
- ✅ **Zerodha Kite** - Leading discount broker
- ✅ **Angel One** - SmartAPI integration
- ✅ **Upstox** - Modern API platform
- ⚠️ **ICICI Direct** - Institutional access only

## Features

- 🔐 **Secure Credential Storage** - Encrypted with password protection
- 🌐 **Multi-Broker Support** - Use multiple brokers simultaneously
- 📊 **Unified Data Interface** - Single API for all brokers
- 🛡️ **Risk Controls** - Position limits, order value limits
- 📈 **Paper Trading** - Test strategies without risk
- 🔄 **Automatic Failover** - Redundant data sources

## Architecture

```
financial_llm/brokers/
├── base.py              # Abstract base class
├── factory.py           # Broker factory
├── config.py            # Configuration management
├── data_fetcher.py      # Unified data fetching
├── order_manager.py     # Order management
├── us/                  # US broker implementations
│   ├── alpaca_broker.py
│   ├── ibkr_broker.py
│   └── td_ameritrade_broker.py
└── india/               # Indian broker implementations
    ├── zerodha_broker.py
    ├── angel_one_broker.py
    ├── upstox_broker.py
    └── icici_direct_broker.py
```

## Usage Example

```python
from financial_llm.brokers import BrokerFactory, BrokerConfig

# Load configuration
config = BrokerConfig('configs/broker_config.yaml')
config.credential_manager.load_credentials()

# Create broker
broker = BrokerFactory.create_from_config('alpaca', config)

# Fetch data
data = broker.get_historical_data('AAPL', '1Day', lookback=60)
quote = broker.get_quote('AAPL')

# Get account info
account = broker.get_account()
positions = broker.get_positions()

# Place order
from financial_llm.brokers import OrderSide, OrderType

order = broker.place_order(
    symbol='AAPL',
    quantity=10,
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    limit_price=150.00
)
```

## Documentation

See [BROKER_INTEGRATION.md](../../docs/BROKER_INTEGRATION.md) for complete documentation.

## Security

⚠️ **Never commit credentials to version control**

Add to `.gitignore`:
```
.broker_credentials.enc
*.token
*.secret
```

## License

MIT License

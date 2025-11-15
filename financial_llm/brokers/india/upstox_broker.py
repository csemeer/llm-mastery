"""
Upstox Broker Integration

Connects to Upstox API for Indian market trading.

Documentation: https://upstox.com/developer/api-documentation
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

try:
    from upstox_client import Configuration, ApiClient, LoginApi, OrderApi, PortfolioApi, MarketQuoteApi
    UPSTOX_AVAILABLE = True
except ImportError:
    UPSTOX_AVAILABLE = False

from ..base import (
    BaseBroker, BrokerCapability, OrderSide, OrderType, OrderStatus,
    Order, Position, AccountInfo, Quote
)


class UpstoxBroker(BaseBroker):
    """Upstox API implementation for Indian markets"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if not UPSTOX_AVAILABLE:
            raise ImportError("upstox_client not installed. Install with: pip install upstox-python-sdk")

        self.api_key = config['api_key']
        self.api_secret = config.get('api_secret')
        self.redirect_uri = config.get('redirect_uri', 'https://localhost')
        self.access_token = config.get('access_token')
        self.exchange = config.get('exchange', 'NSE')

        self.api_client: Optional[ApiClient] = None

    def connect(self) -> bool:
        try:
            configuration = Configuration()

            if self.access_token:
                configuration.access_token = self.access_token
            else:
                print("Upstox authentication required.")
                print(f"Get auth code from: https://api.upstox.com/v2/login/authorization/dialog?")
                print(f"response_type=code&client_id={self.api_key}&redirect_uri={self.redirect_uri}")

                auth_code = input("Enter authorization code: ")

                # Exchange code for access token
                login_api = LoginApi()
                token_response = login_api.token(
                    api_version='2.0',
                    code=auth_code,
                    client_id=self.api_key,
                    client_secret=self.api_secret,
                    redirect_uri=self.redirect_uri,
                    grant_type='authorization_code'
                )

                self.access_token = token_response.access_token
                configuration.access_token = self.access_token

                print(f"Access token: {self.access_token}")
                print("Save this token for future use!")

            self.api_client = ApiClient(configuration)

            self.is_connected = True
            print("✓ Connected to Upstox")
            return True

        except Exception as e:
            print(f"✗ Failed to connect to Upstox: {str(e)}")
            self.is_connected = False
            return False

    def disconnect(self) -> bool:
        self.api_client = None
        self.is_connected = False
        print("✓ Disconnected from Upstox")
        return True

    def get_capabilities(self) -> List[BrokerCapability]:
        return [
            BrokerCapability.MARKET_DATA,
            BrokerCapability.HISTORICAL_DATA,
            BrokerCapability.REAL_TIME_QUOTES,
            BrokerCapability.ORDER_EXECUTION,
            BrokerCapability.OPTIONS_TRADING,
            BrokerCapability.FUTURES_TRADING,
        ]

    def get_historical_data(self, symbol: str, timeframe: str,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          lookback: Optional[int] = None) -> pd.DataFrame:
        if not self.is_connected:
            raise RuntimeError("Not connected to Upstox")

        print("⚠ Upstox historical data requires instrument key mapping")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def get_quote(self, symbol: str) -> Quote:
        if not self.is_connected:
            raise RuntimeError("Not connected to Upstox")

        try:
            market_api = MarketQuoteApi(self.api_client)
            instrument_key = f"{self.exchange}_EQ|{symbol}"  # Simplified instrument key

            quote_response = market_api.ltp(instrument_key, 'latest')

            if quote_response.status == 'success':
                data = quote_response.data[instrument_key]
                return Quote(
                    symbol=symbol,
                    bid_price=data.last_price,
                    ask_price=data.last_price,
                    bid_size=0,
                    ask_size=0,
                    last_price=data.last_price,
                    last_size=0,
                    volume=0,
                    timestamp=datetime.now(),
                    exchange=self.exchange
                )
            else:
                raise ValueError(f"Failed to get quote: {quote_response.status}")

        except Exception as e:
            raise ValueError(f"Failed to get quote: {str(e)}")

    def get_account(self) -> AccountInfo:
        if not self.is_connected:
            raise RuntimeError("Not connected to Upstox")

        try:
            portfolio_api = PortfolioApi(self.api_client)
            funds = portfolio_api.get_user_funds_and_margin()

            if funds.status == 'success':
                equity = funds.data.equity
                return AccountInfo(
                    account_id='upstox_account',
                    buying_power=equity.available_margin,
                    cash=equity.available_margin,
                    portfolio_value=equity.used_margin + equity.available_margin,
                    equity=equity.used_margin + equity.available_margin,
                    currency='INR',
                    margin_used=equity.used_margin,
                    last_updated=datetime.now()
                )
            else:
                raise RuntimeError(f"Failed to get funds: {funds.status}")

        except Exception as e:
            raise RuntimeError(f"Failed to get account info: {str(e)}")

    def get_positions(self) -> List[Position]:
        if not self.is_connected:
            raise RuntimeError("Not connected to Upstox")

        try:
            portfolio_api = PortfolioApi(self.api_client)
            positions_response = portfolio_api.get_positions()

            if positions_response.status != 'success':
                return []

            positions = []
            for pos in positions_response.data:
                if pos.quantity == 0:
                    continue

                positions.append(Position(
                    symbol=pos.trading_symbol,
                    quantity=abs(pos.quantity),
                    avg_entry_price=pos.average_price,
                    current_price=pos.last_price,
                    market_value=pos.quantity * pos.last_price,
                    unrealized_pnl=pos.pnl,
                    unrealized_pnl_percent=0,
                    side='long' if pos.quantity > 0 else 'short',
                    exchange=pos.exchange
                ))

            return positions

        except Exception as e:
            raise RuntimeError(f"Failed to get positions: {str(e)}")

    def place_order(self, symbol: str, quantity: float, side: OrderSide,
                   order_type: OrderType, limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None, time_in_force: str = "day",
                   **kwargs) -> Order:
        if not self.is_connected:
            raise RuntimeError("Not connected to Upstox")

        print("⚠ Upstox order placement requires proper instrument key and implementation")
        raise NotImplementedError("Upstox order placement not fully implemented")

    def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected:
            raise RuntimeError("Not connected to Upstox")

        try:
            order_api = OrderApi(self.api_client)
            response = order_api.cancel_order(order_id)
            return response.status == 'success'
        except Exception as e:
            print(f"✗ Failed to cancel order: {str(e)}")
            return False

    def get_order(self, order_id: str) -> Order:
        raise NotImplementedError("Upstox get_order not fully implemented")

    def get_orders(self, status: Optional[OrderStatus] = None, limit: int = 100) -> List[Order]:
        if not self.is_connected:
            raise RuntimeError("Not connected to Upstox")

        try:
            order_api = OrderApi(self.api_client)
            orders_response = order_api.get_order_book()

            if orders_response.status != 'success':
                return []

            # Convert orders (simplified)
            return []

        except Exception as e:
            raise RuntimeError(f"Failed to get orders: {str(e)}")

"""
Angel One Broker Integration

Connects to Angel One (formerly Angel Broking) SmartAPI for Indian market trading.

Documentation: https://smartapi.angelbroking.com/docs
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

try:
    from SmartApi import SmartConnect
    ANGEL_AVAILABLE = True
except ImportError:
    ANGEL_AVAILABLE = False

from ..base import (
    BaseBroker, BrokerCapability, OrderSide, OrderType, OrderStatus,
    Order, Position, AccountInfo, Quote
)


class AngelOneBroker(BaseBroker):
    """Angel One SmartAPI implementation for Indian markets"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if not ANGEL_AVAILABLE:
            raise ImportError("SmartApi not installed. Install with: pip install smartapi-python")

        self.api_key = config['api_key']
        self.client_code = config.get('client_code')
        self.password = config.get('password')
        self.totp_token = config.get('totp_token')
        self.exchange = config.get('exchange', 'NSE')

        self.smart_api: Optional[SmartConnect] = None

    def connect(self) -> bool:
        try:
            self.smart_api = SmartConnect(api_key=self.api_key)

            # Generate session
            data = self.smart_api.generateSession(
                clientCode=self.client_code,
                password=self.password,
                totp=self.totp_token
            )

            if data['status']:
                self.is_connected = True
                print(f"✓ Connected to Angel One (client: {self.client_code})")
                return True
            else:
                print(f"✗ Failed to connect: {data.get('message')}")
                return False

        except Exception as e:
            print(f"✗ Failed to connect to Angel One: {str(e)}")
            self.is_connected = False
            return False

    def disconnect(self) -> bool:
        if self.smart_api:
            try:
                self.smart_api.terminateSession(self.client_code)
            except:
                pass
        self.is_connected = False
        print("✓ Disconnected from Angel One")
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
            raise RuntimeError("Not connected to Angel One")

        # Angel One historical data implementation
        # Note: Requires symbol token which needs to be mapped
        print("⚠ Angel One historical data requires symbol token mapping")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def get_quote(self, symbol: str) -> Quote:
        if not self.is_connected:
            raise RuntimeError("Not connected to Angel One")

        # Get LTP (Last Traded Price)
        data = self.smart_api.ltpData(self.exchange, symbol, symbol)

        if data['status']:
            ltp = float(data['data']['ltp'])
            return Quote(
                symbol=symbol,
                bid_price=ltp,  # Angel One LTP endpoint doesn't provide bid/ask
                ask_price=ltp,
                bid_size=0,
                ask_size=0,
                last_price=ltp,
                last_size=0,
                volume=0,
                timestamp=datetime.now(),
                exchange=self.exchange
            )
        else:
            raise ValueError(f"Failed to get quote: {data.get('message')}")

    def get_account(self) -> AccountInfo:
        if not self.is_connected:
            raise RuntimeError("Not connected to Angel One")

        data = self.smart_api.rmsLimit()

        if data['status']:
            rms = data['data']
            return AccountInfo(
                account_id=self.client_code,
                buying_power=float(rms.get('net', 0)),
                cash=float(rms.get('availablecash', 0)),
                portfolio_value=float(rms.get('net', 0)),
                equity=float(rms.get('net', 0)),
                currency='INR',
                last_updated=datetime.now()
            )
        else:
            raise RuntimeError(f"Failed to get account info: {data.get('message')}")

    def get_positions(self) -> List[Position]:
        if not self.is_connected:
            raise RuntimeError("Not connected to Angel One")

        data = self.smart_api.position()

        if not data['status']:
            return []

        positions = []
        for pos in data.get('data', []):
            if pos['netqty'] == '0':
                continue

            positions.append(Position(
                symbol=pos['tradingsymbol'],
                quantity=abs(int(pos['netqty'])),
                avg_entry_price=float(pos['netprice']),
                current_price=float(pos['ltp']),
                market_value=float(pos['netvalue']),
                unrealized_pnl=float(pos['pnl']),
                unrealized_pnl_percent=0,
                side='long' if int(pos['netqty']) > 0 else 'short',
                exchange=pos['exchange']
            ))

        return positions

    def place_order(self, symbol: str, quantity: float, side: OrderSide,
                   order_type: OrderType, limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None, time_in_force: str = "day",
                   **kwargs) -> Order:
        if not self.is_connected:
            raise RuntimeError("Not connected to Angel One")

        order_params = {
            'variety': kwargs.get('variety', 'NORMAL'),
            'tradingsymbol': symbol,
            'symboltoken': kwargs.get('symboltoken', symbol),  # Requires token mapping
            'transactiontype': 'BUY' if side == OrderSide.BUY else 'SELL',
            'exchange': kwargs.get('exchange', self.exchange),
            'ordertype': 'MARKET' if order_type == OrderType.MARKET else 'LIMIT',
            'producttype': kwargs.get('producttype', 'INTRADAY'),
            'duration': 'DAY',
            'quantity': int(quantity),
        }

        if limit_price:
            order_params['price'] = limit_price

        data = self.smart_api.placeOrder(order_params)

        if data['status']:
            return Order(
                order_id=data['data']['orderid'],
                symbol=symbol,
                quantity=float(quantity),
                side=side,
                order_type=order_type,
                status=OrderStatus.SUBMITTED,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                limit_price=limit_price,
                submitted_at=datetime.now()
            )
        else:
            raise RuntimeError(f"Failed to place order: {data.get('message')}")

    def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected:
            raise RuntimeError("Not connected to Angel One")

        try:
            data = self.smart_api.cancelOrder(order_id, 'NORMAL')
            return data['status']
        except Exception as e:
            print(f"✗ Failed to cancel order: {str(e)}")
            return False

    def get_order(self, order_id: str) -> Order:
        if not self.is_connected:
            raise RuntimeError("Not connected to Angel One")

        data = self.smart_api.orderBook()
        if data['status']:
            for order_data in data.get('data', []):
                if order_data['orderid'] == order_id:
                    return self._convert_angel_order(order_data)

        raise ValueError(f"Order {order_id} not found")

    def _convert_angel_order(self, order_data: Dict) -> Order:
        status_mapping = {
            'open': OrderStatus.SUBMITTED,
            'complete': OrderStatus.FILLED,
            'rejected': OrderStatus.REJECTED,
            'cancelled': OrderStatus.CANCELED,
        }

        return Order(
            order_id=order_data['orderid'],
            symbol=order_data['tradingsymbol'],
            quantity=float(order_data['quantity']),
            side=OrderSide.BUY if order_data['transactiontype'] == 'BUY' else OrderSide.SELL,
            order_type=OrderType.MARKET if order_data['ordertype'] == 'MARKET' else OrderType.LIMIT,
            status=status_mapping.get(order_data['status'].lower(), OrderStatus.PENDING),
            filled_quantity=float(order_data.get('filledshares', 0)),
            avg_fill_price=float(order_data.get('averageprice', 0)),
            submitted_at=datetime.now()
        )

    def get_orders(self, status: Optional[OrderStatus] = None, limit: int = 100) -> List[Order]:
        if not self.is_connected:
            raise RuntimeError("Not connected to Angel One")

        data = self.smart_api.orderBook()
        if not data['status']:
            return []

        orders = [self._convert_angel_order(o) for o in data.get('data', [])]

        if status:
            orders = [o for o in orders if o.status == status]

        return orders[:limit]

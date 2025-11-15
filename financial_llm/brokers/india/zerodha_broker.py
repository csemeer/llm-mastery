"""
Zerodha Kite Broker Integration

Connects to Zerodha Kite API for Indian market trading.
Supports NSE, BSE, NFO, MCX, and other Indian exchanges.

Documentation: https://kite.trade/docs/connect/v3/

Note: Requires Kite Connect subscription and API credentials.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

try:
    from kiteconnect import KiteConnect
    ZERODHA_AVAILABLE = True
except ImportError:
    ZERODHA_AVAILABLE = False

from ..base import (
    BaseBroker, BrokerCapability, OrderSide, OrderType, OrderStatus,
    Order, Position, AccountInfo, Quote
)


class ZerodhaBroker(BaseBroker):
    """
    Zerodha Kite broker implementation.

    Supports:
    - NSE, BSE equity trading
    - F&O (Futures & Options)
    - Currency and commodity trading
    - Real-time and historical data
    - WebSocket for live data
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Zerodha broker.

        Required config:
            - api_key: Kite Connect API key
            - api_secret: Kite Connect API secret
            - access_token: Access token (obtained via OAuth)
            - exchange: Default exchange (NSE, BSE, NFO, etc.)
        """
        super().__init__(config)

        if not ZERODHA_AVAILABLE:
            raise ImportError(
                "kiteconnect not installed. Install with: "
                "pip install kiteconnect"
            )

        self.api_key = config['api_key']
        self.api_secret = config.get('api_secret')
        self.access_token = config.get('access_token')
        self.exchange = config.get('exchange', 'NSE')

        self.kite: Optional[KiteConnect] = None

    def connect(self) -> bool:
        """Connect to Zerodha Kite"""
        try:
            self.kite = KiteConnect(api_key=self.api_key)

            if self.access_token:
                # Use existing access token
                self.kite.set_access_token(self.access_token)
            else:
                # Need to generate access token via OAuth
                print("Zerodha login required.")
                print(f"Visit: {self.kite.login_url()}")
                request_token = input("Enter request token: ")

                # Generate access token
                data = self.kite.generate_session(request_token, api_secret=self.api_secret)
                self.access_token = data['access_token']
                self.kite.set_access_token(self.access_token)

                print(f"Access token: {self.access_token}")
                print("Save this token for future use!")

            # Validate connection
            profile = self.kite.profile()

            self.is_connected = True
            print(f"✓ Connected to Zerodha Kite (user: {profile['user_name']})")
            return True

        except Exception as e:
            print(f"✗ Failed to connect to Zerodha: {str(e)}")
            self.is_connected = False
            return False

    def disconnect(self) -> bool:
        """Disconnect from Zerodha"""
        # Zerodha doesn't have explicit disconnect
        self.kite = None
        self.is_connected = False
        print("✓ Disconnected from Zerodha")
        return True

    def get_capabilities(self) -> List[BrokerCapability]:
        """Get Zerodha capabilities"""
        return [
            BrokerCapability.MARKET_DATA,
            BrokerCapability.HISTORICAL_DATA,
            BrokerCapability.REAL_TIME_QUOTES,
            BrokerCapability.REAL_TIME_BARS,
            BrokerCapability.ORDER_EXECUTION,
            BrokerCapability.OPTIONS_TRADING,
            BrokerCapability.FUTURES_TRADING,
            BrokerCapability.BRACKET_ORDERS,
            BrokerCapability.STOP_LOSS,
            BrokerCapability.TAKE_PROFIT,
        ]

    def _get_instrument_token(self, symbol: str, exchange: Optional[str] = None) -> str:
        """Get trading symbol in Zerodha format"""
        exchange = exchange or self.exchange
        return f"{exchange}:{symbol}"

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Zerodha format"""
        mapping = {
            '1Min': 'minute',
            '3Min': '3minute',
            '5Min': '5minute',
            '10Min': '10minute',
            '15Min': '15minute',
            '30Min': '30minute',
            '1Hour': '60minute',
            '1Day': 'day',
        }
        if timeframe not in mapping:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return mapping[timeframe]

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetch historical data from Zerodha"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Zerodha")

        # Determine date range
        if end_date is None:
            end_date = datetime.now()

        if start_date is None and lookback is not None:
            # Estimate start date
            if 'Min' in timeframe:
                minutes = int(timeframe.replace('Min', ''))
                start_date = end_date - timedelta(days=lookback * minutes // 375)  # ~375 mins per trading day
            elif 'Hour' in timeframe:
                start_date = end_date - timedelta(days=lookback // 6)
            elif 'Day' in timeframe:
                start_date = end_date - timedelta(days=lookback)
            else:
                start_date = end_date - timedelta(days=lookback)

        if start_date is None:
            start_date = end_date - timedelta(days=365)

        # Get instrument token
        instrument_token = self._get_instrument_token(symbol)

        # Fetch historical data
        interval = self._convert_timeframe(timeframe)

        try:
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=start_date,
                to_date=end_date,
                interval=interval
            )

            if not data:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convert to DataFrame
            df = pd.DataFrame(data)
            df = df.rename(columns={'date': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df[['open', 'high', 'low', 'close', 'volume']]

            if lookback is not None:
                df = df.tail(lookback)

            return df

        except Exception as e:
            print(f"✗ Error fetching historical data: {str(e)}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def get_quote(self, symbol: str) -> Quote:
        """Get real-time quote from Zerodha"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Zerodha")

        instrument_token = self._get_instrument_token(symbol)

        try:
            quotes = self.kite.quote([instrument_token])

            if instrument_token in quotes:
                q = quotes[instrument_token]

                # Zerodha provides depth data
                depth = q.get('depth', {})
                buy_depth = depth.get('buy', [{}])
                sell_depth = depth.get('sell', [{}])

                return Quote(
                    symbol=symbol,
                    bid_price=float(buy_depth[0].get('price', 0)) if buy_depth else 0.0,
                    ask_price=float(sell_depth[0].get('price', 0)) if sell_depth else 0.0,
                    bid_size=int(buy_depth[0].get('quantity', 0)) if buy_depth else 0,
                    ask_size=int(sell_depth[0].get('quantity', 0)) if sell_depth else 0,
                    last_price=float(q.get('last_price', 0)),
                    last_size=0,
                    volume=int(q.get('volume', 0)),
                    timestamp=datetime.now(),
                    exchange=self.exchange
                )
            else:
                raise ValueError(f"No quote available for {symbol}")

        except Exception as e:
            raise ValueError(f"Failed to get quote: {str(e)}")

    def get_account(self) -> AccountInfo:
        """Get Zerodha account information"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Zerodha")

        try:
            margins = self.kite.margins()

            # Get equity margin
            equity = margins.get('equity', {})

            return AccountInfo(
                account_id=self.kite.profile()['user_id'],
                buying_power=float(equity.get('available', {}).get('live_balance', 0)),
                cash=float(equity.get('available', {}).get('cash', 0)),
                portfolio_value=float(equity.get('net', 0)),
                equity=float(equity.get('net', 0)),
                currency='INR',
                margin_used=float(equity.get('utilised', {}).get('debits', 0)),
                last_updated=datetime.now()
            )

        except Exception as e:
            raise RuntimeError(f"Failed to get account info: {str(e)}")

    def get_positions(self) -> List[Position]:
        """Get all positions from Zerodha"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Zerodha")

        try:
            positions = self.kite.positions()

            # Zerodha returns 'net' and 'day' positions
            net_positions = positions.get('net', [])

            result = []
            for pos in net_positions:
                if pos['quantity'] == 0:
                    continue

                result.append(Position(
                    symbol=pos['tradingsymbol'],
                    quantity=abs(float(pos['quantity'])),
                    avg_entry_price=float(pos['average_price']),
                    current_price=float(pos['last_price']),
                    market_value=float(pos['value']),
                    unrealized_pnl=float(pos['pnl']),
                    unrealized_pnl_percent=0,  # Calculate manually
                    side='long' if pos['quantity'] > 0 else 'short',
                    exchange=pos['exchange']
                ))

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to get positions: {str(e)}")

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
        """Place order with Zerodha"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Zerodha")

        # Map order types
        kite_transaction_type = 'BUY' if side == OrderSide.BUY else 'SELL'

        kite_order_type_mapping = {
            OrderType.MARKET: 'MARKET',
            OrderType.LIMIT: 'LIMIT',
            OrderType.STOP: 'SL',
            OrderType.STOP_LIMIT: 'SL-M',
        }

        kite_order_type = kite_order_type_mapping.get(order_type)
        if not kite_order_type:
            raise ValueError(f"Unsupported order type: {order_type}")

        # Build order params
        order_params = {
            'tradingsymbol': symbol,
            'exchange': kwargs.get('exchange', self.exchange),
            'transaction_type': kite_transaction_type,
            'quantity': int(quantity),
            'order_type': kite_order_type,
            'product': kwargs.get('product', 'MIS'),  # MIS (intraday) or CNC (delivery)
            'validity': 'DAY' if time_in_force.lower() == 'day' else 'TTL',
        }

        if limit_price is not None:
            order_params['price'] = limit_price

        if stop_price is not None:
            order_params['trigger_price'] = stop_price

        try:
            # Place order
            order_id = self.kite.place_order(
                variety=kwargs.get('variety', 'regular'),
                **order_params
            )

            return Order(
                order_id=str(order_id),
                symbol=symbol,
                quantity=float(quantity),
                side=side,
                order_type=order_type,
                status=OrderStatus.SUBMITTED,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                limit_price=limit_price,
                stop_price=stop_price,
                submitted_at=datetime.now(),
                exchange=order_params['exchange']
            )

        except Exception as e:
            raise RuntimeError(f"Failed to place order: {str(e)}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Zerodha")

        try:
            self.kite.cancel_order(
                variety='regular',
                order_id=order_id
            )
            return True
        except Exception as e:
            print(f"✗ Failed to cancel order {order_id}: {str(e)}")
            return False

    def get_order(self, order_id: str) -> Order:
        """Get order details"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Zerodha")

        try:
            orders = self.kite.orders()
            for order_data in orders:
                if str(order_data['order_id']) == order_id:
                    return self._convert_zerodha_order(order_data)

            raise ValueError(f"Order {order_id} not found")

        except Exception as e:
            raise RuntimeError(f"Failed to get order: {str(e)}")

    def _convert_zerodha_order(self, order_data: Dict) -> Order:
        """Convert Zerodha order to our Order format"""
        status_mapping = {
            'OPEN': OrderStatus.SUBMITTED,
            'COMPLETE': OrderStatus.FILLED,
            'CANCELLED': OrderStatus.CANCELED,
            'REJECTED': OrderStatus.REJECTED,
            'TRIGGER PENDING': OrderStatus.PENDING,
        }

        type_mapping = {
            'MARKET': OrderType.MARKET,
            'LIMIT': OrderType.LIMIT,
            'SL': OrderType.STOP,
            'SL-M': OrderType.STOP_LIMIT,
        }

        return Order(
            order_id=str(order_data['order_id']),
            symbol=order_data['tradingsymbol'],
            quantity=float(order_data['quantity']),
            side=OrderSide.BUY if order_data['transaction_type'] == 'BUY' else OrderSide.SELL,
            order_type=type_mapping.get(order_data['order_type'], OrderType.MARKET),
            status=status_mapping.get(order_data['status'], OrderStatus.PENDING),
            filled_quantity=float(order_data['filled_quantity']),
            avg_fill_price=float(order_data['average_price']) if order_data['average_price'] else 0.0,
            limit_price=float(order_data['price']) if order_data['price'] else None,
            stop_price=float(order_data['trigger_price']) if order_data['trigger_price'] else None,
            submitted_at=order_data['order_timestamp'],
            exchange=order_data['exchange']
        )

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get list of orders"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Zerodha")

        try:
            orders_data = self.kite.orders()
            orders = [self._convert_zerodha_order(order_data) for order_data in orders_data]

            # Filter by status if specified
            if status:
                orders = [o for o in orders if o.status == status]

            return orders[:limit]

        except Exception as e:
            raise RuntimeError(f"Failed to get orders: {str(e)}")

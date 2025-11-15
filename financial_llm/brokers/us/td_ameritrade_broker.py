"""
TD Ameritrade Broker Integration

Connects to TD Ameritrade API for market data and trading.
Requires OAuth authentication and developer account.

Documentation: https://developer.tdameritrade.com/

Note: TD Ameritrade is merging with Charles Schwab. This integration
may need updates as the migration completes.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

try:
    import tda
    from tda import auth, client
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False

from ..base import (
    BaseBroker, BrokerCapability, OrderSide, OrderType, OrderStatus,
    Order, Position, AccountInfo, Quote
)


class TDAmeritradeBroker(BaseBroker):
    """
    TD Ameritrade broker implementation.

    Supports:
    - US stocks, ETFs, options
    - Real-time and historical data
    - Market, limit, stop orders
    - Paper and live trading
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TD Ameritrade broker.

        Required config:
            - api_key: TD Ameritrade API key (consumer key)
            - redirect_uri: OAuth redirect URI
            - token_path: Path to store OAuth token
            - account_id: TD Ameritrade account number
        """
        super().__init__(config)

        if not TDA_AVAILABLE:
            raise ImportError(
                "tda-api not installed. Install with: "
                "pip install tda-api"
            )

        self.api_key = config['api_key']
        self.redirect_uri = config.get('redirect_uri', 'https://localhost:8080')
        self.token_path = config.get('token_path', 'td_token.json')
        self.account_id = config.get('account_id')

        self.client: Optional[client.Client] = None

    def connect(self) -> bool:
        """Connect to TD Ameritrade using OAuth"""
        try:
            # Try to load existing token
            try:
                self.client = auth.client_from_token_file(
                    self.token_path,
                    self.api_key
                )
            except FileNotFoundError:
                # Need to authenticate
                print("TD Ameritrade authentication required.")
                print("Opening browser for OAuth authentication...")

                from selenium import webdriver

                with webdriver.Chrome() as driver:
                    self.client = auth.client_from_login_flow(
                        driver,
                        self.api_key,
                        self.redirect_uri,
                        self.token_path
                    )

            # Validate connection
            self.client.get_accounts()

            self.is_connected = True
            print("✓ Connected to TD Ameritrade")
            return True

        except Exception as e:
            print(f"✗ Failed to connect to TD Ameritrade: {str(e)}")
            self.is_connected = False
            return False

    def disconnect(self) -> bool:
        """Disconnect from TD Ameritrade"""
        self.client = None
        self.is_connected = False
        print("✓ Disconnected from TD Ameritrade")
        return True

    def get_capabilities(self) -> List[BrokerCapability]:
        """Get TD Ameritrade capabilities"""
        return [
            BrokerCapability.MARKET_DATA,
            BrokerCapability.HISTORICAL_DATA,
            BrokerCapability.REAL_TIME_QUOTES,
            BrokerCapability.ORDER_EXECUTION,
            BrokerCapability.OPTIONS_TRADING,
            BrokerCapability.STOP_LOSS,
        ]

    def _convert_timeframe(self, timeframe: str) -> tuple:
        """Convert timeframe to TDA format (periodType, period, frequencyType, frequency)"""
        mapping = {
            '1Min': ('day', 1, 'minute', 1),
            '5Min': ('day', 5, 'minute', 5),
            '15Min': ('day', 10, 'minute', 15),
            '1Hour': ('month', 1, 'minute', 60),
            '1Day': ('year', 1, 'daily', 1),
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
        """Fetch historical data from TD Ameritrade"""
        if not self.is_connected:
            raise RuntimeError("Not connected to TD Ameritrade")

        period_type, period, frequency_type, frequency = self._convert_timeframe(timeframe)

        # Get price history
        response = self.client.get_price_history(
            symbol,
            period_type=getattr(client.Client.PriceHistory.PeriodType, period_type.upper()),
            period=getattr(client.Client.PriceHistory.Period, f'{period_type.upper()}_{period}'),
            frequency_type=getattr(client.Client.PriceHistory.FrequencyType, frequency_type.upper()),
            frequency=getattr(client.Client.PriceHistory.Frequency, f'EVERY_{frequency}_MINUTE' if frequency_type == 'minute' else 'DAILY')
        )

        assert response.status_code == 200, f"Error fetching data: {response.status_code}"

        data = response.json()

        if 'candles' not in data:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': datetime.fromtimestamp(candle['datetime'] / 1000),
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume'],
            }
            for candle in data['candles']
        ])

        df = df.set_index('timestamp')

        if lookback is not None:
            df = df.tail(lookback)

        return df

    def get_quote(self, symbol: str) -> Quote:
        """Get real-time quote from TD Ameritrade"""
        if not self.is_connected:
            raise RuntimeError("Not connected to TD Ameritrade")

        response = self.client.get_quote(symbol)
        assert response.status_code == 200

        data = response.json()[symbol]

        return Quote(
            symbol=symbol,
            bid_price=float(data['bidPrice']),
            ask_price=float(data['askPrice']),
            bid_size=int(data['bidSize']),
            ask_size=int(data['askSize']),
            last_price=float(data['lastPrice']),
            last_size=int(data['lastSize']),
            volume=int(data['totalVolume']),
            timestamp=datetime.fromtimestamp(data['quoteTimeInLong'] / 1000),
            exchange=data.get('exchangeName')
        )

    def get_account(self) -> AccountInfo:
        """Get TD Ameritrade account information"""
        if not self.is_connected:
            raise RuntimeError("Not connected to TD Ameritrade")

        response = self.client.get_account(
            self.account_id,
            fields=client.Client.Account.Fields.POSITIONS
        )
        assert response.status_code == 200

        data = response.json()['securitiesAccount']

        return AccountInfo(
            account_id=self.account_id,
            buying_power=float(data['currentBalances']['buyingPower']),
            cash=float(data['currentBalances']['cashBalance']),
            portfolio_value=float(data['currentBalances']['liquidationValue']),
            equity=float(data['currentBalances']['equity']),
            currency='USD',
            last_updated=datetime.now()
        )

    def get_positions(self) -> List[Position]:
        """Get all positions from TD Ameritrade"""
        if not self.is_connected:
            raise RuntimeError("Not connected to TD Ameritrade")

        response = self.client.get_account(
            self.account_id,
            fields=client.Client.Account.Fields.POSITIONS
        )
        assert response.status_code == 200

        data = response.json()['securitiesAccount']

        if 'positions' not in data:
            return []

        positions = []
        for pos in data['positions']:
            instrument = pos['instrument']

            # Only handle equity positions
            if instrument['assetType'] == 'EQUITY':
                quantity = float(pos['longQuantity']) - float(pos['shortQuantity'])
                avg_price = float(pos['averagePrice'])
                current_price = float(pos['marketValue']) / abs(quantity) if quantity != 0 else 0

                positions.append(Position(
                    symbol=instrument['symbol'],
                    quantity=abs(quantity),
                    avg_entry_price=avg_price,
                    current_price=current_price,
                    market_value=float(pos['marketValue']),
                    unrealized_pnl=float(pos.get('currentDayProfitLoss', 0)),
                    unrealized_pnl_percent=float(pos.get('currentDayProfitLossPercentage', 0)),
                    side='long' if quantity > 0 else 'short'
                ))

        return positions

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
        """Place order with TD Ameritrade"""
        if not self.is_connected:
            raise RuntimeError("Not connected to TD Ameritrade")

        # Build order spec
        from tda.orders.equities import equity_buy_market, equity_sell_market
        from tda.orders.equities import equity_buy_limit, equity_sell_limit

        if order_type == OrderType.MARKET:
            if side == OrderSide.BUY:
                order_spec = equity_buy_market(symbol, int(quantity))
            else:
                order_spec = equity_sell_market(symbol, int(quantity))

        elif order_type == OrderType.LIMIT:
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")

            if side == OrderSide.BUY:
                order_spec = equity_buy_limit(symbol, int(quantity), limit_price)
            else:
                order_spec = equity_sell_limit(symbol, int(quantity), limit_price)

        else:
            raise ValueError(f"Order type {order_type} not fully implemented for TDA")

        # Set duration
        if time_in_force.lower() == 'gtc':
            order_spec.set_duration(client.Client.Order.Duration.GOOD_TILL_CANCEL)
        else:
            order_spec.set_duration(client.Client.Order.Duration.DAY)

        # Place order
        response = self.client.place_order(self.account_id, order_spec.build())

        if response.status_code in [200, 201]:
            # Extract order ID from response headers
            order_id = response.headers.get('Location', '').split('/')[-1]

            return Order(
                order_id=order_id,
                symbol=symbol,
                quantity=float(quantity),
                side=side,
                order_type=order_type,
                status=OrderStatus.SUBMITTED,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                limit_price=limit_price,
                stop_price=stop_price,
                submitted_at=datetime.now()
            )
        else:
            raise RuntimeError(f"Failed to place order: {response.status_code} - {response.text}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.is_connected:
            raise RuntimeError("Not connected to TD Ameritrade")

        try:
            response = self.client.cancel_order(order_id, self.account_id)
            return response.status_code == 200
        except Exception as e:
            print(f"✗ Failed to cancel order {order_id}: {str(e)}")
            return False

    def get_order(self, order_id: str) -> Order:
        """Get order details"""
        if not self.is_connected:
            raise RuntimeError("Not connected to TD Ameritrade")

        response = self.client.get_order(order_id, self.account_id)
        assert response.status_code == 200

        order_data = response.json()
        return self._convert_tda_order(order_data)

    def _convert_tda_order(self, order_data: Dict) -> Order:
        """Convert TDA order to our Order format"""
        status_mapping = {
            'AWAITING_PARENT_ORDER': OrderStatus.PENDING,
            'AWAITING_CONDITION': OrderStatus.PENDING,
            'AWAITING_MANUAL_REVIEW': OrderStatus.PENDING,
            'ACCEPTED': OrderStatus.SUBMITTED,
            'AWAITING_UR_OUT': OrderStatus.SUBMITTED,
            'PENDING_ACTIVATION': OrderStatus.SUBMITTED,
            'QUEUED': OrderStatus.SUBMITTED,
            'WORKING': OrderStatus.SUBMITTED,
            'REJECTED': OrderStatus.REJECTED,
            'PENDING_CANCEL': OrderStatus.CANCELED,
            'CANCELED': OrderStatus.CANCELED,
            'PENDING_REPLACE': OrderStatus.SUBMITTED,
            'REPLACED': OrderStatus.SUBMITTED,
            'FILLED': OrderStatus.FILLED,
            'EXPIRED': OrderStatus.EXPIRED,
        }

        type_mapping = {
            'MARKET': OrderType.MARKET,
            'LIMIT': OrderType.LIMIT,
            'STOP': OrderType.STOP,
            'STOP_LIMIT': OrderType.STOP_LIMIT,
        }

        leg = order_data['orderLegCollection'][0]
        instrument = leg['instrument']

        return Order(
            order_id=str(order_data['orderId']),
            symbol=instrument['symbol'],
            quantity=float(leg['quantity']),
            side=OrderSide.BUY if leg['instruction'] == 'BUY' else OrderSide.SELL,
            order_type=type_mapping.get(order_data['orderType'], OrderType.MARKET),
            status=status_mapping.get(order_data['status'], OrderStatus.PENDING),
            filled_quantity=float(order_data.get('filledQuantity', 0)),
            avg_fill_price=0.0,  # TDA doesn't provide this directly
            limit_price=float(order_data['price']) if 'price' in order_data else None,
            stop_price=float(order_data['stopPrice']) if 'stopPrice' in order_data else None,
            submitted_at=datetime.fromisoformat(order_data['enteredTime'].replace('Z', '+00:00'))
        )

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get list of orders"""
        if not self.is_connected:
            raise RuntimeError("Not connected to TD Ameritrade")

        response = self.client.get_orders_by_path(self.account_id)
        assert response.status_code == 200

        orders_data = response.json()
        orders = [self._convert_tda_order(order_data) for order_data in orders_data]

        # Filter by status if specified
        if status:
            orders = [o for o in orders if o.status == status]

        return orders[:limit]

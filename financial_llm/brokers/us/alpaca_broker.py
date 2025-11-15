"""
Alpaca Broker Integration

Supports both paper and live trading through Alpaca's REST API.
Provides market data, order execution, and account management.

Documentation: https://alpaca.markets/docs/
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
        StopLimitOrderRequest, TrailingStopOrderRequest
    )
    from alpaca.trading.enums import (
        OrderSide as AlpacaOrderSide,
        TimeInForce,
        OrderType as AlpacaOrderType
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from ..base import (
    BaseBroker, BrokerCapability, OrderSide, OrderType, OrderStatus,
    Order, Position, AccountInfo, Quote
)


class AlpacaBroker(BaseBroker):
    """
    Alpaca broker implementation.

    Supports:
    - US stocks and ETFs
    - Market data (IEX and SIP feeds)
    - Paper and live trading
    - Market, limit, stop, and trailing stop orders
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Alpaca broker.

        Required config:
            - api_key: Alpaca API key
            - api_secret: Alpaca API secret
            - paper_trading: True for paper trading, False for live
            - base_url: API endpoint (optional)
            - data_feed: 'iex' or 'sip' (optional, default 'iex')
        """
        super().__init__(config)

        if not ALPACA_AVAILABLE:
            raise ImportError(
                "Alpaca SDK not installed. Install with: "
                "pip install alpaca-py"
            )

        self.api_key = config['api_key']
        self.api_secret = config['api_secret']
        self.data_feed = config.get('data_feed', 'iex')

        # Set base URL based on paper/live trading
        if self.is_paper_trading:
            self.base_url = config.get('base_url', 'https://paper-api.alpaca.markets')
        else:
            self.base_url = config.get('base_url', 'https://api.alpaca.markets')

        self.trading_client: Optional[TradingClient] = None
        self.data_client: Optional[StockHistoricalDataClient] = None

    def connect(self) -> bool:
        """Establish connection to Alpaca"""
        try:
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.is_paper_trading,
                url_override=self.base_url
            )

            # Initialize data client
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret
            )

            # Validate connection by fetching account
            self.trading_client.get_account()

            self.is_connected = True
            print(f"✓ Connected to Alpaca ({'paper' if self.is_paper_trading else 'live'} trading)")
            return True

        except Exception as e:
            print(f"✗ Failed to connect to Alpaca: {str(e)}")
            self.is_connected = False
            return False

    def disconnect(self) -> bool:
        """Close connection to Alpaca"""
        self.trading_client = None
        self.data_client = None
        self.is_connected = False
        print("✓ Disconnected from Alpaca")
        return True

    def get_capabilities(self) -> List[BrokerCapability]:
        """Get Alpaca capabilities"""
        return [
            BrokerCapability.MARKET_DATA,
            BrokerCapability.HISTORICAL_DATA,
            BrokerCapability.REAL_TIME_QUOTES,
            BrokerCapability.ORDER_EXECUTION,
            BrokerCapability.PAPER_TRADING,
            BrokerCapability.STOP_LOSS,
            BrokerCapability.TAKE_PROFIT,
            BrokerCapability.BRACKET_ORDERS,
        ]

    def _convert_timeframe(self, timeframe: str) -> TimeFrame:
        """Convert timeframe string to Alpaca TimeFrame"""
        mapping = {
            '1Min': TimeFrame(1, TimeFrameUnit.Minute),
            '5Min': TimeFrame(5, TimeFrameUnit.Minute),
            '15Min': TimeFrame(15, TimeFrameUnit.Minute),
            '1Hour': TimeFrame(1, TimeFrameUnit.Hour),
            '1Day': TimeFrame(1, TimeFrameUnit.Day),
            '1Week': TimeFrame(1, TimeFrameUnit.Week),
            '1Month': TimeFrame(1, TimeFrameUnit.Month),
        }
        if timeframe not in mapping:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. "
                f"Supported: {list(mapping.keys())}"
            )
        return mapping[timeframe]

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from Alpaca"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Alpaca")

        # Determine date range
        if end_date is None:
            end_date = datetime.now()

        if start_date is None and lookback is not None:
            # Estimate start date based on lookback and timeframe
            if '1Min' in timeframe:
                start_date = end_date - timedelta(days=lookback // 390)  # ~390 mins per trading day
            elif '5Min' in timeframe:
                start_date = end_date - timedelta(days=lookback // 78)
            elif '1Hour' in timeframe:
                start_date = end_date - timedelta(days=lookback // 6.5)
            elif '1Day' in timeframe:
                start_date = end_date - timedelta(days=lookback)
            else:
                start_date = end_date - timedelta(days=lookback)

        if start_date is None:
            start_date = end_date - timedelta(days=365)  # Default to 1 year

        # Create request
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self._convert_timeframe(timeframe),
            start=start_date,
            end=end_date
        )

        # Fetch data
        bars = self.data_client.get_stock_bars(request_params)

        # Convert to DataFrame
        if symbol in bars.data:
            df = pd.DataFrame([
                {
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume),
                }
                for bar in bars.data[symbol]
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            if lookback is not None:
                df = df.tail(lookback)

            return df
        else:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def get_quote(self, symbol: str) -> Quote:
        """Get real-time quote from Alpaca"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Alpaca")

        request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self.data_client.get_stock_latest_quote(request_params)

        if symbol in quotes:
            q = quotes[symbol]
            return Quote(
                symbol=symbol,
                bid_price=float(q.bid_price),
                ask_price=float(q.ask_price),
                bid_size=int(q.bid_size),
                ask_size=int(q.ask_size),
                last_price=float((q.bid_price + q.ask_price) / 2),
                last_size=0,
                volume=0,
                timestamp=q.timestamp,
                exchange=q.bid_exchange if hasattr(q, 'bid_exchange') else None
            )
        else:
            raise ValueError(f"No quote available for {symbol}")

    def get_account(self) -> AccountInfo:
        """Get Alpaca account information"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Alpaca")

        account = self.trading_client.get_account()

        return AccountInfo(
            account_id=account.account_number,
            buying_power=float(account.buying_power),
            cash=float(account.cash),
            portfolio_value=float(account.portfolio_value),
            equity=float(account.equity),
            currency='USD',
            last_updated=datetime.now()
        )

    def get_positions(self) -> List[Position]:
        """Get all positions from Alpaca"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Alpaca")

        positions = self.trading_client.get_all_positions()

        return [
            Position(
                symbol=pos.symbol,
                quantity=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_percent=float(pos.unrealized_plpc) * 100,
                side='long' if float(pos.qty) > 0 else 'short',
                exchange=pos.exchange if hasattr(pos, 'exchange') else None
            )
            for pos in positions
        ]

    def _convert_order_side(self, side: OrderSide) -> AlpacaOrderSide:
        """Convert OrderSide to Alpaca OrderSide"""
        return AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL

    def _convert_time_in_force(self, tif: str) -> TimeInForce:
        """Convert time in force string to Alpaca TimeInForce"""
        mapping = {
            'day': TimeInForce.DAY,
            'gtc': TimeInForce.GTC,
            'ioc': TimeInForce.IOC,
            'fok': TimeInForce.FOK,
        }
        return mapping.get(tif.lower(), TimeInForce.DAY)

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
        """Place order with Alpaca"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Alpaca")

        alpaca_side = self._convert_order_side(side)
        tif = self._convert_time_in_force(time_in_force)

        # Create order request based on type
        if order_type == OrderType.MARKET:
            request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                time_in_force=tif
            )
        elif order_type == OrderType.LIMIT:
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                time_in_force=tif,
                limit_price=limit_price
            )
        elif order_type == OrderType.STOP:
            if stop_price is None:
                raise ValueError("stop_price required for stop orders")
            request = StopOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                time_in_force=tif,
                stop_price=stop_price
            )
        elif order_type == OrderType.STOP_LIMIT:
            if stop_price is None or limit_price is None:
                raise ValueError("stop_price and limit_price required for stop-limit orders")
            request = StopLimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                time_in_force=tif,
                stop_price=stop_price,
                limit_price=limit_price
            )
        elif order_type == OrderType.TRAILING_STOP:
            trail_percent = kwargs.get('trail_percent')
            trail_price = kwargs.get('trail_price')
            if trail_percent is None and trail_price is None:
                raise ValueError("trail_percent or trail_price required for trailing stop")
            request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                time_in_force=tif,
                trail_percent=trail_percent,
                trail_price=trail_price
            )
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

        # Submit order
        alpaca_order = self.trading_client.submit_order(request)

        # Convert to our Order format
        return self._convert_alpaca_order(alpaca_order)

    def _convert_alpaca_order(self, alpaca_order) -> Order:
        """Convert Alpaca order to our Order format"""
        # Map Alpaca status to our OrderStatus
        status_mapping = {
            'pending_new': OrderStatus.PENDING,
            'accepted': OrderStatus.SUBMITTED,
            'new': OrderStatus.SUBMITTED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.CANCELED,
            'canceled': OrderStatus.CANCELED,
            'expired': OrderStatus.EXPIRED,
            'replaced': OrderStatus.CANCELED,
            'pending_cancel': OrderStatus.CANCELED,
            'pending_replace': OrderStatus.SUBMITTED,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.REJECTED,
            'calculated': OrderStatus.SUBMITTED,
        }

        # Map order type
        type_mapping = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop': OrderType.STOP,
            'stop_limit': OrderType.STOP_LIMIT,
            'trailing_stop': OrderType.TRAILING_STOP,
        }

        return Order(
            order_id=str(alpaca_order.id),
            symbol=alpaca_order.symbol,
            quantity=float(alpaca_order.qty),
            side=OrderSide.BUY if alpaca_order.side == 'buy' else OrderSide.SELL,
            order_type=type_mapping.get(alpaca_order.order_type, OrderType.MARKET),
            status=status_mapping.get(alpaca_order.status, OrderStatus.PENDING),
            filled_quantity=float(alpaca_order.filled_qty or 0),
            avg_fill_price=float(alpaca_order.filled_avg_price or 0),
            limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
            stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
            submitted_at=alpaca_order.submitted_at,
            filled_at=alpaca_order.filled_at,
            canceled_at=alpaca_order.canceled_at,
            client_order_id=alpaca_order.client_order_id
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Alpaca")

        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            print(f"✗ Failed to cancel order {order_id}: {str(e)}")
            return False

    def get_order(self, order_id: str) -> Order:
        """Get order details"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Alpaca")

        alpaca_order = self.trading_client.get_order_by_id(order_id)
        return self._convert_alpaca_order(alpaca_order)

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get list of orders"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Alpaca")

        # Alpaca uses different status values
        alpaca_status = None
        if status == OrderStatus.SUBMITTED:
            alpaca_status = 'open'
        elif status == OrderStatus.FILLED:
            alpaca_status = 'closed'
        elif status == OrderStatus.CANCELED:
            alpaca_status = 'canceled'

        alpaca_orders = self.trading_client.get_orders(
            status=alpaca_status,
            limit=limit
        )

        return [self._convert_alpaca_order(order) for order in alpaca_orders]

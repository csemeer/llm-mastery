"""
Interactive Brokers (IBKR) Integration

Connects to Interactive Brokers via IB Gateway or TWS using ib_insync library.
Supports stocks, options, futures, and international markets.

Requirements:
    - IB Gateway or TWS running
    - ib_insync library installed

Documentation: https://interactivebrokers.github.io/
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import time

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, Order as IBOrder
    from ib_insync import util
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False

from ..base import (
    BaseBroker, BrokerCapability, OrderSide, OrderType, OrderStatus,
    Order, Position, AccountInfo, Quote
)


class InteractiveBrokersBroker(BaseBroker):
    """
    Interactive Brokers implementation.

    Supports:
    - Global markets (US, Europe, Asia, etc.)
    - Stocks, options, futures, forex, bonds
    - Real-time and historical data
    - Complex order types
    - Paper and live trading
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize IBKR broker.

        Required config:
            - host: IB Gateway/TWS host (default '127.0.0.1')
            - port: IB Gateway/TWS port (7497 paper, 7496 live, 4001/4002 Gateway)
            - client_id: Unique client ID (default 1)
            - paper_trading: True for paper account
        """
        super().__init__(config)

        if not IBKR_AVAILABLE:
            raise ImportError(
                "ib_insync not installed. Install with: "
                "pip install ib_insync"
            )

        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 7497 if self.is_paper_trading else 7496)
        self.client_id = config.get('client_id', 1)
        self.readonly = config.get('readonly', False)

        self.ib: Optional[IB] = None
        self.account_id: Optional[str] = None

    def connect(self) -> bool:
        """Connect to IB Gateway/TWS"""
        try:
            self.ib = IB()
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                readonly=self.readonly
            )

            # Get account ID
            accounts = self.ib.managedAccounts()
            if accounts:
                self.account_id = accounts[0]
                self.is_connected = True
                print(f"✓ Connected to Interactive Brokers (account: {self.account_id})")
                return True
            else:
                print("✗ No accounts found")
                return False

        except Exception as e:
            print(f"✗ Failed to connect to IBKR: {str(e)}")
            print("  Make sure IB Gateway or TWS is running")
            self.is_connected = False
            return False

    def disconnect(self) -> bool:
        """Disconnect from IBKR"""
        if self.ib:
            self.ib.disconnect()
        self.is_connected = False
        print("✓ Disconnected from Interactive Brokers")
        return True

    def get_capabilities(self) -> List[BrokerCapability]:
        """Get IBKR capabilities"""
        return [
            BrokerCapability.MARKET_DATA,
            BrokerCapability.HISTORICAL_DATA,
            BrokerCapability.REAL_TIME_QUOTES,
            BrokerCapability.REAL_TIME_BARS,
            BrokerCapability.ORDER_EXECUTION,
            BrokerCapability.PAPER_TRADING,
            BrokerCapability.OPTIONS_TRADING,
            BrokerCapability.FUTURES_TRADING,
            BrokerCapability.MARGIN_TRADING,
            BrokerCapability.BRACKET_ORDERS,
            BrokerCapability.STOP_LOSS,
            BrokerCapability.TAKE_PROFIT,
        ]

    def _create_contract(self, symbol: str, exchange: str = 'SMART') -> Stock:
        """Create stock contract"""
        return Stock(symbol, exchange, 'USD')

    def _convert_timeframe(self, timeframe: str) -> tuple:
        """Convert timeframe to IBKR format (barSize, duration)"""
        mapping = {
            '1Min': ('1 min', '1 D'),
            '5Min': ('5 mins', '5 D'),
            '15Min': ('15 mins', '1 M'),
            '1Hour': ('1 hour', '1 M'),
            '1Day': ('1 day', '1 Y'),
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
        """Fetch historical data from IBKR"""
        if not self.is_connected:
            raise RuntimeError("Not connected to IBKR")

        contract = self._create_contract(symbol)
        bar_size, duration = self._convert_timeframe(timeframe)

        # Request historical data
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_date or datetime.now(),
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True,  # Regular trading hours only
            formatDate=1
        )

        if not bars:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert to DataFrame
        df = util.df(bars)
        df = df.rename(columns={'date': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df[['open', 'high', 'low', 'close', 'volume']]

        if lookback is not None:
            df = df.tail(lookback)

        return df

    def get_quote(self, symbol: str) -> Quote:
        """Get real-time quote from IBKR"""
        if not self.is_connected:
            raise RuntimeError("Not connected to IBKR")

        contract = self._create_contract(symbol)

        # Request market data
        ticker = self.ib.reqMktData(contract, '', False, False)

        # Wait for data
        self.ib.sleep(1)

        if ticker.bid and ticker.ask:
            quote = Quote(
                symbol=symbol,
                bid_price=float(ticker.bid),
                ask_price=float(ticker.ask),
                bid_size=int(ticker.bidSize) if ticker.bidSize else 0,
                ask_size=int(ticker.askSize) if ticker.askSize else 0,
                last_price=float(ticker.last) if ticker.last else float((ticker.bid + ticker.ask) / 2),
                last_size=int(ticker.lastSize) if ticker.lastSize else 0,
                volume=int(ticker.volume) if ticker.volume else 0,
                timestamp=datetime.now()
            )

            # Cancel market data subscription
            self.ib.cancelMktData(contract)

            return quote
        else:
            raise ValueError(f"No quote available for {symbol}")

    def get_account(self) -> AccountInfo:
        """Get IBKR account information"""
        if not self.is_connected:
            raise RuntimeError("Not connected to IBKR")

        # Request account summary
        account_values = self.ib.accountSummary(self.account_id)

        # Extract values
        values = {item.tag: float(item.value) for item in account_values if item.value}

        return AccountInfo(
            account_id=self.account_id,
            buying_power=values.get('BuyingPower', 0.0),
            cash=values.get('TotalCashValue', 0.0),
            portfolio_value=values.get('NetLiquidation', 0.0),
            equity=values.get('NetLiquidation', 0.0),
            currency=values.get('Currency', 'USD'),
            margin_used=values.get('MaintMarginReq', 0.0),
            maintenance_margin=values.get('MaintMarginReq', 0.0),
            last_updated=datetime.now()
        )

    def get_positions(self) -> List[Position]:
        """Get all positions from IBKR"""
        if not self.is_connected:
            raise RuntimeError("Not connected to IBKR")

        positions = self.ib.positions()

        return [
            Position(
                symbol=pos.contract.symbol,
                quantity=float(pos.position),
                avg_entry_price=float(pos.avgCost / abs(pos.position)) if pos.position != 0 else 0,
                current_price=float(pos.marketPrice) if pos.marketPrice else 0,
                market_value=float(pos.marketValue) if pos.marketValue else 0,
                unrealized_pnl=float(pos.unrealizedPNL) if pos.unrealizedPNL else 0,
                unrealized_pnl_percent=0,  # Calculate manually
                side='long' if pos.position > 0 else 'short',
                exchange=pos.contract.exchange if hasattr(pos.contract, 'exchange') else None
            )
            for pos in positions
        ]

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
        """Place order with IBKR"""
        if not self.is_connected:
            raise RuntimeError("Not connected to IBKR")

        contract = self._create_contract(symbol)
        action = 'BUY' if side == OrderSide.BUY else 'SELL'

        # Create order based on type
        if order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, quantity)
        elif order_type == OrderType.LIMIT:
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            ib_order = LimitOrder(action, quantity, limit_price)
        elif order_type == OrderType.STOP:
            if stop_price is None:
                raise ValueError("stop_price required for stop orders")
            ib_order = StopOrder(action, quantity, stop_price)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

        # Set time in force
        tif_mapping = {
            'day': 'DAY',
            'gtc': 'GTC',
            'ioc': 'IOC',
            'fok': 'FOK'
        }
        ib_order.tif = tif_mapping.get(time_in_force.lower(), 'DAY')

        # Place order
        trade = self.ib.placeOrder(contract, ib_order)

        # Wait for order to be submitted
        while not trade.isDone():
            self.ib.sleep(0.1)
            if trade.orderStatus.status in ['Submitted', 'PreSubmitted', 'Filled']:
                break

        # Convert to our Order format
        return self._convert_ib_order(trade)

    def _convert_ib_order(self, trade) -> Order:
        """Convert IB order to our Order format"""
        status_mapping = {
            'PendingSubmit': OrderStatus.PENDING,
            'PreSubmitted': OrderStatus.SUBMITTED,
            'Submitted': OrderStatus.SUBMITTED,
            'PartiallyFilled': OrderStatus.PARTIALLY_FILLED,
            'Filled': OrderStatus.FILLED,
            'Cancelled': OrderStatus.CANCELED,
            'Inactive': OrderStatus.REJECTED,
        }

        type_mapping = {
            'MKT': OrderType.MARKET,
            'LMT': OrderType.LIMIT,
            'STP': OrderType.STOP,
            'STP LMT': OrderType.STOP_LIMIT,
        }

        order_status = trade.orderStatus

        return Order(
            order_id=str(trade.order.orderId),
            symbol=trade.contract.symbol,
            quantity=float(trade.order.totalQuantity),
            side=OrderSide.BUY if trade.order.action == 'BUY' else OrderSide.SELL,
            order_type=type_mapping.get(trade.order.orderType, OrderType.MARKET),
            status=status_mapping.get(order_status.status, OrderStatus.PENDING),
            filled_quantity=float(order_status.filled),
            avg_fill_price=float(order_status.avgFillPrice) if order_status.avgFillPrice else 0,
            limit_price=float(trade.order.lmtPrice) if trade.order.lmtPrice else None,
            stop_price=float(trade.order.auxPrice) if trade.order.auxPrice else None,
            submitted_at=datetime.now(),
            commission=float(order_status.commission) if order_status.commission else 0
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.is_connected:
            raise RuntimeError("Not connected to IBKR")

        try:
            self.ib.cancelOrder(IBOrder(orderId=int(order_id)))
            return True
        except Exception as e:
            print(f"✗ Failed to cancel order {order_id}: {str(e)}")
            return False

    def get_order(self, order_id: str) -> Order:
        """Get order details"""
        if not self.is_connected:
            raise RuntimeError("Not connected to IBKR")

        trades = self.ib.trades()
        for trade in trades:
            if str(trade.order.orderId) == order_id:
                return self._convert_ib_order(trade)

        raise ValueError(f"Order {order_id} not found")

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get list of orders"""
        if not self.is_connected:
            raise RuntimeError("Not connected to IBKR")

        trades = self.ib.trades()
        orders = [self._convert_ib_order(trade) for trade in trades]

        # Filter by status if specified
        if status:
            orders = [o for o in orders if o.status == status]

        return orders[:limit]

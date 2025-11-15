"""
Order Manager

Unified order execution and management across multiple brokers.
Provides order tracking, position management, and risk controls.
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import threading
import time

from .base import (
    BaseBroker, Order, OrderSide, OrderType, OrderStatus,
    Position, AccountInfo
)


@dataclass
class OrderRecord:
    """Extended order record with tracking information"""
    order: Order
    broker_name: str
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    notes: str = ""


class OrderManager:
    """
    Unified order manager for multiple brokers.

    Features:
    - Order execution across multiple brokers
    - Order tracking and history
    - Position aggregation
    - Risk controls (max position size, daily loss limit, etc.)
    - Auto-refresh of order status
    """

    def __init__(
        self,
        brokers: Dict[str, BaseBroker],
        default_broker: Optional[str] = None,
        enable_risk_controls: bool = True
    ):
        """
        Initialize order manager.

        Args:
            brokers: Dictionary mapping broker name to broker instance
            default_broker: Default broker for order execution
            enable_risk_controls: Whether to enable risk controls
        """
        self.brokers = brokers
        self.default_broker = default_broker or list(brokers.keys())[0]
        self.enable_risk_controls = enable_risk_controls

        # Order tracking
        self.orders: Dict[str, OrderRecord] = {}
        self.order_lock = threading.Lock()

        # Risk controls
        self.max_position_size: Dict[str, float] = {}  # symbol -> max size
        self.max_order_value: float = float('inf')
        self.daily_loss_limit: float = float('inf')
        self.daily_pnl: float = 0.0

        # Position tracking (aggregated across all brokers)
        self.positions: Dict[str, Position] = {}

    def place_order(
        self,
        symbol: str,
        quantity: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        broker_name: Optional[str] = None,
        time_in_force: str = "day",
        **kwargs
    ) -> OrderRecord:
        """
        Place an order through specified broker.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: OrderSide.BUY or OrderSide.SELL
            order_type: Type of order
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            broker_name: Broker to use (uses default if None)
            time_in_force: Time in force
            **kwargs: Additional broker-specific parameters

        Returns:
            OrderRecord with order details

        Raises:
            ValueError: If risk controls prevent order
            RuntimeError: If order placement fails
        """
        broker_name = broker_name or self.default_broker

        if broker_name not in self.brokers:
            raise ValueError(f"Broker '{broker_name}' not found")

        broker = self.brokers[broker_name]

        # Apply risk controls
        if self.enable_risk_controls:
            self._check_risk_controls(symbol, quantity, side, limit_price)

        # Place order
        order = broker.place_order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            **kwargs
        )

        # Track order
        order_record = OrderRecord(
            order=order,
            broker_name=broker_name,
            notes=kwargs.get('notes', '')
        )

        with self.order_lock:
            self.orders[order.order_id] = order_record

        print(f"✓ Order placed: {order.order_id} ({broker_name})")

        return order_record

    def cancel_order(
        self,
        order_id: str,
        broker_name: Optional[str] = None
    ) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            broker_name: Broker name (auto-detected if None)

        Returns:
            True if cancellation successful
        """
        # Find order record
        with self.order_lock:
            if order_id not in self.orders:
                raise ValueError(f"Order {order_id} not found")

            order_record = self.orders[order_id]
            broker_name = broker_name or order_record.broker_name

        broker = self.brokers[broker_name]

        # Cancel order
        success = broker.cancel_order(order_id)

        if success:
            # Update order status
            with self.order_lock:
                order_record.order.status = OrderStatus.CANCELED
                order_record.last_updated = datetime.now()

        return success

    def get_order(
        self,
        order_id: str,
        refresh: bool = True
    ) -> OrderRecord:
        """
        Get order details.

        Args:
            order_id: Order ID
            refresh: Whether to refresh from broker

        Returns:
            OrderRecord
        """
        with self.order_lock:
            if order_id not in self.orders:
                raise ValueError(f"Order {order_id} not found")

            order_record = self.orders[order_id]

        if refresh:
            broker = self.brokers[order_record.broker_name]
            try:
                updated_order = broker.get_order(order_id)
                with self.order_lock:
                    order_record.order = updated_order
                    order_record.last_updated = datetime.now()
            except Exception as e:
                print(f"⚠ Failed to refresh order {order_id}: {str(e)}")

        return order_record

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None,
        broker_name: Optional[str] = None
    ) -> List[OrderRecord]:
        """
        Get filtered list of orders.

        Args:
            status: Filter by order status
            symbol: Filter by symbol
            broker_name: Filter by broker

        Returns:
            List of OrderRecord
        """
        with self.order_lock:
            orders = list(self.orders.values())

        # Apply filters
        if status:
            orders = [o for o in orders if o.order.status == status]

        if symbol:
            orders = [o for o in orders if o.order.symbol == symbol]

        if broker_name:
            orders = [o for o in orders if o.broker_name == broker_name]

        return orders

    def get_positions(
        self,
        refresh: bool = True,
        aggregate: bool = True
    ) -> Dict[str, Position]:
        """
        Get positions across all brokers.

        Args:
            refresh: Whether to refresh from brokers
            aggregate: Whether to aggregate positions across brokers

        Returns:
            Dictionary mapping symbol to Position
        """
        if refresh:
            self._refresh_positions()

        if aggregate:
            return self.positions
        else:
            # Return positions per broker
            positions_by_broker = {}
            for name, broker in self.brokers.items():
                try:
                    positions_by_broker[name] = broker.get_positions()
                except Exception as e:
                    print(f"⚠ Failed to get positions from {name}: {str(e)}")

            return positions_by_broker

    def get_account_summary(self) -> Dict[str, AccountInfo]:
        """
        Get account info from all brokers.

        Returns:
            Dictionary mapping broker name to AccountInfo
        """
        accounts = {}

        for name, broker in self.brokers.items():
            try:
                accounts[name] = broker.get_account()
            except Exception as e:
                print(f"⚠ Failed to get account info from {name}: {str(e)}")

        return accounts

    def _refresh_positions(self):
        """Refresh positions from all brokers"""
        aggregated_positions: Dict[str, List[Position]] = {}

        for name, broker in self.brokers.items():
            try:
                positions = broker.get_positions()
                for pos in positions:
                    if pos.symbol not in aggregated_positions:
                        aggregated_positions[pos.symbol] = []
                    aggregated_positions[pos.symbol].append(pos)
            except Exception as e:
                print(f"⚠ Failed to refresh positions from {name}: {str(e)}")

        # Aggregate positions by symbol
        self.positions = {}
        for symbol, pos_list in aggregated_positions.items():
            if len(pos_list) == 1:
                self.positions[symbol] = pos_list[0]
            else:
                # Merge positions
                total_qty = sum(p.quantity for p in pos_list)
                total_value = sum(p.quantity * p.avg_entry_price for p in pos_list)
                avg_price = total_value / total_qty if total_qty != 0 else 0

                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=total_qty,
                    avg_entry_price=avg_price,
                    current_price=pos_list[0].current_price,
                    market_value=sum(p.market_value for p in pos_list),
                    unrealized_pnl=sum(p.unrealized_pnl for p in pos_list),
                    unrealized_pnl_percent=0,
                    side='long' if total_qty > 0 else 'short'
                )

    def _check_risk_controls(
        self,
        symbol: str,
        quantity: float,
        side: OrderSide,
        price: Optional[float]
    ):
        """Check if order passes risk controls"""

        # Check max position size
        if symbol in self.max_position_size:
            current_pos = self.positions.get(symbol)
            current_qty = current_pos.quantity if current_pos else 0

            new_qty = current_qty + quantity if side == OrderSide.BUY else current_qty - quantity

            if abs(new_qty) > self.max_position_size[symbol]:
                raise ValueError(
                    f"Order exceeds max position size for {symbol}: "
                    f"{abs(new_qty)} > {self.max_position_size[symbol]}"
                )

        # Check max order value
        if price:
            order_value = abs(quantity * price)
            if order_value > self.max_order_value:
                raise ValueError(
                    f"Order value exceeds limit: ${order_value:,.2f} > ${self.max_order_value:,.2f}"
                )

        # Check daily loss limit
        if self.daily_pnl < -self.daily_loss_limit:
            raise ValueError(
                f"Daily loss limit exceeded: ${self.daily_pnl:,.2f} < -${self.daily_loss_limit:,.2f}"
            )

    def set_max_position_size(self, symbol: str, max_size: float):
        """Set maximum position size for a symbol"""
        self.max_position_size[symbol] = max_size

    def set_max_order_value(self, max_value: float):
        """Set maximum order value"""
        self.max_order_value = max_value

    def set_daily_loss_limit(self, limit: float):
        """Set daily loss limit"""
        self.daily_loss_limit = limit

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[OrderRecord]:
        """
        Get order history with filters.

        Args:
            symbol: Filter by symbol
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of OrderRecord
        """
        with self.order_lock:
            orders = list(self.orders.values())

        # Apply filters
        if symbol:
            orders = [o for o in orders if o.order.symbol == symbol]

        if start_date:
            orders = [o for o in orders if o.created_at >= start_date]

        if end_date:
            orders = [o for o in orders if o.created_at <= end_date]

        # Sort by creation time
        orders.sort(key=lambda x: x.created_at, reverse=True)

        return orders

    def get_fills(self) -> List[OrderRecord]:
        """Get all filled orders"""
        return self.get_orders(status=OrderStatus.FILLED)

    def get_open_orders(self) -> List[OrderRecord]:
        """Get all open orders"""
        with self.order_lock:
            orders = []
            for record in self.orders.values():
                if record.order.is_pending:
                    orders.append(record)
        return orders

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Cancel only orders for this symbol (None for all)

        Returns:
            Number of orders canceled
        """
        open_orders = self.get_open_orders()

        if symbol:
            open_orders = [o for o in open_orders if o.order.symbol == symbol]

        canceled_count = 0
        for order_record in open_orders:
            try:
                if self.cancel_order(order_record.order.order_id):
                    canceled_count += 1
            except Exception as e:
                print(f"⚠ Failed to cancel {order_record.order.order_id}: {str(e)}")

        return canceled_count

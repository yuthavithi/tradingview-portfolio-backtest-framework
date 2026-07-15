"""
Portfolio Engine Module.

This module implements the core event-driven execution handlers, risk managers,
portfolio tracking, and the central SimulationEngine orchestrator.
"""

import logging
from typing import Dict, List, Optional, Callable

from portfolio.account import Account
from portfolio.events import (
    BaseEvent,
    EntryEvent,
    ExitEvent,
    OrderEvent,
    TradeEvent,
    MarginEvent,
    AccountEvent,
    PortfolioEvent,
    MarketDataEvent,
)
from portfolio.queue import EventQueue

logger = logging.getLogger("portfolio.engine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class RiskManager:
    """
    Evaluates OrderEvents before they are sent to the ExecutionHandler.
    Monitors account health and issues margin events.
    """

    def __init__(self, max_leverage: float = 10.0, max_drawdown_limit: float = 0.5) -> None:
        self.max_leverage = max_leverage
        self.max_drawdown_limit = max_drawdown_limit

    def validate_order(self, order: OrderEvent, account: Account) -> bool:
        """
        Validates if an order satisfies risk limits.

        Args:
            order: The OrderEvent to validate.
            account: The current Account state.

        Returns:
            bool: True if order is valid and allowed, False otherwise.
        """
        # Ensure we have enough free margin to cover the order size
        # Estimate order value
        order_value = order.quantity * order.price if order.price > 0 else 0.0
        required_margin = order_value * account.margin_requirement

        if required_margin > account.free_margin:
            logger.warning(
                f"[RiskManager] Order REJECTED for {order.ticker}: "
                f"Required margin ${required_margin:.2f} exceeds Free Margin ${account.free_margin:.2f}."
            )
            return False

        # Calculate leverage if this order is filled
        estimated_used_margin = account.used_margin + required_margin
        estimated_leverage = (
            (estimated_used_margin / account.margin_requirement) / account.equity
            if account.equity > 0
            else float("inf")
        )

        if estimated_leverage > self.max_leverage:
            logger.warning(
                f"[RiskManager] Order REJECTED for {order.ticker}: "
                f"Estimated leverage {estimated_leverage:.1f}x exceeds limit {self.max_leverage:.1f}x."
            )
            return False

        return True

    def check_margin(self, account: Account, timestamp) -> Optional[MarginEvent]:
        """
        Checks if the account has hit a margin call or liquidation condition.

        Args:
            account: The current Account state.
            timestamp: The current simulation timestamp.

        Returns:
            Optional[MarginEvent]: A margin event if triggered, otherwise None.
        """
        if account.is_margin_called:
            # Determine status based on equity severity
            status = "LIQUIDATED" if account.equity <= 0 else "CALL"
            logger.warning(
                f"[RiskManager] MARGIN EVENT [{status}] at {timestamp}: "
                f"Equity = ${account.equity:.2f}, Used Margin = ${account.used_margin:.2f}"
            )
            return MarginEvent(
                timestamp=timestamp,
                equity=account.equity,
                used_margin=account.used_margin,
                margin_level=account.margin_level,
                status=status,
            )
        elif account.margin_level < 150.0:  # 150% warning threshold
            logger.info(
                f"[RiskManager] Margin Warning at {timestamp}: "
                f"Margin level is {account.margin_level:.1f}%."
            )
            return MarginEvent(
                timestamp=timestamp,
                equity=account.equity,
                used_margin=account.used_margin,
                margin_level=account.margin_level,
                status="WARNING",
            )
        return None


class ExecutionHandler:
    """
    Simulates the broker execution environment.
    Receives OrderEvents and matches them against market prices to output TradeEvents.
    """

    def __init__(self, queue: EventQueue, commission_rate: float = 0.0005) -> None:
        self.queue = queue
        self.commission_rate = commission_rate
        self.current_prices: Dict[str, float] = {}
        self.pending_orders: List[OrderEvent] = []
        self._fill_id_seq = 0

    def _next_fill_id(self) -> str:
        self._fill_id_seq += 1
        return f"FILL_{self._fill_id_seq:05d}"

    def update_price(self, ticker: str, price: float) -> None:
        """Updates the internal market price for a ticker."""
        self.current_prices[ticker] = price

    def process_order(self, order: OrderEvent) -> Optional[TradeEvent]:
        """
        Processes a newly received OrderEvent.

        Args:
            order: The OrderEvent.

        Returns:
            Optional[TradeEvent]: TradeEvent if filled immediately, None if queued.
        """
        price = self.current_prices.get(order.ticker, order.price)

        if order.order_type.upper() == "MARKET":
            if order.ticker not in self.current_prices:
                logger.warning(
                    f"[ExecutionHandler] No market price for {order.ticker}, "
                    f"using order reference price {order.price:.2f}"
                )
                price = order.price

            commission = order.quantity * price * self.commission_rate
            trade = TradeEvent(
                timestamp=order.timestamp,
                fill_id=self._next_fill_id(),
                order_id=order.order_id,
                ticker=order.ticker,
                action=order.action,
                quantity=order.quantity,
                price=price,
                commission=commission,
            )
            order.status = "FILLED"
            logger.info(
                f"[ExecutionHandler] MARKET Order filled: {order.action} {order.quantity} "
                f"{order.ticker} @ {price:.2f} (Comm: ${commission:.2f})"
            )
            return trade

        # For LIMIT or STOP orders, add to pending list to match on next market updates
        logger.info(
            f"[ExecutionHandler] Order queued: {order.order_type} {order.action} {order.quantity} "
            f"{order.ticker} @ {order.price:.2f}"
        )
        self.pending_orders.append(order)
        return None

    def match_pending_orders(self, ticker: str, current_price: float, timestamp) -> List[TradeEvent]:
        """
        Matches pending LIMIT or STOP orders against a new price update.

        Args:
            ticker: The updated ticker.
            current_price: The new market price.
            timestamp: The update timestamp.

        Returns:
            List[TradeEvent]: Generated TradeEvents for filled orders.
        """
        fills: List[TradeEvent] = []
        still_pending: List[OrderEvent] = []

        for order in self.pending_orders:
            if order.ticker != ticker:
                still_pending.append(order)
                continue

            triggered = False
            fill_price = order.price

            order_type_upper = order.order_type.upper()
            action_upper = order.action.upper()

            if order_type_upper == "LIMIT":
                if action_upper == "BUY" and current_price <= order.price:
                    triggered = True
                    # Fill at the limit price or current_price if better
                    fill_price = min(order.price, current_price)
                elif action_upper == "SELL" and current_price >= order.price:
                    triggered = True
                    fill_price = max(order.price, current_price)

            elif order_type_upper == "STOP":
                if action_upper == "BUY" and current_price >= order.price:
                    triggered = True
                    fill_price = max(order.price, current_price)
                elif action_upper == "SELL" and current_price <= order.price:
                    triggered = True
                    fill_price = min(order.price, current_price)

            if triggered:
                commission = order.quantity * fill_price * self.commission_rate
                trade = TradeEvent(
                    timestamp=timestamp,
                    fill_id=self._next_fill_id(),
                    order_id=order.order_id,
                    ticker=order.ticker,
                    action=order.action,
                    quantity=order.quantity,
                    price=fill_price,
                    commission=commission,
                )
                order.status = "FILLED"
                fills.append(trade)
                logger.info(
                    f"[ExecutionHandler] {order_type_upper} Order filled: {order.action} {order.quantity} "
                    f"{order.ticker} @ {fill_price:.2f} (Comm: ${commission:.2f})"
                )
            else:
                still_pending.append(order)

        self.pending_orders = still_pending
        return fills


class Portfolio:
    """
    Manages active holdings, translates signal events into orders,
    and publishes PortfolioEvents.
    """

    def __init__(self, queue: EventQueue, account: Account) -> None:
        self.queue = queue
        self.account = account
        self._order_id_seq = 0

    def _next_order_id(self) -> str:
        self._order_id_seq += 1
        return f"ORD_{self._order_id_seq:05d}"

    def process_entry(self, entry: EntryEvent) -> Optional[OrderEvent]:
        """
        Translates an EntryEvent signal into an OrderEvent.
        """
        # Determine the order type and price from entry params
        order_type = entry.params.get("order_type", "MARKET")
        price = entry.params.get("price", 0.0)

        order = OrderEvent(
            timestamp=entry.timestamp,
            order_id=self._next_order_id(),
            ticker=entry.ticker,
            action=entry.action,
            quantity=entry.quantity,
            order_type=order_type,
            price=price,
        )
        logger.info(
            f"[Portfolio] Generated entry Order for {entry.ticker}: "
            f"{entry.action} {entry.quantity} (Type: {order_type})"
        )
        return order

    def process_exit(self, exit_event: ExitEvent) -> Optional[OrderEvent]:
        """
        Translates an ExitEvent into an OrderEvent to close/reduce positions.
        """
        # Fetch current position from account
        pos = self.account.positions.get(exit_event.ticker)
        if pos is None or pos.quantity == 0:
            logger.info(f"[Portfolio] No active position in {exit_event.ticker} to exit.")
            return None

        # Exit action is opposite of current position direction
        action = "SELL" if pos.quantity > 0 else "BUY"
        quantity = abs(pos.quantity)

        order_type = exit_event.params.get("order_type", "MARKET")
        price = exit_event.params.get("price", 0.0)

        order = OrderEvent(
            timestamp=exit_event.timestamp,
            order_id=self._next_order_id(),
            ticker=exit_event.ticker,
            action=action,
            quantity=quantity,
            order_type=order_type,
            price=price,
        )
        logger.info(
            f"[Portfolio] Generated exit Order for {exit_event.ticker}: "
            f"Closing {pos.direction} position of size {quantity}"
        )
        return order

    def generate_portfolio_event(self, timestamp) -> PortfolioEvent:
        """
        Creates a snapshot event of current positions.
        """
        positions_snapshot = {
            ticker: pos.quantity for ticker, pos in self.account.positions.items()
        }
        return PortfolioEvent(
            timestamp=timestamp,
            positions_snapshot=positions_snapshot,
            total_value=self.account.equity,
        )


class SimulationEngine:
    """
    Coordinates components and executes the event queue loop by timestamp.
    No bar-by-bar iterations; entirely event driven.
    """

    def __init__(
        self,
        event_queue: EventQueue,
        account: Account,
        portfolio: Portfolio,
        execution_handler: ExecutionHandler,
        risk_manager: RiskManager,
    ) -> None:
        self.event_queue = event_queue
        self.account = account
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.risk_manager = risk_manager

        # Map event types to their respective handling methods
        self._handlers: Dict[str, Callable[[BaseEvent], None]] = {
            "EntryEvent": self._handle_entry,
            "ExitEvent": self._handle_exit,
            "OrderEvent": self._handle_order,
            "TradeEvent": self._handle_trade,
            "MarketDataEvent": self._handle_market_data,
            "MarginEvent": self._handle_margin,
            "AccountEvent": self._handle_account,
            "PortfolioEvent": self._handle_portfolio,
        }

    def dispatch(self, event: BaseEvent) -> None:
        """Dispatches an event to the registered handler."""
        handler = self._handlers.get(event.event_type)
        if handler:
            handler(event)
        else:
            logger.warning(f"No handler registered for event type: {event.event_type}")

    def run(self) -> None:
        """Runs the event loop until the queue is empty."""
        logger.info("Simulation started.")
        while not self.event_queue.empty():
            event = self.event_queue.pop()
            if event is None:
                break
            self.dispatch(event)
        logger.info("Simulation completed.")

    def _handle_entry(self, event: BaseEvent) -> None:
        assert isinstance(event, EntryEvent)
        order = self.portfolio.process_entry(event)
        if order:
            self.event_queue.push(order)

    def _handle_exit(self, event: BaseEvent) -> None:
        assert isinstance(event, ExitEvent)
        order = self.portfolio.process_exit(event)
        if order:
            self.event_queue.push(order)

    def _handle_order(self, event: BaseEvent) -> None:
        assert isinstance(event, OrderEvent)
        # 1. Risk Manager validation
        if self.risk_manager.validate_order(event, self.account):
            # 2. Sent to execution handler
            trade = self.execution_handler.process_order(event)
            if trade:
                self.event_queue.push(trade)
        else:
            event.status = "REJECTED"

    def _handle_trade(self, event: BaseEvent) -> None:
        assert isinstance(event, TradeEvent)
        # 1. Apply to Account
        success = self.account.apply_fill(
            ticker=event.ticker,
            action=event.action,
            quantity=event.quantity,
            price=event.price,
            commission=event.commission,
        )

        if success:
            # 2. Push state updates (PortfolioEvent, AccountEvent)
            self.event_queue.push(
                self.portfolio.generate_portfolio_event(event.timestamp)
            )
            self.event_queue.push(
                AccountEvent(
                    timestamp=event.timestamp,
                    balance=self.account.balance,
                    equity=self.account.equity,
                    realized_pnl=self.account.total_realized_pnl,
                    unrealized_pnl=self.account.unrealized_pnl,
                )
            )

            # 3. Post-trade Margin Check
            margin_evt = self.risk_manager.check_margin(self.account, event.timestamp)
            if margin_evt:
                self.event_queue.push(margin_evt)

    def _handle_market_data(self, event: BaseEvent) -> None:
        assert isinstance(event, MarketDataEvent)
        # 1. Update prices in execution and account modules
        self.execution_handler.update_price(event.ticker, event.price)
        self.account.update_market_price(event.ticker, event.price)

        # 2. Match any pending LIMIT / STOP orders
        fills = self.execution_handler.match_pending_orders(
            event.ticker, event.price, event.timestamp
        )
        for trade in fills:
            self.event_queue.push(trade)

        # 3. Mark to market updates can trigger margin warnings/liquidations or AccountEvents
        # We push a periodic AccountEvent on price updates
        self.event_queue.push(
            AccountEvent(
                timestamp=event.timestamp,
                balance=self.account.balance,
                equity=self.account.equity,
                realized_pnl=self.account.total_realized_pnl,
                unrealized_pnl=self.account.unrealized_pnl,
            )
        )

        margin_evt = self.risk_manager.check_margin(self.account, event.timestamp)
        if margin_evt:
            self.event_queue.push(margin_evt)

    def _handle_margin(self, event: BaseEvent) -> None:
        assert isinstance(event, MarginEvent)
        # In a real environment, trigger automatic liquidation if Status == 'LIQUIDATED'
        if event.status == "LIQUIDATED":
            logger.critical(
                f"[SimulationEngine] LIQUIDATING PORTFOLIO due to Margin Event: {event}"
            )
            # Generate exits for all open positions
            for ticker, pos in list(self.account.positions.items()):
                if pos.quantity != 0:
                    exit_evt = ExitEvent(timestamp=event.timestamp, ticker=ticker)
                    self.event_queue.push(exit_evt)

    def _handle_account(self, event: BaseEvent) -> None:
        assert isinstance(event, AccountEvent)
        logger.debug(
            f"Account Update -> Balance: ${event.balance:.2f}, Equity: ${event.equity:.2f}, "
            f"Unrealized PnL: ${event.unrealized_pnl:.2f}"
        )

    def _handle_portfolio(self, event: BaseEvent) -> None:
        assert isinstance(event, PortfolioEvent)
        logger.debug(
            f"Portfolio Snapshot -> Value: ${event.total_value:.2f}, Positions: {event.positions_snapshot}"
        )

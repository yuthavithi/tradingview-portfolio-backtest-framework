"""
Unit tests for the Event Driven Portfolio Engine.
"""

from datetime import datetime
import pytest

from portfolio.account import Account
from portfolio.events import (
    BaseEvent,
    EntryEvent,
    ExitEvent,
    OrderEvent,
    TradeEvent,
    MarketDataEvent,
    MarginEvent,
)
from portfolio.queue import EventQueue
from portfolio.engine import (
    RiskManager,
    ExecutionHandler,
    Portfolio,
    SimulationEngine,
)


def test_event_sorting():
    """Verify that events are sorted chronologically in the EventQueue."""
    eq = EventQueue()
    t1 = datetime(2026, 7, 1, 10, 0, 0)
    t2 = datetime(2026, 7, 1, 10, 5, 0)
    t3 = datetime(2026, 7, 1, 10, 10, 0)

    # Push out of order
    eq.push(EntryEvent(timestamp=t3, ticker="AAPL", action="BUY", quantity=10))
    eq.push(EntryEvent(timestamp=t1, ticker="AAPL", action="BUY", quantity=20))
    eq.push(EntryEvent(timestamp=t2, ticker="AAPL", action="BUY", quantity=30))

    assert len(eq) == 3

    # Pop should return them in chronological order
    e1 = eq.pop()
    e2 = eq.pop()
    e3 = eq.pop()

    assert e1.timestamp == t1
    assert e1.quantity == 20

    assert e2.timestamp == t2
    assert e2.quantity == 30

    assert e3.timestamp == t3
    assert e3.quantity == 10


def test_execution_handler_market_order():
    """Test market order execution in ExecutionHandler."""
    eq = EventQueue()
    eh = ExecutionHandler(queue=eq, commission_rate=0.001)

    # Set current price
    eh.update_price("AAPL", 150.0)

    order = OrderEvent(
        timestamp=datetime.now(),
        order_id="ORD1",
        ticker="AAPL",
        action="BUY",
        quantity=10,
        order_type="MARKET",
    )

    trade = eh.process_order(order)
    assert trade is not None
    assert trade.order_id == "ORD1"
    assert trade.ticker == "AAPL"
    assert trade.price == 150.0
    assert trade.quantity == 10
    assert trade.commission == 10 * 150.0 * 0.001
    assert order.status == "FILLED"


def test_execution_handler_limit_order():
    """Test limit order queuing and triggering on price updates."""
    eq = EventQueue()
    eh = ExecutionHandler(queue=eq, commission_rate=0.0)

    eh.update_price("AAPL", 155.0)

    # Buy Limit order at 150 (current price is 155, so it should not trigger)
    order = OrderEvent(
        timestamp=datetime.now(),
        order_id="ORD1",
        ticker="AAPL",
        action="BUY",
        quantity=10,
        order_type="LIMIT",
        price=150.0,
    )

    trade = eh.process_order(order)
    assert trade is None
    assert len(eh.pending_orders) == 1

    # Update price to 151 (still not triggered)
    fills = eh.match_pending_orders("AAPL", 151.0, datetime.now())
    assert len(fills) == 0
    assert len(eh.pending_orders) == 1

    # Update price to 149 (triggered!)
    fills = eh.match_pending_orders("AAPL", 149.0, datetime.now())
    assert len(fills) == 1
    assert fills[0].price == 149.0  # Fills at the better price
    assert len(eh.pending_orders) == 0


def test_risk_manager_leverage_limits():
    """Test RiskManager order rejection based on leverage/margin constraints."""
    acct = Account(initial_capital=1000.0, margin_requirement=0.1)  # 10x max leverage
    rm = RiskManager(max_leverage=5.0)  # Max leverage 5x allowed by RiskManager

    # Trying to buy asset valued at 6000 (leverage = 6x, should fail)
    order = OrderEvent(
        timestamp=datetime.now(),
        order_id="ORD1",
        ticker="AAPL",
        action="BUY",
        quantity=60,
        order_type="MARKET",
        price=100.0,
    )

    assert not rm.validate_order(order, acct)

    # Trying to buy asset valued at 4000 (leverage = 4x, should succeed)
    order_ok = OrderEvent(
        timestamp=datetime.now(),
        order_id="ORD2",
        ticker="AAPL",
        action="BUY",
        quantity=40,
        order_type="MARKET",
        price=100.0,
    )
    assert rm.validate_order(order_ok, acct)


def test_simulation_engine_flow():
    """End-to-end test of the event propagation loop in SimulationEngine."""
    eq = EventQueue()
    acct = Account(initial_capital=10000.0, margin_requirement=1.0)
    port = Portfolio(queue=eq, account=acct)
    eh = ExecutionHandler(queue=eq, commission_rate=0.001)
    rm = RiskManager()
    se = SimulationEngine(
        event_queue=eq,
        account=acct,
        portfolio=port,
        execution_handler=eh,
        risk_manager=rm,
    )

    t = datetime(2026, 7, 1, 12, 0, 0)

    # 1. Update market price first
    eq.push(MarketDataEvent(timestamp=t, ticker="AAPL", price=100.0))

    # 2. Push entry signal
    eq.push(
        EntryEvent(
            timestamp=t,
            ticker="AAPL",
            action="BUY",
            quantity=50,
            params={"order_type": "MARKET"},
        )
    )

    # Execute simulation
    se.run()

    # Verify state updates
    assert acct.balance == 10000.0 - (50 * 100.0 * 0.001)  # cash - commission
    assert "AAPL" in acct.positions
    assert acct.positions["AAPL"].quantity == 50

    # 3. Queue exit signal
    t_exit = datetime(2026, 7, 1, 12, 10, 0)
    eq.push(ExitEvent(timestamp=t_exit, ticker="AAPL", params={"order_type": "MARKET"}))

    # Run again for the exit
    se.run()

    assert "AAPL" not in acct.positions
    assert acct.total_realized_pnl == 0.0  # bought and sold at same price (100)

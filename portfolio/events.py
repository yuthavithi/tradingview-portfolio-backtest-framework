"""
Portfolio Events Module.

This module defines all the event classes used in the event-driven backtesting framework.
All events inherit from BaseEvent and include timestamps for sorting in the event queue.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Union


@dataclass
class BaseEvent:
    """
    Base class for all events in the portfolio engine.

    Attributes:
        timestamp: The datetime or numeric timestamp of the event.
        event_type: A string identifying the type of the event.
    """
    timestamp: Union[datetime, float, int]
    event_type: str = field(init=False)

    def __post_init__(self) -> None:
        self.event_type = self.__class__.__name__

    def __lt__(self, other: "BaseEvent") -> bool:
        """
        Comparison method for sorting events by timestamp.
        If timestamps are equal, falls back to event type comparison
        to prevent comparison of other non-comparable fields.
        """
        if not isinstance(other, BaseEvent):
            return NotImplemented
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.event_type < other.event_type


@dataclass
class EntryEvent(BaseEvent):
    """
    Event representing a signal to enter/buy/short an asset.
    """
    ticker: str
    action: str  # 'BUY' or 'SELL' (short)
    quantity: float
    params: Dict = field(default_factory=dict)


@dataclass
class ExitEvent(BaseEvent):
    """
    Event representing a signal to exit/liquidate an asset.
    """
    ticker: str
    params: Dict = field(default_factory=dict)


@dataclass
class OrderEvent(BaseEvent):
    """
    Event representing a submitted order.
    """
    order_id: str
    ticker: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    order_type: str  # 'MARKET', 'LIMIT', 'STOP'
    price: float = 0.0
    status: str = "SUBMITTED"  # 'SUBMITTED', 'FILLED', 'CANCELLED', 'REJECTED'


@dataclass
class TradeEvent(BaseEvent):
    """
    Event representing a completed trade fill.
    """
    fill_id: str
    order_id: str
    ticker: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    commission: float


@dataclass
class MarginEvent(BaseEvent):
    """
    Event representing margin updates, margin calls, or liquidation status.
    """
    equity: float
    used_margin: float
    margin_level: float
    status: str  # 'WARNING', 'CALL', 'LIQUIDATED'


@dataclass
class AccountEvent(BaseEvent):
    """
    Event representing an update to the account balance, equity, and P&L.
    """
    balance: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float


@dataclass
class PortfolioEvent(BaseEvent):
    """
    Event representing a snapshot/update of the active portfolio holdings.
    """
    positions_snapshot: Dict[str, float]  # Map of ticker to current quantity
    total_value: float


@dataclass
class MarketDataEvent(BaseEvent):
    """
    Event representing a market price tick or update for a specific ticker.
    """
    ticker: str
    price: float


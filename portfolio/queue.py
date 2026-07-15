"""
Portfolio Queue Module.

This module provides the priority queue implementation for scheduling and retrieving
events chronologically in the event-driven simulation.
"""

import heapq
import logging
from typing import Optional

from portfolio.events import BaseEvent

logger = logging.getLogger("portfolio.queue")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class EventQueue:
    """
    Priority Queue wrapper for managing BaseEvents chronologically.

    Under the hood, uses heapq (Min-Heap) comparing event timestamps.
    """

    def __init__(self) -> None:
        self._queue = []

    def push(self, event: BaseEvent) -> None:
        """
        Pushes a new event onto the queue.

        Args:
            event: The event instance to queue.
        """
        heapq.heappush(self._queue, event)
        logger.debug(f"Queued event: {event.event_type} at {event.timestamp}")

    def pop(self) -> Optional[BaseEvent]:
        """
        Pops and returns the earliest event from the queue.

        Returns:
            The chronologically earliest event, or None if the queue is empty.
        """
        if self.empty():
            return None
        event = heapq.heappop(self._queue)
        logger.debug(f"Popped event: {event.event_type} at {event.timestamp}")
        return event

    def peek(self) -> Optional[BaseEvent]:
        """
        Peeks at the earliest event in the queue without removing it.

        Returns:
            The earliest event, or None if empty.
        """
        if self.empty():
            return None
        return self._queue[0]

    def empty(self) -> bool:
        """
        Checks if the queue is empty.

        Returns:
            True if empty, False otherwise.
        """
        return len(self._queue) == 0

    def clear(self) -> None:
        """Clears all events in the queue."""
        self._queue.clear()

    def __len__(self) -> int:
        return len(self._queue)

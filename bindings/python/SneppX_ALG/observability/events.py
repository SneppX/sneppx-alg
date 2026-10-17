"""A tiny synchronous event bus (pure Python)."""

from collections import defaultdict
from typing import Callable, Dict, List, Optional


class EventBus:
    """Publish/subscribe with named topics."""

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)

    def on(self, topic: str, handler: Callable) -> Callable:
        self._handlers[topic].append(handler)
        return handler

    def subscribe(self, topic: str):
        """Decorator form of :meth:`on`."""

        def decorator(func: Callable) -> Callable:
            self.on(topic, func)
            return func

        return decorator

    def off(self, topic: str, handler: Callable) -> None:
        if handler in self._handlers.get(topic, []):
            self._handlers[topic].remove(handler)

    def emit(self, topic: str, payload=None) -> list:
        """Call every handler for ``topic`` and return their results."""
        return [h(payload) for h in list(self._handlers.get(topic, []))]

    def clear(self, topic: Optional[str] = None) -> None:
        if topic is None:
            self._handlers.clear()
        else:
            self._handlers.pop(topic, None)

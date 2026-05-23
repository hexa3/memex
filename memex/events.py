"""Tiny synchronous event bus for hooks and integrations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from memex.errors import MemexError

EventHandler = Callable[[dict[str, Any]], None]


class EventBus:
    """Synchronous hook registry.

    Handlers are intentionally isolated from core persistence: a failing hook
    is surfaced to the caller after the storage operation succeeds.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def on(self, event: str, handler: EventHandler) -> None:
        """Register a handler for an event name."""

        self._handlers[event].append(handler)

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        """Emit an event to all registered handlers."""

        errors: list[Exception] = []
        for handler in self._handlers.get(event, []):
            try:
                handler(payload)
            except Exception as exc:  # pragma: no cover - defensive isolation
                errors.append(exc)
        if errors:
            raise MemexError(f"{len(errors)} event handler(s) failed for {event}") from errors[0]

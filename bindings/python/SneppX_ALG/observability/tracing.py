"""Tracing primitives: nested spans with attributes and events (pure Python)."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Span:
    """A unit of work with a name, timing, attributes and events."""

    name: str
    start: float
    end: Optional[float] = None
    attributes: Dict[str, object] = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)
    parent: Optional[str] = None
    status: str = "ok"

    @property
    def duration(self) -> Optional[float]:
        if self.end is None:
            return None
        return self.end - self.start

    def set_attribute(self, key: str, value) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, **fields) -> None:
        self.events.append({"name": name, "time": time.perf_counter(), **fields})

    def finish(self, status: str = "ok") -> None:
        self.end = time.perf_counter()
        self.status = status

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "attributes": self.attributes,
            "events": self.events,
            "parent": self.parent,
            "status": self.status,
        }


class Tracer:
    """Collects spans for a single trace."""

    def __init__(self, name: str = "trace"):
        self.name = name
        self.spans: List[Span] = []
        self._stack: List[Span] = []

    def start_span(self, name: str, **attributes) -> Span:
        parent = self._stack[-1].name if self._stack else None
        span = Span(
            name=name,
            start=time.perf_counter(),
            attributes=dict(attributes),
            parent=parent,
        )
        self.spans.append(span)
        self._stack.append(span)
        return span

    def end_span(self, span: Span, status: str = "ok") -> None:
        span.finish(status)
        if self._stack and self._stack[-1] is span:
            self._stack.pop()

    @contextmanager
    def span(self, name: str, **attributes):
        s = self.start_span(name, **attributes)
        try:
            yield s
        except Exception:
            self.end_span(s, status="error")
            raise
        else:
            self.end_span(s)

    def to_dict(self) -> dict:
        return {"name": self.name, "spans": [s.to_dict() for s in self.spans]}

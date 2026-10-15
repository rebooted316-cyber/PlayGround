"""Monitoring for deployed agents (roadmap stage 11): traces and metrics."""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start: float
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000


class Tracer:
    """Records nested spans (agent run -> LLM call / tool call)."""

    def __init__(self, max_spans: int = 10000) -> None:
        self.max_spans = max_spans
        self.spans: list[Span] = []
        self._local = threading.local()
        self._lock = threading.Lock()

    def _stack(self) -> list[Span]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    @contextmanager
    def span(self, name: str, **attributes: Any):
        stack = self._stack()
        parent = stack[-1] if stack else None
        sp = Span(
            name=name,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex[:16],
            span_id=uuid.uuid4().hex[:8],
            parent_id=parent.span_id if parent else None,
            start=time.perf_counter(),
            attributes=dict(attributes),
        )
        stack.append(sp)
        try:
            yield sp
        except Exception as exc:
            sp.status = "error"
            sp.attributes["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.end = time.perf_counter()
            stack.pop()
            with self._lock:
                self.spans.append(sp)
                if len(self.spans) > self.max_spans:
                    del self.spans[: len(self.spans) - self.max_spans]

    def trace(self, trace_id: str) -> list[dict[str, Any]]:
        out = []
        for sp in sorted((s for s in self.spans if s.trace_id == trace_id), key=lambda s: s.start):
            item = asdict(sp)
            item["duration_ms"] = round(sp.duration_ms, 2)
            out.append(item)
        return out


class Metrics:
    """Thread-safe counters and latency histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, float] = {}
        self.timings: dict[str, list[float]] = {}

    def incr(self, name: str, value: float = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + value

    def observe(self, name: str, ms: float) -> None:
        with self._lock:
            samples = self.timings.setdefault(name, [])
            samples.append(ms)
            if len(samples) > 5000:
                del samples[:1000]

    @contextmanager
    def timer(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - start) * 1000)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            timings = {}
            for name, samples in self.timings.items():
                ordered = sorted(samples)
                timings[name] = {
                    "count": len(ordered),
                    "p50_ms": round(_pct(ordered, 0.5), 2),
                    "p95_ms": round(_pct(ordered, 0.95), 2),
                    "max_ms": round(ordered[-1], 2) if ordered else 0.0,
                }
            return {"counters": dict(self.counters), "timings": timings}


def _pct(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


__all__ = ["Metrics", "Span", "Tracer"]

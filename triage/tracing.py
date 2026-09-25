"""Lightweight local tracing for the RAG/agent flow.

Records per-node, per-LLM and per-tool latencies so a run can be inspected
after the fact. For a hosted observability backend, see the README section on
Langfuse/LangSmith integration.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self._events: list[dict[str, Any]] = []

    @contextmanager
    def span(self, name: str, **detail: Any) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            self._events.append(
                {"name": name, "latency_ms": round(latency_ms, 3), **detail}
            )

    def record(self, name: str, latency_ms: float, **detail: Any) -> None:
        self._events.append({"name": name, "latency_ms": round(latency_ms, 3), **detail})

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def summary(self) -> dict[str, dict[str, float]]:
        by_name: dict[str, list[float]] = {}
        for event in self._events:
            by_name.setdefault(event["name"], []).append(event["latency_ms"])
        return {
            name: {
                "count": len(values),
                "avg_ms": round(sum(values) / len(values), 3),
                "p95_ms": _p95(values),
            }
            for name, values in by_name.items()
        }

    def save(self, path: Path | str | None = None) -> None:
        target = Path(path) if path else self._path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self._events, indent=2))


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return round(ordered[index], 3)

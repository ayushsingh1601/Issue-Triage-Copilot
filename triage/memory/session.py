"""Session state for a single triage run."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Session(BaseModel):
    issue_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    events: list[dict[str, Any]] = Field(default_factory=list)
    decision: dict[str, Any] | None = None

    def record(self, name: str, **data: Any) -> None:
        self.events.append(
            {"name": name, "at": datetime.now(UTC).isoformat(), **data}
        )

    @property
    def latency_seconds(self) -> float:
        return (datetime.now(UTC) - self.started_at).total_seconds()

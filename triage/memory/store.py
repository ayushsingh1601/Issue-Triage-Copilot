"""Agent evidence cache keyed by issue id, with optional JSON persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvidenceStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self._data: dict[str, dict[str, Any]] = {}
        if self._path and self._path.exists():
            self._data = json.loads(self._path.read_text())

    def get(self, issue_id: str) -> dict[str, Any] | None:
        return self._data.get(issue_id)

    def put(self, issue_id: str, evidence: dict[str, Any]) -> None:
        self._data[issue_id] = evidence
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data))

    def clear(self) -> None:
        self._data.clear()

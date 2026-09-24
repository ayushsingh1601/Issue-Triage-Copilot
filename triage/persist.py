"""Dataset persistence: save and load IssueRecord and ProcessDoc collections."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def save_records(records: list[T], path: Path | str, fmt: str = "json") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [record.model_dump(mode="json") for record in records]
    if fmt == "json":
        path.write_text(json.dumps(data, indent=2))
    elif fmt == "parquet":
        pq.write_table(pa.Table.from_pylist(data), path)
    else:
        raise ValueError(f"unknown format: {fmt}")


def load_records(path: Path | str, model: type[T], fmt: str | None = None) -> list[T]:
    path = Path(path)
    fmt = fmt or path.suffix.lstrip(".")
    if fmt == "json":
        data = json.loads(path.read_text())
    elif fmt == "parquet":
        data = pq.read_table(path).to_pylist()
    else:
        raise ValueError(f"unknown format: {fmt}")
    return [model.model_validate(item) for item in data]

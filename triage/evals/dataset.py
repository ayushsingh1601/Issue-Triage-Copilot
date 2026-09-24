"""Held-out splitter: carves an eval set that never enters the corpus."""
from __future__ import annotations

import random

from pydantic import BaseModel
from triage.rag.parse import IssueRecord


class HeldOutSplit(BaseModel):
    corpus: list[IssueRecord]
    held_out: list[IssueRecord]

    def assert_disjoint(self) -> None:
        overlap = {
            (r.repo, r.number) for r in self.corpus
        } & {(r.repo, r.number) for r in self.held_out}
        if overlap:
            raise ValueError(f"corpus and held-out overlap: {overlap}")


def held_out_split(
    records: list[IssueRecord],
    held_out_fraction: float = 0.15,
    seed: int = 42,
) -> HeldOutSplit:
    if not 0.0 < held_out_fraction < 1.0:
        raise ValueError("held_out_fraction must be in (0, 1)")
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    n_held = round(len(shuffled) * held_out_fraction)
    split = HeldOutSplit(corpus=shuffled[n_held:], held_out=shuffled[:n_held])
    split.assert_disjoint()
    return split

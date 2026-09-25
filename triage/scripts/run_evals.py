#!/usr/bin/env python
"""Run the vanilla-vs-multi-agent comparison on held-out issues."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from triage.evals.runner import EvaluationRunner, sweep_doc_chunking
from triage.rag.embed import Embedder

PROCESSED = Path("triage/data/processed")
INDEXES = Path("triage/data/indexes")


def print_table(results) -> None:
    header = ["system"] + list(next(iter(results.values())).to_dict())
    rows = [
        [system] + [str(value) for value in result.to_dict().values()]
        for system, result in results.items()
    ]
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    parser.add_argument("--indexes", type=Path, default=INDEXES)
    parser.add_argument("--sweep", action="store_true", help="run the doc-chunking sweep")
    parser.add_argument("--held-out-limit", type=int, default=None)
    args = parser.parse_args()

    runner = EvaluationRunner(args.indexes, args.processed)
    results = runner.run_comparison()
    print_table(results)
    print()

    if args.sweep:
        embedder = Embedder(cache_path=args.indexes / "embeddings.json")
        rows = sweep_doc_chunking(
            args.processed, args.indexes, embedder, held_out_limit=args.held_out_limit
        )
        print("doc-chunking sweep:")
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()

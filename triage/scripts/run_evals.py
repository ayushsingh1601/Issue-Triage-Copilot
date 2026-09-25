#!/usr/bin/env python
"""Run the vanilla-vs-multi-agent comparison on held-out issues."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from triage.evals.runner import (
    EvaluationRunner,
    format_results_table,
    sweep_doc_chunking,
    sweep_retrieval_strategies,
)
from triage.logging import silence_libraries
from triage.rag.embed import Embedder

PROCESSED = Path("triage/data/processed")
INDEXES = Path("triage/data/indexes")


def main() -> None:
    silence_libraries()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    parser.add_argument("--indexes", type=Path, default=INDEXES)
    parser.add_argument("--sweep", action="store_true", help="run the doc-chunking sweep")
    parser.add_argument(
        "--sweep-retrieval", action="store_true", help="run the retrieval-strategy sweep"
    )
    parser.add_argument("--held-out-limit", type=int, default=None)
    args = parser.parse_args()

    runner = EvaluationRunner(args.indexes, args.processed)
    results = runner.run_comparison(limit=args.held_out_limit)
    print(format_results_table(results))
    print()

    if args.sweep:
        embedder = Embedder(cache_path=args.indexes / "embeddings.json")
        rows = sweep_doc_chunking(
            args.processed, args.indexes, embedder, held_out_limit=args.held_out_limit
        )
        print("doc-chunking sweep:")
        for row in rows:
            print(row)

    if args.sweep_retrieval:
        rows = sweep_retrieval_strategies(
            args.processed,
            args.indexes,
            held_out_limit=args.held_out_limit,
        )
        print("retrieval-strategy sweep:")
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()

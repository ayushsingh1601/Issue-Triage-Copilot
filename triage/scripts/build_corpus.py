#!/usr/bin/env python
"""Build or refresh both RAG indexes from the persisted dataset.

Default: clean build (both collections cleared first, embedding cache kept).
--refresh: incremental — adds newly-ingested issues, deletes evicted ones, and
rebuilds the small doc collection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from triage.persist import load_records
from triage.rag.embed import Embedder
from triage.rag.index_build import (
    build_doc_index,
    build_issue_index,
    sync_doc_index,
    sync_issue_index,
)
from triage.rag.parse import IssueRecord, ProcessDoc
from triage.rag.store import ChromaStore

INDEXES = Path("triage/data/indexes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("triage/data/processed"))
    parser.add_argument("--indexes", type=Path, default=INDEXES)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--flat", action="store_true", help="use flat doc chunking (no headings)")
    parser.add_argument("--refresh", action="store_true", help="incremental sync of the indexes")
    args = parser.parse_args()

    records = load_records(args.processed / "issues_corpus.json", IssueRecord)
    docs = load_records(args.processed / "process_docs.json", ProcessDoc)
    embedder = Embedder(cache_path=args.indexes / "embeddings.json")

    if args.refresh:
        stats = sync_issue_index(records, embedder, args.indexes)
        doc_stats = sync_doc_index(
            docs,
            embedder,
            args.indexes,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            structural=not args.flat,
        )
        print(
            f"refreshed issue index: added={stats['added']} removed={stats['removed']} "
            f"unchanged={stats['unchanged']}; docs={doc_stats}"
        )
        return

    ChromaStore(args.indexes / "issues", "issues").clear()
    ChromaStore(args.indexes / "docs", "docs").clear()
    build_issue_index(records, embedder, args.indexes)
    build_doc_index(
        docs,
        embedder,
        args.indexes,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        structural=not args.flat,
    )
    print(
        f"built issue index ({len(records)} issues) and doc index "
        f"({len(docs)} docs) in {args.indexes}"
    )


if __name__ == "__main__":
    main()

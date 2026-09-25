#!/usr/bin/env python
"""Run the multi-agent triage demo on a fresh issue and compare with the actual resolution."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from triage.mcp_tools.langchain import AgentToolbox
from triage.mcp_tools.tools import TriageTools
from triage.memory.session import Session
from triage.observability import graph_config
from triage.orchestration.graph import build_graph
from triage.orchestration.state import TriageState
from triage.persist import load_records
from triage.rag.parse import IssueRecord
from triage.tracing import Tracer

PROCESSED = Path("triage/data/processed")
INDEXES = Path("triage/data/indexes")


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def summarize_actual(record: IssueRecord) -> dict[str, Any]:
    return {
        "issue_id": f"{record.repo}#{record.number}",
        "title": record.title,
        "actual_labels": record.labels,
        "linked_prs": record.linked_prs,
        "closing_comment": record.comments[-1].body if record.comments else None,
    }


async def run_triage(
    tools: TriageTools,
    query: str,
    issue_id: str,
    tracer: Tracer | None = None,
) -> dict[str, Any]:
    async with AgentToolbox(tools) as box:
        graph = build_graph(toolbox=box, tracer=tracer).compile()
        return await graph.ainvoke(
            TriageState(issue=query, issue_id=issue_id),
            config=graph_config(),
        )


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    parser.add_argument("--indexes", type=Path, default=INDEXES)
    parser.add_argument(
        "--held-out-index", type=int, default=None, help="triage the Nth held-out issue"
    )
    parser.add_argument("--issue-id", default="fresh#1")
    parser.add_argument("--text", default="", help="title and body of a fresh issue")
    parser.add_argument(
        "--trace", type=Path, default=None, help="save a latency trace to this path"
    )
    args = parser.parse_args()

    actual: dict[str, Any] | None = None
    if args.held_out_index is not None:
        held_out = load_records(args.processed / "issues_held_out.json", IssueRecord)
        record = held_out[args.held_out_index]
        query = f"{record.title}\n\n{record.body}"
        issue_id = f"{record.repo}#{record.number}"
        actual = summarize_actual(record)
    else:
        if not args.text:
            raise SystemExit("provide --held-out-index or --text for a fresh issue")
        query = args.text
        issue_id = args.issue_id

    tools = TriageTools(indexes_dir=args.indexes, processed_dir=args.processed)
    session = Session(issue_id=issue_id)
    tracer = Tracer(path=args.trace) if args.trace else None
    session.record("start", query_length=len(query))
    result = asyncio.run(run_triage(tools, query, issue_id, tracer))
    session.record("end")
    session.decision = result["decision"].model_dump(mode="json") if result["decision"] else None
    if tracer:
        tracer.save()
        print("TRACE SUMMARY (avg/p95 ms per stage):")
        for name, stats in tracer.summary().items():
            print(f"  {name}: count={stats['count']} avg={stats['avg_ms']} p95={stats['p95_ms']}")
        print(f"  trace saved to {args.trace}")

    print("=" * 60)
    print(f"ISSUE {issue_id}  ({'held-out' if actual else 'fresh'})")
    print(f"classification: {result['classification']}")
    print(f"needs_human: {result['needs_human']}")
    print()
    print("FINAL DECISION:")
    print(result["decision"].model_dump_json(indent=2) if result["decision"] else "none")
    if actual:
        print()
        print("ACTUAL RESOLUTION:")
        for key, value in actual.items():
            print(f"  {key}: {value}")
        if result["decision"]:
            print("  suggested vs actual labels:",
                  set(result["decision"].suggested_labels), "vs", set(actual["actual_labels"]))
    print()
    print(f"latency: {session.latency_seconds:.2f}s over {len(session.events)} events")


if __name__ == "__main__":
    main()

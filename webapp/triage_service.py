"""Run the multi-agent triage graph over a cached repo and return a renderable result."""
from __future__ import annotations

import time
from typing import Any

from triage.guardrails.input_guard import InputGuard
from triage.mcp_tools.langchain import AgentToolbox
from triage.mcp_tools.tools import TriageTools
from triage.observability import graph_config
from triage.orchestration.graph import build_graph
from triage.orchestration.state import TriageState
from triage.rag.parse import IssueRecord
from triage.rag.rerank import Reranker
from triage.rag.rewrite import QueryRewriter
from triage.tracing import Tracer

from webapp.repo_service import RepoPaths, run_async, triage_lock


def build_tools(paths: RepoPaths) -> TriageTools:
    return TriageTools(
        indexes_dir=paths.indexes,
        processed_dir=paths.processed,
        rewriter=QueryRewriter(),
        reranker=Reranker(),
    )


def summarize_issue(record: IssueRecord | None) -> dict[str, Any] | None:
    """Compact view of an issue's actual resolution for the UI."""
    if record is None:
        return None
    return {
        "issue_id": f"{record.repo}#{record.number}",
        "title": record.title,
        "state": record.state,
        "actual_labels": record.labels,
        "linked_prs": record.linked_prs,
        "closing_comment": record.comments[-1].body if record.comments else None,
    }


def triage_issue(
    paths: RepoPaths,
    query: str,
    issue_id: str,
    *,
    use_guard: bool = True,
) -> dict[str, Any]:
    """Run one triage synchronously (Streamlit-friendly) and return the full result.

    The graph runs on a single long-lived process event loop and is
    serialized per repo: LLM/HTTP clients must not be torn down against a
    closed loop (a fresh ``asyncio.run`` per call breaks the second triage
    with "Event loop is closed"), and ChromaDB is not thread-safe.
    """
    with triage_lock(paths.key):
        return run_async(_triage_async(paths, query, issue_id, use_guard=use_guard))


async def _triage_async(
    paths: RepoPaths,
    query: str,
    issue_id: str,
    *,
    use_guard: bool,
) -> dict[str, Any]:
    tools = build_tools(paths)
    tracer = Tracer()
    input_guard = InputGuard() if use_guard else None
    start = time.perf_counter()
    async with AgentToolbox(tools) as box:
        graph = build_graph(toolbox=box, tracer=tracer, input_guard=input_guard).compile()
        result = await graph.ainvoke(
            TriageState(issue=query, issue_id=issue_id), config=graph_config()
        )
    guard = result["guard_result"]
    decision = result["decision"].model_dump(mode="json") if result["decision"] else None
    return {
        "issue_id": issue_id,
        "guard": {"allowed": guard.allowed, "reason": guard.reason},
        "rejected": result["rejected"],
        "classification": result["classification"],
        "needs_human": result["needs_human"],
        "decision": decision,
        "latency_seconds": round(time.perf_counter() - start, 2),
        "trace_summary": tracer.summary(),
    }

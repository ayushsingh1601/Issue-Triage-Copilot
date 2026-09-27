"""Run the multi-agent triage graph over a cached repo and return a renderable result."""
from __future__ import annotations

import json
import time
from typing import Any

from triage.agents.vanilla import VanillaAgent
from triage.evals.judge import Judge
from triage.evals.metrics import action_overlap, label_accuracy
from triage.evals.runner import _context_text, _evidence_text
from triage.guardrails.input_guard import InputGuard
from triage.mcp_tools.langchain import AgentToolbox
from triage.mcp_tools.tools import TriageTools
from triage.observability import graph_config
from triage.orchestration.graph import build_graph
from triage.orchestration.state import TriageState
from triage.persist import load_records
from triage.rag.parse import IssueRecord
from triage.rag.pipeline import VanillaPipeline
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
        "historical_evidence": result["historical_evidence"],
        "runbook_steps": result["runbook_steps"],
        "latency_seconds": round(time.perf_counter() - start, 2),
        "trace_summary": tracer.summary(),
    }


def evaluate_issue(
    paths: RepoPaths,
    record: IssueRecord,
    multi_result: dict[str, Any],
) -> dict[str, Any]:
    """Score the multi-agent decision for ONE issue vs the vanilla baseline.

    Objective metrics use the actual resolution (labels, closing comment,
    linked PRs) as the reference; judge metrics use the same evidence-aware
    context as the aggregate eval (multi is judged against the evidence its
    specialists gathered, vanilla against the raw retriever top-k). Only the
    cheap vanilla pipeline is re-run — the multi-agent result is reused.
    """
    query = f"{record.title}\n\n{record.body}"
    issue_id = f"{record.repo}#{record.number}"

    with triage_lock(paths.key):
        async def _run() -> dict[str, Any]:
            tools = build_tools(paths)
            vanilla = await _run_vanilla(tools, query, issue_id)
            issue_matches, doc_matches = tools.retriever().retrieve(
                query, k_issues=10, k_docs=3
            )
            raw_context = _context_text(issue_matches, doc_matches)
            corpus = load_records(paths.processed / "issues_corpus.json", IssueRecord)
            evidence = {
                "similar_issues": multi_result.get("historical_evidence") or [],
                "runbook_steps": multi_result.get("runbook_steps") or [],
            }
            multi_context = f"{_evidence_text(evidence, corpus)}\n\n{raw_context}"
            if not (evidence["similar_issues"] or evidence["runbook_steps"]):
                multi_context = raw_context
            judge = Judge()
            return {
                "issue_id": issue_id,
                "actual_labels": record.labels,
                "multi": _score_decision(
                    multi_result.get("decision") or {}, record, query, multi_context, judge
                ),
                "vanilla": _score_decision(
                    vanilla.model_dump(mode="json"), record, query, raw_context, judge
                ),
            }

        return run_async(_run())


async def _run_vanilla(tools: TriageTools, query: str, issue_id: str) -> Any:
    pipeline = VanillaPipeline(retriever=tools.retriever(), agent=VanillaAgent())
    return await pipeline.run(query, issue_id, config=graph_config())


def _score_decision(
    decision: dict[str, Any],
    record: IssueRecord,
    query: str,
    context: str,
    judge: Judge,
) -> dict[str, float]:
    reference = record.comments[-1].body if record.comments else ""
    labels = label_accuracy(decision.get("suggested_labels") or [], record.labels)
    overlap = action_overlap(
        decision.get("next_steps") or [],
        reference,
        record.linked_prs,
    )
    judged = judge.evaluate(query, json.dumps(decision), context)
    return {
        "label_top1": labels["top1"],
        "label_top3": labels["top3"],
        "action_rouge_l": overlap["rouge_l"],
        "action_entity_match": overlap["entity_match"],
        "answer_relevancy": float(judged["answer_relevancy"].score),
        "context_relevance": float(judged["context_relevance"].score),
        "groundedness": float(judged["groundedness"].score),
    }

import json

from langchain_core.messages import AIMessage
from triage.evals.runner import (
    Components,
    EvaluationRunner,
    sweep_doc_chunking,
    sweep_retrieval_strategies,
)
from triage.mcp_tools.tools import TriageTools
from triage.persist import save_records
from triage.rag.embed import Embedder
from triage.rag.index_build import build_issue_index
from triage.rag.parse import IssueComment, IssueRecord, ProcessDoc
from triage.rag.store import ChromaStore


def make_record(repo: str, number: int, labels: list[str], closing: str = "") -> IssueRecord:
    comments = []
    if closing:
        comments = [
            IssueComment(
                id=1,
                author="maint",
                author_association="MEMBER",
                body=closing,
                created_at="2024-01-02",
            )
        ]
    return IssueRecord(
        repo=repo,
        number=number,
        title=f"Bug {number}",
        body="crash on empty frame with stack trace",
        state="closed",
        author="a",
        created_at="2024-01-01",
        closed_at="2024-01-02",
        labels=labels,
        linked_prs=[45],
        comments=comments,
    )


def fake_embed(texts):
    return [[1.0, 0.0] for _ in texts]


def decision_json(labels: list[str] | None = None, steps: list[str] | None = None) -> str:
    return json.dumps(
        {
            "issue_id": "x/y#99",
            "suggested_labels": labels or ["bug"],
            "triage_route": "bug",
            "next_steps": steps or ["fixed the crash"],
            "affected_modules": ["frame.py"],
            "similar_issues": [],
            "citations": ["x/y#1"],
        }
    )


class HistoricalScriptedModel:
    def __init__(self) -> None:
        self._steps = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_issue_details",
                        "args": {"issue_id": "x/y#1"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="similar crash"),
        ]
        self._index = 0

    async def ainvoke(self, messages, config=None):
        step = self._steps[self._index % len(self._steps)]
        self._index += 1
        return step


class ProcessScriptedModel:
    def __init__(self) -> None:
        self._steps = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_runbook_steps",
                        "args": {"issue_type": "bug", "query": "crash"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="runbook"),
        ]
        self._index = 0

    async def ainvoke(self, messages, config=None):
        step = self._steps[self._index % len(self._steps)]
        self._index += 1
        return step


def fixed_judge(prompt):
    return json.dumps({"verdict": "yes", "reason": "ok"})


def make_env(tmp_path):
    corpus = [make_record("x/y", 1, ["bug"], "fixed the crash"), make_record("x/y", 2, ["bug"])]
    held_out = [make_record("x/y", 50, ["bug"], "fixed the crash"), make_record("x/y", 51, ["bug"])]
    docs = [
        ProcessDoc(repo="x/y", path="CONTRIBUTING.md", content="# Intro\nhow to contribute.\n")
    ]
    save_records(corpus, tmp_path / "processed" / "issues_corpus.json")
    save_records(held_out, tmp_path / "processed" / "issues_held_out.json")
    save_records(docs, tmp_path / "processed" / "process_docs.json")
    embedder = Embedder(embed_fn=fake_embed)
    build_issue_index(corpus, embedder, tmp_path / "indexes")
    doc_store = ChromaStore(tmp_path / "indexes" / "docs", "docs")
    doc_store.add(
        ids=["CONTRIBUTING.md#top#0"],
        texts=["# Contributing\nhow to contribute."],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"repo": "x/y", "path": "CONTRIBUTING.md", "heading_path": ""}],
    )


def make_runner(tmp_path) -> EvaluationRunner:
    make_env(tmp_path)
    components = Components(
        tools=TriageTools(
            indexes_dir=tmp_path / "indexes",
            processed_dir=tmp_path / "processed",
            embed_fn=fake_embed,
            llm_respond=lambda prompt: json.dumps({"type": "bug", "confidence": 0.9}),
        ),
        vanilla_respond=lambda prompt: decision_json(),
        orchestrator_respond=lambda prompt: decision_json(),
        historical_model=HistoricalScriptedModel(),
        process_model=ProcessScriptedModel(),
        judge_respond=fixed_judge,
    )
    return EvaluationRunner(tmp_path / "indexes", tmp_path / "processed", components)


def test_comparison_runs_both_systems(tmp_path):
    runner = make_runner(tmp_path)
    results = runner.run_comparison()
    assert set(results) == {"vanilla", "multi"}
    for _system, result in results.items():
        assert result.label_top1 == 1.0
        assert result.label_top3 == 1.0
        assert result.action_rouge_l >= 0.5
        assert result.answer_relevancy == 1.0
        assert result.context_relevance == 1.0
        assert result.groundedness == 1.0
        assert 0.0 <= result.recall_at_k <= 1.0
        assert result.latency_p95 >= 0.0


def test_multi_cost_higher_than_vanilla(tmp_path):
    runner = make_runner(tmp_path)
    results = runner.run_comparison()
    assert results["multi"].cost > results["vanilla"].cost
    assert results["vanilla"].cost > 0.0


def test_sweep_doc_chunking(tmp_path):
    make_env(tmp_path)
    embedder = Embedder(embed_fn=fake_embed)
    configs = [
        {"chunk_size": 100, "overlap": 0.1, "structural": True},
        {"chunk_size": 100, "overlap": 0.1, "structural": False},
    ]
    rows = sweep_doc_chunking(
        tmp_path / "processed",
        tmp_path / "sweep",
        embedder,
        judge_respond=fixed_judge,
        held_out_limit=1,
        configs=configs,
    )
    assert len(rows) == 2
    for row in rows:
        assert "recall_at_10" in row
        assert "precision_at_10" in row
        assert row["context_relevance"] == 1.0


def test_sweep_retrieval_strategies(tmp_path):
    make_env(tmp_path)
    rows = sweep_retrieval_strategies(
        tmp_path / "processed",
        tmp_path / "indexes",
        embed_fn=fake_embed,
        llm_respond=lambda prompt: json.dumps({"type": "bug", "confidence": 0.9}),
        rewrite_respond=lambda prompt: json.dumps({"query": "crash"}),
        rerank_respond=lambda prompt: json.dumps({"order": [0], "reason": "ok"}),
        judge_respond=fixed_judge,
        held_out_limit=1,
    )
    assert [row["strategy"] for row in rows] == [
        "base",
        "rewrite",
        "rerank",
        "rewrite+rerank",
    ]
    for row in rows:
        assert "recall_at_10" in row
        assert "precision_at_10" in row
        assert row["context_relevance"] == 1.0

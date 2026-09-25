import json

from triage.mcp_tools.tools import TriageTools
from triage.persist import save_records
from triage.rag.embed import Embedder
from triage.rag.rerank import Reranker
from triage.rag.retriever import Retriever
from triage.rag.rewrite import QueryRewriter
from triage.rag.store import ChromaStore


def make_env(tmp_path):
    records = [
        {"id": "x/y#1", "text": "Bug 1\n\ncrash on empty frame", "vec": [1.0, 0.0]},
        {"id": "x/y#2", "text": "Bug 2\n\ninstall failure", "vec": [0.0, 1.0]},
        {"id": "x/y#3", "text": "Bug 3\n\nmerge conflict", "vec": [-1.0, 0.0]},
    ]
    from triage.rag.parse import IssueRecord

    issue_records = [
        IssueRecord(
            repo="x/y",
            number=int(item["id"].split("#")[1]),
            title=f"Bug {item['id'].split('#')[1]}",
            body="body",
            state="closed",
            author="a",
            created_at="",
            closed_at=None,
            labels=["bug"],
            linked_prs=[],
            comments=[],
        )
        for item in records
    ]
    save_records(issue_records, tmp_path / "processed" / "issues_corpus.json")
    issue_store = ChromaStore(tmp_path / "indexes" / "issues", "issues")
    issue_store.add(
        ids=[item["id"] for item in records],
        texts=[item["text"] for item in records],
        embeddings=[item["vec"] for item in records],
        metadatas=[{"repo": "x/y", "number": int(item["id"].split("#")[1])} for item in records],
    )
    doc_store = ChromaStore(tmp_path / "indexes" / "docs", "docs")
    doc_store.add(
        ids=["CONTRIBUTING.md#top#0"],
        texts=["# Contributing\nreport bugs with a repro."],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"repo": "x/y", "path": "CONTRIBUTING.md", "heading_path": ""}],
    )
    return issue_store, doc_store


def fake_embed(texts):
    return [[1.0, 0.0] for _ in texts]


def make_rewriter() -> QueryRewriter:
    return QueryRewriter(respond=lambda prompt: json.dumps({"query": "crash on empty frame"}))


def make_reranker(order) -> Reranker:
    return Reranker(respond=lambda prompt: json.dumps({"order": order, "reason": "ok"}))


def test_retriever_without_components_keeps_order(tmp_path):
    issue_store, doc_store = make_env(tmp_path)
    retriever = Retriever(
        embedder=Embedder(embed_fn=fake_embed),
        issue_store=issue_store,
        doc_store=doc_store,
    )
    issues, docs = retriever.retrieve("crash", k_issues=2, k_docs=1)
    assert issues[0].id == "x/y#1"


def test_retriever_rerank_reorders(tmp_path):
    issue_store, doc_store = make_env(tmp_path)
    retriever = Retriever(
        embedder=Embedder(embed_fn=fake_embed),
        issue_store=issue_store,
        doc_store=doc_store,
        rewriter=make_rewriter(),
        reranker=make_reranker([2, 0]),
    )
    issues, _ = retriever.retrieve("crash", k_issues=2, k_docs=1)
    assert [m.id for m in issues] == ["x/y#3", "x/y#1"]


def test_tools_search_past_issues_reranks(tmp_path):
    issue_store, doc_store = make_env(tmp_path)
    tools = TriageTools(
        indexes_dir=tmp_path / "indexes",
        processed_dir=tmp_path / "processed",
        embed_fn=fake_embed,
        llm_respond=lambda prompt: json.dumps({"type": "bug", "confidence": 0.9}),
        rewriter=make_rewriter(),
        reranker=make_reranker([2, 0, 1]),
    )
    results = tools.search_past_issues("crash on empty frame", limit=3)
    assert [r["issue_id"] for r in results] == ["x/y#3", "x/y#1", "x/y#2"]


def test_tools_runbook_reranks_to_limit(tmp_path):
    issue_store, doc_store = make_env(tmp_path)
    tools = TriageTools(
        indexes_dir=tmp_path / "indexes",
        processed_dir=tmp_path / "processed",
        embed_fn=fake_embed,
        llm_respond=lambda prompt: json.dumps({"type": "bug", "confidence": 0.9}),
        reranker=make_reranker([0]),
    )
    steps = tools.get_runbook_steps("bug", "how to report")
    assert len(steps) <= 4
    assert steps[0]["source"] == "CONTRIBUTING.md#top#0"

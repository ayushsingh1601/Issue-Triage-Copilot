import asyncio
import json

from triage.agents.vanilla import VanillaAgent
from triage.guardrails.schema import TriageDecision
from triage.jsonutil import extract_json
from triage.prompts.vanilla import build_vanilla_prompt
from triage.rag.embed import Embedder
from triage.rag.pipeline import VanillaPipeline, build_retriever
from triage.rag.retriever import Retriever
from triage.rag.store import ChromaStore


def fake_embed_fn(texts):
    return [[float(i + 1), 0.0] for i, _ in enumerate(texts)]


def make_indexes(tmp_path):
    embedder = Embedder(embed_fn=fake_embed_fn)
    issue_store = ChromaStore(tmp_path / "issues", "issues")
    issue_store.add(
        ids=["x/y#1", "x/y#2", "x/y#3"],
        texts=[
            "Bug 1\n\ncrash on empty frame with stack trace",
            "Bug 2\n\nmerge conflict on main",
            "Bug 3\n\npip install fails offline",
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        metadatas=[
            {"repo": "x/y", "number": 1, "title": "Bug 1"},
            {"repo": "x/y", "number": 2, "title": "Bug 2"},
            {"repo": "x/y", "number": 3, "title": "Bug 3"},
        ],
    )
    doc_store = ChromaStore(tmp_path / "docs", "docs")
    doc_store.add(
        ids=["CONTRIBUTING.md#top#0"],
        texts=[
            "# Contributing\nReport bugs with a minimal repro. "
            "Labels are applied by maintainers."
        ],
        embeddings=[[1.5, 0.0]],
        metadatas=[{"repo": "x/y", "path": "CONTRIBUTING.md", "heading_path": ""}],
    )
    return embedder, issue_store, doc_store


def decision_json(issue_id: str = "x/y#999") -> str:
    return json.dumps(
        {
            "issue_id": issue_id,
            "suggested_labels": ["bug"],
            "triage_route": "route to maintainers",
            "next_steps": ["Reproduce with a minimal example"],
            "affected_modules": ["pandas/core/frame.py"],
            "similar_issues": [
                {"issue_id": "x/y#1", "repo": "x/y", "title": "Bug 1", "reason": "same crash"}
            ],
            "citations": ["x/y#1", "CONTRIBUTING.md (top)"],
        }
    )


def test_prompt_includes_context_and_citations(tmp_path):
    embedder, issue_store, doc_store = make_indexes(tmp_path)
    retriever = Retriever(embedder=embedder, issue_store=issue_store, doc_store=doc_store)
    issue_matches, doc_matches = retriever.retrieve("crash on empty frame", k_issues=2, k_docs=1)
    prompt = build_vanilla_prompt("crash on empty frame", issue_matches, doc_matches)
    assert "x/y#1" in prompt
    assert "crash on empty frame" in prompt
    assert "CONTRIBUTING.md" in prompt
    assert "citations" in prompt


def test_vanilla_pipeline_validates_schema_and_citations(tmp_path):
    embedder, issue_store, doc_store = make_indexes(tmp_path)
    retriever = Retriever(embedder=embedder, issue_store=issue_store, doc_store=doc_store)
    agent = VanillaAgent(respond=lambda prompt: decision_json())
    pipeline = VanillaPipeline(retriever=retriever, agent=agent)
    decision = asyncio.run(pipeline.run("crash on empty frame", "x/y#999"))
    assert isinstance(decision, TriageDecision)
    assert decision.issue_id == "x/y#999"
    assert decision.suggested_labels == ["bug"]
    assert decision.similar_issues[0].issue_id == "x/y#1"
    assert decision.citations
    assert all("x/y#" in c or ".md" in c for c in decision.citations)


def test_vanilla_pipeline_handles_fenced_json(tmp_path):
    embedder, issue_store, doc_store = make_indexes(tmp_path)
    retriever = Retriever(embedder=embedder, issue_store=issue_store, doc_store=doc_store)
    agent = VanillaAgent(respond=lambda prompt: f"```json\n{decision_json()}\n```")
    pipeline = VanillaPipeline(retriever=retriever, agent=agent)
    decision = asyncio.run(pipeline.run("crash", "x/y#999"))
    assert decision.issue_id == "x/y#999"


def test_build_retriever_from_disk(tmp_path):
    embedder, _, _ = make_indexes(tmp_path)
    retriever = build_retriever(tmp_path, embedder=embedder)
    issue_matches, doc_matches = retriever.retrieve("crash on empty frame", k_issues=3, k_docs=1)
    assert issue_matches[0].id == "x/y#1"
    assert doc_matches[0].metadata["path"] == "CONTRIBUTING.md"


def test_extract_json_strips_fences():
    assert extract_json("```json\n{\"a\": 1}\n```") == '{"a": 1}'
    assert extract_json('{"a": 1}') == '{"a": 1}'

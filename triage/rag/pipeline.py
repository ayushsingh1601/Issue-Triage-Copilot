"""Vanilla RAG pipeline: retrieval plus one-shot agent, no orchestration."""
from __future__ import annotations

from pathlib import Path

from triage.agents.vanilla import VanillaAgent
from triage.guardrails.schema import TriageDecision
from triage.rag.embed import Embedder
from triage.rag.retriever import Retriever
from triage.rag.store import ChromaStore


def build_retriever(indexes_dir: Path, embedder: Embedder | None = None) -> Retriever:
    embedder = embedder or Embedder()
    return Retriever(
        embedder=embedder,
        issue_store=ChromaStore(indexes_dir / "issues", "issues"),
        doc_store=ChromaStore(indexes_dir / "docs", "docs"),
    )


class VanillaPipeline:
    def __init__(self, retriever: Retriever, agent: VanillaAgent) -> None:
        self._retriever = retriever
        self._agent = agent

    def run(
        self,
        query: str,
        issue_id: str,
        k_issues: int = 5,
        k_docs: int = 3,
    ) -> TriageDecision:
        issue_matches, doc_matches = self._retriever.retrieve(
            query, k_issues=k_issues, k_docs=k_docs
        )
        return self._agent.run(query, issue_id, issue_matches, doc_matches)

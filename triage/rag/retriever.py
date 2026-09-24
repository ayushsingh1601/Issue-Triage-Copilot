"""Dual-index retrieval for the vanilla RAG baseline."""
from __future__ import annotations

from triage.rag.embed import Embedder
from triage.rag.store import ChromaStore, Match


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        issue_store: ChromaStore,
        doc_store: ChromaStore,
    ) -> None:
        self._embedder = embedder
        self._issue_store = issue_store
        self._doc_store = doc_store

    def retrieve(
        self,
        query: str,
        k_issues: int = 5,
        k_docs: int = 3,
    ) -> tuple[list[Match], list[Match]]:
        vector = self._embedder.embed(query)
        issue_matches = self._issue_store.query(vector, k=k_issues)
        doc_matches = self._doc_store.query(vector, k=k_docs)
        return issue_matches, doc_matches

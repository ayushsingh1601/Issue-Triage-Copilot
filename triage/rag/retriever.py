"""Dual-index retrieval for the vanilla RAG baseline, with optional query
rewriting and LLM re-ranking."""
from __future__ import annotations

from triage.rag.embed import Embedder
from triage.rag.rerank import Reranker
from triage.rag.rewrite import QueryRewriter
from triage.rag.store import ChromaStore, Match


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        issue_store: ChromaStore,
        doc_store: ChromaStore,
        rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._issue_store = issue_store
        self._doc_store = doc_store
        self._rewriter = rewriter
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        k_issues: int = 5,
        k_docs: int = 3,
        candidate_issues: int = 15,
        candidate_docs: int = 10,
    ) -> tuple[list[Match], list[Match]]:
        search_text = self._rewriter.rewrite(query) if self._rewriter else query
        vector = self._embedder.embed(search_text)
        issue_candidates = self._issue_store.query(
            vector, k=max(k_issues, candidate_issues)
        )
        doc_candidates = self._doc_store.query(vector, k=max(k_docs, candidate_docs))
        issue_matches = self._rerank(query, issue_candidates, k_issues)
        doc_matches = self._rerank(query, doc_candidates, k_docs)
        return issue_matches, doc_matches

    def _rerank(
        self,
        query: str,
        candidates: list[Match],
        top_k: int,
    ) -> list[Match]:
        if self._reranker is None:
            return candidates[:top_k]
        return self._reranker.rerank(query, candidates, top_k)

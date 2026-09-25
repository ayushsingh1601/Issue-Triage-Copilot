"""Five read-only MCP tools over the local corpus."""
from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from triage.jsonutil import extract_json
from triage.persist import load_records
from triage.prompts.classify import build_classify_prompt
from triage.rag.embed import Embedder
from triage.rag.parse import IssueRecord
from triage.rag.store import ChromaStore

DEFAULT_ISSUE_TYPES = ["bug", "enhancement", "documentation", "question", "maintenance"]

RespondFn = Callable[[str], str]


def default_llm_respond() -> RespondFn:
    from langchain_openai import ChatOpenAI

    model = os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    return lambda prompt: llm.invoke(prompt).content


class TriageTools:
    def __init__(
        self,
        indexes_dir: Path | str,
        processed_dir: Path | str,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        llm_respond: RespondFn | None = None,
    ) -> None:
        indexes_dir = Path(indexes_dir)
        self._issue_store = ChromaStore(indexes_dir / "issues", "issues")
        self._doc_store = ChromaStore(indexes_dir / "docs", "docs")
        self._records: dict[str, IssueRecord] = {
            f"{r.repo}#{r.number}": r
            for r in load_records(Path(processed_dir) / "issues_corpus.json", IssueRecord)
        }
        self._embedder = Embedder(embed_fn=embed_fn)
        self._llm_respond = llm_respond or default_llm_respond()

    def classify_issue(self, issue: str) -> dict[str, Any]:
        types = self._label_types()
        prompt = build_classify_prompt(issue, types)
        raw = self._llm_respond(prompt)
        data = json.loads(extract_json(raw))
        return {"type": data["type"], "confidence": float(data["confidence"])}

    def search_past_issues(
        self,
        query: str,
        repo: str | None = None,
        issue_type: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        where = {"repo": repo} if repo else None
        matches = self._issue_store.query(
            self._embedder.embed(query), k=max(limit * 3, 10), where=where
        )
        results: list[dict[str, Any]] = []
        for match in matches:
            record = self._records.get(match.id)
            if record is None:
                continue
            if issue_type and issue_type not in record.labels:
                continue
            closing = [
                c.body
                for c in record.comments
                if c.author_association in ("OWNER", "MEMBER", "COLLABORATOR")
            ]
            results.append(
                {
                    "issue_id": match.id,
                    "title": record.title,
                    "labels": record.labels,
                    "linked_prs": record.linked_prs,
                    "closed_at": record.closed_at,
                    "closing_comments": closing[-2:],
                    "distance": match.distance,
                }
            )
            if len(results) >= limit:
                break
        return results

    def get_issue_details(self, issue_id: str) -> dict[str, Any] | None:
        record = self._records.get(issue_id)
        if record is None:
            return None
        return {
            "issue_id": f"{record.repo}#{record.number}",
            "repo": record.repo,
            "number": record.number,
            "title": record.title,
            "body": record.body,
            "state": record.state,
            "closed_at": record.closed_at,
            "labels": record.labels,
            "linked_prs": record.linked_prs,
            "comments": [
                {
                    "author": comment.author,
                    "author_association": comment.author_association,
                    "body": comment.body,
                }
                for comment in record.comments
            ],
        }

    def get_runbook_steps(self, issue_type: str, query: str = "") -> list[dict[str, Any]]:
        text = f"{issue_type}: {query}" if query else issue_type
        matches = self._doc_store.query(self._embedder.embed(text), k=4)
        return [
            {
                "step": match.text,
                "source": match.id,
                "path": match.metadata.get("path", ""),
                "heading": match.metadata.get("heading_path", ""),
            }
            for match in matches
        ]

    def get_resolution_patterns(self, issue_type: str, repo: str | None = None) -> dict[str, Any]:
        records = [r for r in self._records.values() if issue_type in r.labels]
        if repo:
            records = [r for r in records if r.repo == repo]
        if not records:
            return {
                "issue_type": issue_type,
                "count": 0,
                "co_occurring_labels": [],
                "pr_link_rate": 0.0,
                "median_comments": 0,
                "first_response_associations": {},
            }
        co_labels = Counter(label for r in records for label in r.labels if label != issue_type)
        pr_link_rate = sum(bool(r.linked_prs) for r in records) / len(records)
        comment_counts = sorted(len(r.comments) for r in records)
        first_response = Counter(
            r.comments[0].author_association for r in records if r.comments
        )
        return {
            "issue_type": issue_type,
            "count": len(records),
            "co_occurring_labels": [label for label, _ in co_labels.most_common(5)],
            "pr_link_rate": round(pr_link_rate, 3),
            "median_comments": comment_counts[len(comment_counts) // 2],
            "first_response_associations": dict(first_response),
        }

    def _label_types(self, limit: int = 6) -> list[str]:
        counter = Counter(label for r in self._records.values() for label in r.labels)
        return [label for label, _ in counter.most_common(limit)] or DEFAULT_ISSUE_TYPES

    def known_issue_ids(self) -> set[str]:
        return set(self._records)

    def known_doc_ids(self) -> set[str]:
        return set(self._doc_store.ids())

    def retriever(self):
        from triage.rag.retriever import Retriever

        return Retriever(
            embedder=self._embedder,
            issue_store=self._issue_store,
            doc_store=self._doc_store,
        )

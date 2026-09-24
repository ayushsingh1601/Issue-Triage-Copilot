"""Normalize fetched GitHub data into Pydantic models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from triage.fetch import FetchedDocs, FetchedIssue


class IssueComment(BaseModel):
    id: int
    author: str
    author_association: str
    body: str
    created_at: str


class IssueRecord(BaseModel):
    repo: str
    number: int
    title: str
    body: str
    state: str
    author: str
    created_at: str
    closed_at: str | None
    labels: list[str]
    linked_prs: list[int]
    comments: list[IssueComment]


class ProcessDoc(BaseModel):
    repo: str
    path: str
    content: str


def parse_issue(fetched: FetchedIssue) -> IssueRecord:
    issue = fetched.issue
    comments = [
        IssueComment(
            id=comment["id"],
            author=comment.get("user", {}).get("login", ""),
            author_association=comment.get("author_association", ""),
            body=comment.get("body", ""),
            created_at=comment.get("created_at", ""),
        )
        for comment in fetched.comments
        if not _is_bot_comment(comment)
    ]
    return IssueRecord(
        repo=fetched.repo,
        number=fetched.number,
        title=issue.get("title", ""),
        body=issue.get("body") or "",
        state=issue.get("state", "closed"),
        author=issue.get("user", {}).get("login", ""),
        created_at=issue.get("created_at", ""),
        closed_at=issue.get("closed_at"),
        labels=fetched.labels,
        linked_prs=fetched.linked_prs,
        comments=comments,
    )


def parse_docs(fetched: FetchedDocs) -> list[ProcessDoc]:
    return [
        ProcessDoc(repo=fetched.repo, path=doc["path"], content=doc["content"])
        for doc in fetched.docs
    ]


def _is_bot_comment(comment: dict[str, Any]) -> bool:
    user = comment.get("user", {})
    if user.get("type") == "Bot":
        return True
    return str(user.get("login", "")).endswith("[bot]")

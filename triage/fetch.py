"""Fetch per-issue comments, labels, linked PRs, and repo process docs."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

from triage.github import GitHubClient


@dataclass
class FetchedIssue:
    repo: str
    issue: dict[str, Any]
    comments: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def number(self) -> int:
        return int(self.issue["number"])

    @property
    def labels(self) -> list[str]:
        return [label["name"] for label in self.issue.get("labels", [])]

    @property
    def linked_prs(self) -> list[int]:
        return linked_prs(self.events)


@dataclass
class FetchedDocs:
    repo: str
    docs: list[dict[str, Any]]


def fetch_issues(
    gh: GitHubClient, repo: str, state: str = "closed", limit: int | None = None
) -> list[FetchedIssue]:
    issues: list[FetchedIssue] = []
    for issue in gh.list_issues(repo, state=state):
        if "pull_request" in issue:
            continue
        issues.append(fetch_issue_details(gh, repo, issue))
        if limit is not None and len(issues) >= limit:
            break
    return issues


def fetch_issue_details(
    gh: GitHubClient, repo: str, issue: dict[str, Any]
) -> FetchedIssue:
    number = issue["number"]
    comments = list(gh.list_issue_comments(repo, number))
    events = list(gh.paginate(f"/repos/{repo}/issues/{number}/events"))
    return FetchedIssue(repo=repo, issue=issue, comments=comments, events=events)


def linked_prs(events: list[dict[str, Any]]) -> list[int]:
    prs: set[int] = set()
    for event in events:
        source = event.get("source", {}).get("issue")
        if source and source.get("pull_request"):
            prs.add(int(source["number"]))
    return sorted(prs)


def fetch_process_docs(gh: GitHubClient, repo: str) -> FetchedDocs:
    tree = gh.get_json(f"/repos/{repo}/git/trees/HEAD", {"recursive": "1"})
    paths = [entry["path"] for entry in tree["tree"] if entry.get("type") == "blob"]
    docs: list[dict[str, Any]] = []
    for path in paths:
        if not _is_doc_path(path):
            continue
        item = gh.get_json(f"/repos/{repo}/contents/{path}")
        content = item.get("content")
        if content is None:
            continue
        text = base64.b64decode(content).decode("utf-8", errors="replace")
        docs.append({"path": path, "content": text})
    return FetchedDocs(repo=repo, docs=docs)


def _is_doc_path(path: str) -> bool:
    lower = path.lower()
    if lower.endswith("contributing.md"):
        return True
    if lower.startswith("docs/"):
        return True
    if "issue_template" in lower or "pull_request_template" in lower:
        return True
    return lower.endswith(".md") and "release" in lower

import json
from pathlib import Path
from typing import Any

from triage.fetch import FetchedDocs, FetchedIssue
from triage.rag.parse import IssueRecord, ProcessDoc, parse_docs, parse_issue

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def fetched_issue(**kwargs: Any) -> FetchedIssue:
    defaults = dict(
        repo="x/y",
        issue=load("issue.json"),
        comments=load("comments.json"),
        events=load("events.json"),
    )
    defaults.update(kwargs)
    return FetchedIssue(**defaults)


def test_parse_issue():
    record = parse_issue(fetched_issue())
    assert isinstance(record, IssueRecord)
    assert record.repo == "x/y"
    assert record.number == 42
    assert record.title == "Regression: crash on empty DataFrame"
    assert "Traceback" in record.body
    assert record.author == "octocat"
    assert record.state == "closed"
    assert record.closed_at == "2024-01-05T00:00:00Z"
    assert record.labels == ["bug", "DataFrame"]
    assert record.linked_prs == [45]
    assert len(record.comments) == 1
    assert record.comments[0].author_association == "COLLABORATOR"


def test_parse_issue_drops_bot_comments():
    comments = [
        {
            "id": 1,
            "user": {"login": "alice"},
            "author_association": "NONE",
            "body": "me too",
            "created_at": "2024-01-01T00:00:00Z",
        },
        {
            "id": 2,
            "user": {"login": "some-bot", "type": "Bot"},
            "author_association": "NONE",
            "body": "closing as stale",
            "created_at": "2024-01-01T00:00:00Z",
        },
        {
            "id": 3,
            "user": {"login": "renovate[bot]"},
            "author_association": "NONE",
            "body": "update",
            "created_at": "2024-01-01T00:00:00Z",
        },
    ]
    record = parse_issue(fetched_issue(comments=comments))
    assert [c.id for c in record.comments] == [1]


def test_parse_issue_null_body_becomes_empty():
    record = parse_issue(fetched_issue(issue={"number": 1, "title": "t", "state": "closed"}))
    assert record.body == ""
    assert record.number == 1


def test_parse_docs():
    fetched = FetchedDocs(
        repo="x/y",
        docs=[
            {"path": "CONTRIBUTING.md", "content": "hello"},
            {"path": "RELEASE.md", "content": "world"},
        ],
    )
    parsed = parse_docs(fetched)
    assert parsed == [
        ProcessDoc(repo="x/y", path="CONTRIBUTING.md", content="hello"),
        ProcessDoc(repo="x/y", path="RELEASE.md", content="world"),
    ]

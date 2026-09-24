import base64
import json
from pathlib import Path
from typing import Any

import httpx
from triage.fetch import fetch_issue_details, fetch_issues, fetch_process_docs, linked_prs
from triage.github import GitHubClient

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def ok(data: Any) -> httpx.Response:
    return httpx.Response(200, json=data, headers={"X-RateLimit-Remaining": "500"})


def make_client(handler) -> GitHubClient:
    return GitHubClient(
        token="t", retry_buffer=0.0, retry_backoff=0.0, transport=httpx.MockTransport(handler)
    )


def issues_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "/comments" in url:
        return ok(load("comments.json"))
    if "/events" in url:
        return ok(load("events.json"))
    if "/issues" in url:
        return ok([load("pr.json"), load("issue.json")])
    return httpx.Response(404, json={})


def test_fetch_issue_details():
    client = make_client(issues_handler)
    fetched = fetch_issue_details(client, "x/y", load("issue.json"))
    assert fetched.number == 42
    assert fetched.labels == ["bug", "DataFrame"]
    assert [c["author_association"] for c in fetched.comments] == ["COLLABORATOR", "MEMBER"]
    assert fetched.linked_prs == [45]


def test_linked_prs_deduplicates_and_ignores_plain_issues():
    assert linked_prs(load("events.json")) == [45]


def test_linked_prs_empty_for_plain_issue_references():
    events = [{"event": "cross-referenced", "source": {"issue": {"number": 99}}}]
    assert linked_prs(events) == []


def test_fetch_issues_skips_prs_and_honors_limit():
    client = make_client(issues_handler)
    fetched = fetch_issues(client, "x/y", state="closed", limit=1)
    assert len(fetched) == 1
    assert fetched[0].number == 42
    assert fetched[0].linked_prs == [45]


def test_fetch_process_docs_selects_wanted_paths():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/contents/" in url:
            path = url.split("/contents/", 1)[1]
            encoded = base64.b64encode(f"content of {path}".encode()).decode()
            return ok({"path": path, "content": encoded})
        return ok(load("tree.json"))

    docs = fetch_process_docs(make_client(handler), "x/y")
    assert docs.repo == "x/y"
    assert {d["path"] for d in docs.docs} == {
        "CONTRIBUTING.md",
        "docs/contributing_guide.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/PULL_REQUEST_TEMPLATE.md",
        "RELEASE.md",
    }
    contributing = next(d for d in docs.docs if d["path"] == "CONTRIBUTING.md")
    assert contributing["content"] == "content of CONTRIBUTING.md"


def test_fetch_process_docs_skips_null_content():
    def handler(request: httpx.Request) -> httpx.Response:
        if "/contents/" in str(request.url):
            return ok({"path": "docs/x.md", "content": None})
        return ok({"tree": [{"path": "docs/x.md", "type": "blob"}]})

    docs = fetch_process_docs(make_client(handler), "x/y")
    assert docs.docs == []

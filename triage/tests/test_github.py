from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from triage.github import GitHubClient, GitHubError


def client(handler, **kwargs) -> GitHubClient:
    kwargs.setdefault("transport", httpx.MockTransport(handler))
    kwargs.setdefault("token", "test-token")
    return GitHubClient(retry_buffer=0.0, retry_backoff=0.0, **kwargs)


def has_page(url: str) -> bool:
    return "page" in parse_qs(urlsplit(url).query)


def ok_json(data: Any, **headers: str) -> httpx.Response:
    return httpx.Response(200, json=data, headers={"X-RateLimit-Remaining": "500", **headers})


def test_auth_header_from_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "env-secret")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        return ok_json({"number": 1})

    GitHubClient(transport=httpx.MockTransport(handler)).get_issue("x/y", 1)
    assert captured["auth"] == "Bearer env-secret"


def test_explicit_token_overrides_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "env-secret")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        return ok_json([])

    list(client(handler, token="explicit").list_issues("x/y"))
    assert captured["auth"] == "Bearer explicit"


def test_get_issue_returns_json():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.github.com/repos/x/y/issues/42"
        return ok_json({"number": 42, "title": "bug"})

    result = client(handler).get_issue("x/y", 42)
    assert result == {"number": 42, "title": "bug"}


def test_pagination_follows_next_link():
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        if not has_page(str(request.url)):
            return ok_json(
                [{"id": 1}, {"id": 2}],
                Link='<https://api.github.com/repos/x/y/issues?page=2&per_page=2>; rel="next"',
            )
        return ok_json([{"id": 3}])

    items = list(client(handler).list_issues("x/y", per_page=2))
    assert [i["id"] for i in items] == [1, 2, 3]
    assert len(urls) == 2


def test_pagination_single_page_without_link():
    def handler(request: httpx.Request) -> httpx.Response:
        return ok_json([{"id": 1}])

    items = list(client(handler).list_issues("x/y"))
    assert items == [{"id": 1}]


def test_issue_comments_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("https://api.github.com/repos/x/y/issues/7/comments")
        return ok_json([{"body": "hi"}])

    comments = list(client(handler).list_issue_comments("x/y", 7))
    assert comments == [{"body": "hi"}]


def test_rate_limit_waits_and_retries():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                403,
                json={"message": "API rate limit exceeded"},
                headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "0"},
            )
        return ok_json([{"id": 1}])

    items = list(client(handler).list_issues("x/y"))
    assert items == [{"id": 1}]
    assert calls["n"] == 2


def test_retries_on_server_error():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, json={})
        return ok_json([{"id": 1}])

    items = list(client(handler).list_issues("x/y"))
    assert items == [{"id": 1}]
    assert calls["n"] == 2


def test_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    with pytest.raises(GitHubError):
        client(handler).get_issue("x/y", 42)


def test_context_manager_closes():
    class CountingTransport(httpx.MockTransport):
        closed = False

        def close(self) -> None:
            self.closed = True
            super().close()

    transport = CountingTransport(lambda request: ok_json([]))
    with GitHubClient(token="t", transport=transport) as gh:
        list(gh.list_issues("x/y"))
    assert transport.closed

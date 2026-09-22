"""Minimal GitHub REST API client: env auth, pagination, rate-limit handling."""
from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import httpx

GITHUB_API = "https://api.github.com"
MAX_RETRIES = 5


class GitHubError(RuntimeError):
    pass


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        base_url: str = GITHUB_API,
        transport: httpx.BaseTransport | None = None,
        retry_buffer: float = 1.0,
        retry_backoff: float = 1.0,
    ) -> None:
        self._token = token or os.environ["GITHUB_TOKEN"]
        self._base_url = base_url.rstrip("/")
        self._retry_buffer = retry_buffer
        self._retry_backoff = retry_backoff
        self._client = httpx.Client(
            base_url=self._base_url,
            transport=transport,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=httpx.Timeout(30.0),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._request("GET", path, params=params)
        return resp.json()

    def paginate(self, path: str, params: dict[str, Any] | None = None) -> Iterator[Any]:
        url: str | None = f"{self._base_url}{path}"
        params = dict(params or {})
        while url is not None:
            resp = self._request("GET", url, params=params)
            yield from resp.json()
            url = _next_url(resp)
            params = None

    def list_issues(
        self, repo: str, state: str = "all", per_page: int = 100
    ) -> Iterator[dict[str, Any]]:
        return self.paginate(f"/repos/{repo}/issues", {"state": state, "per_page": per_page})

    def get_issue(self, repo: str, number: int) -> dict[str, Any]:
        return self.get_json(f"/repos/{repo}/issues/{number}")

    def list_issue_comments(self, repo: str, number: int) -> Iterator[dict[str, Any]]:
        return self.paginate(f"/repos/{repo}/issues/{number}/comments", {"per_page": 100})

    def _request(
        self, method: str, url: str, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        resp: httpx.Response | None = None
        for attempt in range(MAX_RETRIES):
            resp = self._client.request(method, url, params=params)
            if self._is_rate_limited(resp):
                self._wait_for_rate_limit(resp)
                continue
            if resp.status_code >= 500 and attempt < MAX_RETRIES - 1:
                time.sleep(self._retry_backoff * (2**attempt))
                continue
            break
        assert resp is not None
        if resp.status_code >= 400:
            raise GitHubError(f"{method} {url} failed: {resp.status_code} {resp.text[:200]}")
        return resp

    def _is_rate_limited(self, resp: httpx.Response) -> bool:
        if resp.status_code not in (403, 429):
            return False
        if resp.headers.get("X-RateLimit-Remaining") == "0":
            return True
        return "rate limit" in resp.text.lower()

    def _wait_for_rate_limit(self, resp: httpx.Response) -> None:
        reset = int(resp.headers.get("X-RateLimit-Reset", 0))
        delay = max(0, reset - time.time()) + self._retry_buffer
        time.sleep(delay)


def _next_url(resp: httpx.Response) -> str | None:
    link = resp.headers.get("Link")
    if not link:
        return None
    for part in link.split(","):
        url, rel = part.split(";")
        if 'rel="next"' in rel:
            return url.strip(" <>")
    return None

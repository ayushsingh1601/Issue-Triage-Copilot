"""Tests for webapp.repo_service: per-repo cache, locking, and status handling."""
from __future__ import annotations

import os
import threading
import time

import pytest
import webapp.repo_service as repo_service

from triage.github import GitHubError


class _RaiseOnUse:
    def __init__(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_issue(self, repo, number):
        raise GitHubError("no network in tests")


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(repo_service, "DATA_DIR", tmp_path / "data")
    yield


def test_repo_key_and_paths():
    paths = repo_service.repo_paths("owner/name")
    assert paths.key == "owner__name"
    assert str(paths.indexes).endswith("owner__name/indexes")


def test_ensure_repo_builds_once_then_serves_cache(monkeypatch):
    calls: list[tuple[str, int]] = []

    def fake_build(repo, paths, progress=None, cap=500):
        calls.append((repo, cap))
        repo_service._write_status(
            paths.status, repo_service.READY, message="ready", progress=1.0
        )

    monkeypatch.setattr(repo_service, "build_repo", fake_build)

    paths = repo_service.ensure_repo("a/b", cap=7)
    assert calls == [("a/b", 7)]
    assert not paths.lock.exists()

    repo_service.ensure_repo("a/b")
    assert calls == [("a/b", 7)]
    assert repo_service.is_ready("a/b")


def test_ensure_repo_waits_for_concurrent_build(monkeypatch):
    monkeypatch.setattr(repo_service.time, "sleep", lambda _: None)
    paths = repo_service.repo_paths("c/d")
    paths.root.mkdir(parents=True)
    fd = os.open(paths.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    repo_service._write_status(paths.status, repo_service.BUILDING, message="building")

    calls: list = []
    monkeypatch.setattr(repo_service, "build_repo", lambda *args, **kwargs: calls.append(args))

    results: list[repo_service.RepoPaths] = []

    def worker() -> None:
        results.append(repo_service.ensure_repo("c/d"))

    thread = threading.Thread(target=worker)
    thread.start()
    time.sleep(0.1)
    assert calls == []  # still waiting on the other build

    repo_service._write_status(paths.status, repo_service.READY, message="ready")
    repo_service._release(paths)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert calls == []
    assert results[0].key == "c__d"


def test_ensure_repo_records_failure(monkeypatch):
    def bad_build(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(repo_service, "build_repo", bad_build)

    paths = repo_service.repo_paths("e/f")
    with pytest.raises(RuntimeError):
        repo_service.ensure_repo("e/f")

    status = repo_service._read_status(paths.status)
    assert status["state"] == repo_service.ERROR
    assert "boom" in status["error"]
    assert not paths.lock.exists()


def test_find_issue_reads_cached_records(tmp_path, monkeypatch):
    from triage.persist import save_records
    from triage.rag.parse import IssueRecord

    paths = repo_service.repo_paths("g/h")
    record = IssueRecord(
        repo="g/h",
        number=42,
        title="Broken widget",
        body="It crashes.",
        state="closed",
        author="alice",
        created_at="2026-01-01T00:00:00Z",
        closed_at="2026-01-02T00:00:00Z",
        labels=["bug"],
        linked_prs=[100],
        comments=[],
    )
    save_records([record], paths.processed / "issues_corpus.json")

    assert repo_service.find_issue("g/h", 42) == record
    monkeypatch.setattr(repo_service, "GitHubClient", _RaiseOnUse)
    assert repo_service.find_issue("g/h", 43) is None


def test_seed_and_build_is_offline(tmp_path, monkeypatch):
    from triage.persist import save_records
    from triage.rag.parse import IssueRecord, ProcessDoc

    key = "seed/repo"
    paths = repo_service.repo_paths(key)
    seed = tmp_path / "seed" / paths.key
    (seed / "processed").mkdir(parents=True)
    (seed / "indexes").mkdir()
    (seed / "indexes" / "embeddings.json").write_text("{}")

    record = IssueRecord(
        repo="seed/repo",
        number=1,
        title="T",
        body="B",
        state="closed",
        author="a",
        created_at="2026-01-01T00:00:00Z",
        closed_at="2026-01-02T00:00:00Z",
        labels=["bug"],
        linked_prs=[],
        comments=[],
    )
    save_records([record], seed / "processed" / "issues_corpus.json")
    save_records([], seed / "processed" / "issues_held_out.json")
    save_records(
        [ProcessDoc(repo="seed/repo", path="CONTRIBUTING.md", content="How to triage")],
        seed / "processed" / "process_docs.json",
    )

    calls: list = []

    class FakeEmbedder:
        def __init__(self, cache_path=None):
            calls.append("embedder")

        def embed_many(self, texts):
            return [[0.1, 0.2]] * len(texts)

    monkeypatch.setattr(repo_service, "SEED_DIR", tmp_path / "seed")
    monkeypatch.setattr(repo_service, "Embedder", FakeEmbedder)

    repo_service.seed_and_build(paths)

    status = repo_service._read_status(paths.status)
    assert status["state"] == repo_service.READY
    assert status.get("seeded") is True
    assert (paths.indexes / "issues").is_dir()
    assert (paths.indexes / "docs").is_dir()
    assert repo_service.is_ready(key)
    assert "embedder" in calls

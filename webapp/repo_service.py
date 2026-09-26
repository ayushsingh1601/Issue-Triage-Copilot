"""Per-repo dataset ingestion and caching for the hosted app.

Each repo gets its own processed dataset and Chroma indexes under
``$DATA_DIR`` (default ``./webapp/data``). Repos are built once on first
triage and served from cache afterwards. A lock file guards builds so
concurrent sessions wait instead of rebuilding.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triage.evals.dataset import held_out_split
from triage.fetch import FetchedIssue, fetch_issues, fetch_process_docs
from triage.github import GitHubClient, GitHubError
from triage.persist import load_records, save_records
from triage.rag.embed import Embedder
from triage.rag.index_build import build_doc_index, build_issue_index
from triage.rag.parse import IssueRecord, ProcessDoc, parse_docs, parse_issue

DATA_DIR = Path(os.environ.get("DATA_DIR", "webapp/data"))
SEED_DIR = Path(__file__).resolve().parent / "data" / "seed"
MAX_ISSUES = int(os.environ.get("MAX_ISSUES", "500"))
BUILD_TIMEOUT_S = 30 * 60
STALE_LOCK_S = 30 * 60

BUILDING = "building"
READY = "ready"
ERROR = "error"

ProgressFn = Callable[[str, float | None], None]


@dataclass(frozen=True)
class RepoPaths:
    key: str
    root: Path
    processed: Path
    indexes: Path
    embeddings: Path
    status: Path
    lock: Path


def repo_key(repo: str) -> str:
    return repo.replace("/", "__")


def repo_paths(repo: str) -> RepoPaths:
    key = repo_key(repo)
    root = DATA_DIR / "repos" / key
    return RepoPaths(
        key=key,
        root=root,
        processed=root / "processed",
        indexes=root / "indexes",
        embeddings=root / "indexes" / "embeddings.json",
        status=root / "status.json",
        lock=root / ".lock",
    )


def validate_repo(repo: str) -> None:
    """Raise if the repo is not a usable public repo."""
    with GitHubClient() as gh:
        meta = gh.get_json(f"/repos/{repo}")
    if meta.get("private"):
        raise ValueError("private repos are not supported")
    if meta.get("archived"):
        raise ValueError("archived repos are not supported")
    if not meta.get("permissions", {}).get("pull"):
        raise ValueError("repo is not pull-accessible")


def is_ready(repo: str) -> bool:
    return _is_ready(repo_paths(repo))


def ensure_repo(
    repo: str,
    progress: ProgressFn | None = None,
    cap: int = MAX_ISSUES,
) -> RepoPaths:
    """Return paths for a ready repo, building it (once) if needed."""
    progress = progress or _noop
    paths = repo_paths(repo)
    if _is_ready(paths):
        return paths
    _acquire_or_wait(paths, progress)
    try:
        if _is_ready(paths):
            return paths
        if _seed_dir_for(paths.key):
            seed_and_build(paths, progress=progress)
        else:
            build_repo(repo, paths, progress=progress, cap=cap)
    except Exception as exc:
        _write_status(paths.status, ERROR, error=str(exc))
        raise
    finally:
        _release(paths)
    return paths


def build_repo(
    repo: str,
    paths: RepoPaths,
    progress: ProgressFn | None = None,
    cap: int = MAX_ISSUES,
) -> None:
    progress = progress or _noop
    _write_status(paths.status, BUILDING, message="fetching…", progress=0.0)
    progress("Fetching closed issues and process docs…", 0.1)

    with GitHubClient() as gh:
        fetched = fetch_issues(gh, repo, state="closed", limit=cap)
        if not fetched:
            raise ValueError(f"no closed issues found for {repo}")
        docs = parse_docs(fetch_process_docs(gh, repo))
    records = [parse_issue(item) for item in fetched]

    progress("Splitting out a held-out eval set…", 0.4)
    split = held_out_split(records)
    paths.processed.mkdir(parents=True, exist_ok=True)
    save_records(split.corpus, paths.processed / "issues_corpus.json")
    save_records(split.held_out, paths.processed / "issues_held_out.json")
    save_records(docs, paths.processed / "process_docs.json")

    progress("Building the RAG index…", 0.5)
    embedder = Embedder(cache_path=paths.embeddings)
    build_issue_index(split.corpus, embedder, paths.indexes)
    build_doc_index(docs, embedder, paths.indexes)

    _write_status(
        paths.status,
        READY,
        message="ready",
        progress=1.0,
        corpus=len(split.corpus),
        held_out=len(split.held_out),
        docs=len(docs),
    )
    progress("Done.", 1.0)


def seed_and_build(paths: RepoPaths, progress: ProgressFn | None = None) -> None:
    """Seed a committed dataset into the cache and build indexes offline.

    Free hosts have ephemeral disks, so ``webapp/data/seed`` ships a small
    prebuilt corpus + embedding cache. Seeding avoids a GitHub fetch and any
    embedding spend on first run (all vectors come from the cache).
    """
    progress = progress or _noop
    src = _seed_dir_for(paths.key)
    assert src is not None, f"no seed for {paths.key}"
    _write_status(paths.status, BUILDING, message="seeding…", progress=0.0)

    progress("Seeding the prebuilt dataset…", 0.2)
    paths.indexes.mkdir(parents=True, exist_ok=True)
    if not (paths.indexes / "embeddings.json").exists():
        shutil.copy(src / "indexes" / "embeddings.json", paths.indexes / "embeddings.json")
    if not (paths.processed / "issues_corpus.json").exists():
        shutil.copytree(src / "processed", paths.processed, dirs_exist_ok=True)

    progress("Building the RAG index…", 0.6)
    embedder = Embedder(cache_path=paths.embeddings)
    corpus = load_records(paths.processed / "issues_corpus.json", IssueRecord)
    docs = load_records(paths.processed / "process_docs.json", ProcessDoc)
    build_issue_index(corpus, embedder, paths.indexes)
    build_doc_index(docs, embedder, paths.indexes)

    held_out = load_records(paths.processed / "issues_held_out.json", IssueRecord)
    _write_status(
        paths.status,
        READY,
        message="ready",
        progress=1.0,
        corpus=len(corpus),
        held_out=len(held_out),
        docs=len(docs),
        seeded=True,
    )
    progress("Done.", 1.0)


def find_issue(repo: str, number: int) -> IssueRecord | None:
    """Resolve an issue number from the cache, else fetch it live."""
    paths = repo_paths(repo)
    for filename in ("issues_corpus.json", "issues_held_out.json"):
        file_path = paths.processed / filename
        if not file_path.exists():
            continue
        for record in load_records(file_path, IssueRecord):
            if record.number == number:
                return record
    with GitHubClient() as gh:
        try:
            issue = gh.get_issue(repo, number)
        except GitHubError:
            return None
        if "pull_request" in issue:
            return None
        fetched = FetchedIssue(
            repo=repo,
            issue=issue,
            comments=list(gh.list_issue_comments(repo, number)),
            events=list(gh.paginate(f"/repos/{repo}/issues/{number}/events")),
        )
    return parse_issue(fetched)


def _is_ready(paths: RepoPaths) -> bool:
    status = _read_status(paths.status)
    return bool(status and status.get("state") == READY)


def _seed_dir_for(key: str) -> Path | None:
    path = SEED_DIR / key
    return path if path.is_dir() else None


def _acquire_or_wait(paths: RepoPaths, progress: ProgressFn) -> None:
    waited = 0.0
    while True:
        if _acquire(paths):
            return
        if _is_ready(paths):
            return
        if waited >= BUILD_TIMEOUT_S:
            raise TimeoutError(f"timed out waiting to build {paths.key}")
        progress("Another session is building this repo…", None)
        time.sleep(3)
        waited += 3.0


def _acquire(paths: RepoPaths) -> bool:
    paths.root.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(paths.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        if _lock_is_stale(paths):
            paths.lock.unlink(missing_ok=True)
            return _acquire(paths)
        return False


def _lock_is_stale(paths: RepoPaths) -> bool:
    try:
        return time.time() - paths.lock.stat().st_mtime > STALE_LOCK_S
    except FileNotFoundError:
        return False


def _release(paths: RepoPaths) -> None:
    paths.lock.unlink(missing_ok=True)


def _read_status(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_status(path: Path, state: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "updated_at": time.time(), **extra}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _noop(message: str, pct: float | None) -> None:
    pass

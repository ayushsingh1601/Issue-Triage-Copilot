from triage.rag.embed import Embedder
from triage.rag.index_build import build_issue_index, sync_issue_index
from triage.rag.parse import IssueRecord
from triage.rag.store import ChromaStore


def make_issue(repo: str, number: int) -> IssueRecord:
    return IssueRecord(
        repo=repo,
        number=number,
        title=f"Bug {number}",
        body="crash on empty frame",
        state="closed",
        author="a",
        created_at="2024-01-01",
        closed_at=None,
        labels=["bug"],
        linked_prs=[],
        comments=[],
    )


def counting_embed():
    calls = {"n": 0}

    def embed(texts):
        calls["n"] += len(texts)
        return [[1.0, 0.0] for _ in texts]

    return embed, calls


def test_sync_adds_new_and_removes_evicted(tmp_path):
    embed_fn, calls = counting_embed()
    embedder = Embedder(embed_fn=embed_fn)
    initial = [make_issue("x/y", 1), make_issue("x/y", 2)]
    build_issue_index(initial, embedder, tmp_path / "indexes")

    grown = [make_issue("x/y", 1), make_issue("x/y", 3)]
    stats = sync_issue_index(grown, embedder, tmp_path / "indexes")

    assert stats == {"added": 1, "removed": 1, "unchanged": 1}
    assert set(ChromaStore(tmp_path / "indexes" / "issues", "issues").ids()) == {"x/y#1", "x/y#3"}


def test_sync_embeds_only_new_texts(tmp_path):
    embed_fn, calls = counting_embed()
    embedder = Embedder(embed_fn=embed_fn)
    initial = [make_issue("x/y", 1)]
    build_issue_index(initial, embedder, tmp_path / "indexes")
    calls["n"] = 0

    sync_issue_index([make_issue("x/y", 1), make_issue("x/y", 2)], embedder, tmp_path / "indexes")
    assert calls["n"] == 1


def test_sync_noop_when_in_sync(tmp_path):
    embed_fn, calls = counting_embed()
    embedder = Embedder(embed_fn=embed_fn)
    records = [make_issue("x/y", 1)]
    build_issue_index(records, embedder, tmp_path / "indexes")
    calls["n"] = 0

    stats = sync_issue_index(records, embedder, tmp_path / "indexes")
    assert stats == {"added": 0, "removed": 0, "unchanged": 1}
    assert calls["n"] == 0

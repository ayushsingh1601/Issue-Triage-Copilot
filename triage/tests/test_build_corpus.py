from triage.rag.embed import Embedder
from triage.rag.index_build import build_doc_index, build_issue_index
from triage.rag.parse import IssueComment, IssueRecord, ProcessDoc


def make_issue(repo: str, number: int, body: str = "crash on empty frame") -> IssueRecord:
    return IssueRecord(
        repo=repo,
        number=number,
        title=f"Bug {number}",
        body=body,
        state="closed",
        author="a",
        created_at="2024-01-01",
        closed_at="2024-01-02",
        labels=["bug"],
        linked_prs=[],
        comments=[
            IssueComment(
                id=1,
                author="b",
                author_association="COLLABORATOR",
                body="fixed",
                created_at="2024-01-02",
            )
        ],
    )


def fake_embed_fn(texts):
    return [[float(len(t)), 0.0] for t in texts]


def test_build_indexes_excludes_held_out(tmp_path):
    corpus = [make_issue("x/y", i) for i in range(1, 4)]
    docs = [
        ProcessDoc(repo="x/y", path="CONTRIBUTING.md", content="# Intro\nhow to contribute.\n"),
        ProcessDoc(repo="x/y", path="docs/guide.md", content="# Guide\nrelease process.\n"),
    ]
    indexes = tmp_path / "indexes"
    embedder = Embedder(embed_fn=fake_embed_fn)
    issue_store = build_issue_index(corpus, embedder, indexes)
    doc_store = build_doc_index(docs, embedder, indexes)

    issue_ids = set(issue_store.ids())
    assert issue_ids == {"x/y#1", "x/y#2", "x/y#3"}
    assert "x/y#100" not in issue_ids
    assert doc_store.count() > 0


def test_build_indexes_embedded_and_queryable(tmp_path):
    corpus = [make_issue("x/y", 7, body="dataframe merge crash stack trace")]
    embedder = Embedder(embed_fn=fake_embed_fn)
    issue_store = build_issue_index(corpus, embedder, tmp_path / "indexes")
    matches = issue_store.query([10.0, 0.0], k=1)
    assert matches[0].id == "x/y#7"
    assert matches[0].metadata["labels"] == "bug"
    assert matches[0].metadata["number"] == 7


def test_refresh_grows_index_without_duplicate_error(tmp_path):
    from triage.rag.index_build import sync_issue_index
    from triage.rag.store import ChromaStore

    embedder = Embedder(embed_fn=fake_embed_fn)
    build_issue_index([make_issue("x/y", 1), make_issue("x/y", 2)], embedder, tmp_path / "indexes")
    stats = sync_issue_index(
        [make_issue("x/y", 1), make_issue("x/y", 2), make_issue("x/y", 3)],
        embedder,
        tmp_path / "indexes",
    )
    assert stats["added"] == 1
    assert stats["removed"] == 0
    ids = set(ChromaStore(tmp_path / "indexes" / "issues", "issues").ids())
    assert ids == {"x/y#1", "x/y#2", "x/y#3"}

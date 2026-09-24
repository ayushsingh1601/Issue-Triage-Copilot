import importlib.util
from pathlib import Path

from triage.persist import save_records
from triage.rag.embed import Embedder
from triage.rag.parse import IssueComment, IssueRecord, ProcessDoc


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    build = load_script("build_corpus")
    processed = tmp_path / "processed"
    indexes = tmp_path / "indexes"

    corpus = [make_issue("x/y", i) for i in range(1, 4)]
    held_out = [make_issue("x/y", 100)]
    docs = [
        ProcessDoc(repo="x/y", path="CONTRIBUTING.md", content="# Intro\nhow to contribute.\n"),
        ProcessDoc(repo="x/y", path="docs/guide.md", content="# Guide\nrelease process.\n"),
    ]
    save_records(corpus, processed / "issues_corpus.json")
    save_records(held_out, processed / "issues_held_out.json")
    save_records(docs, processed / "process_docs.json")

    embedder = Embedder(embed_fn=fake_embed_fn)
    issue_store = build.build_issue_index(corpus, embedder, indexes)
    doc_store = build.build_doc_index(docs, embedder, indexes)

    issue_ids = set(issue_store.ids())
    assert issue_ids == {"x/y#1", "x/y#2", "x/y#3"}
    assert "x/y#100" not in issue_ids
    assert doc_store.count() > 0


def test_build_indexes_embedded_and_queryable(tmp_path):
    build = load_script("build_corpus")
    corpus = [make_issue("x/y", 7, body="dataframe merge crash stack trace")]
    embedder = Embedder(embed_fn=fake_embed_fn)
    issue_store = build.build_issue_index(corpus, embedder, tmp_path / "indexes")
    matches = issue_store.query([10.0, 0.0], k=1)
    assert matches[0].id == "x/y#7"
    assert matches[0].metadata["labels"] == "bug"
    assert matches[0].metadata["number"] == 7

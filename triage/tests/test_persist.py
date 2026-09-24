import pytest
from triage.persist import load_records, save_records
from triage.rag.parse import IssueComment, IssueRecord, ProcessDoc


def make_issue(number: int, with_comment: bool = False) -> IssueRecord:
    comments = []
    if with_comment:
        comments = [
            IssueComment(
                id=1, author="a", author_association="NONE", body="hello", created_at="2024-01-01"
            )
        ]
    return IssueRecord(
        repo="x/y",
        number=number,
        title="t",
        body="b",
        state="closed",
        author="a",
        created_at="2024-01-01",
        closed_at="2024-01-02",
        labels=["bug"],
        linked_prs=[10],
        comments=comments,
    )


def test_json_round_trip_issues(tmp_path):
    records = [make_issue(i, with_comment=True) for i in range(3)]
    path = tmp_path / "issues.json"
    save_records(records, path)
    assert load_records(path, IssueRecord) == records


def test_json_round_trip_docs(tmp_path):
    docs = [ProcessDoc(repo="x/y", path=f"docs/{i}.md", content=f"content {i}") for i in range(2)]
    path = tmp_path / "docs.json"
    save_records(docs, path)
    assert load_records(path, ProcessDoc) == docs


def test_parquet_round_trip_issues(tmp_path):
    records = [make_issue(i, with_comment=True) for i in range(3)]
    path = tmp_path / "issues.parquet"
    save_records(records, path, fmt="parquet")
    assert load_records(path, IssueRecord) == records


def test_format_inferred_from_suffix(tmp_path):
    records = [make_issue(1)]
    path = tmp_path / "issues.parquet"
    save_records(records, path, fmt="parquet")
    assert load_records(path, IssueRecord) == records


def test_save_creates_parent_dirs(tmp_path):
    records = [make_issue(1)]
    path = tmp_path / "nested" / "deep" / "issues.json"
    save_records(records, path)
    assert path.exists()
    assert load_records(path, IssueRecord) == records


def test_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError):
        save_records([], tmp_path / "x.txt", fmt="yaml")

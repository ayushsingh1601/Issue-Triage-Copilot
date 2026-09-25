import importlib.util
from pathlib import Path

from triage.rag.parse import IssueComment, IssueRecord


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_actual():
    demo = load_script("run_demo")
    record = IssueRecord(
        repo="x/y",
        number=7,
        title="crash",
        body="b",
        state="closed",
        author="a",
        created_at="2024-01-01",
        closed_at="2024-01-02",
        labels=["bug", "DataFrame"],
        linked_prs=[42],
        comments=[
            IssueComment(
                id=1,
                author="m",
                author_association="MEMBER",
                body="fixed by #42",
                created_at="2024-01-02",
            )
        ],
    )
    summary = demo.summarize_actual(record)
    assert summary["issue_id"] == "x/y#7"
    assert summary["actual_labels"] == ["bug", "DataFrame"]
    assert summary["linked_prs"] == [42]
    assert summary["closing_comment"] == "fixed by #42"


def test_summarize_actual_without_comment():
    demo = load_script("run_demo")
    record = IssueRecord(
        repo="x/y",
        number=8,
        title="t",
        body="b",
        state="closed",
        author="a",
        created_at="",
        closed_at=None,
        labels=[],
        linked_prs=[],
        comments=[],
    )
    assert demo.summarize_actual(record)["closing_comment"] is None

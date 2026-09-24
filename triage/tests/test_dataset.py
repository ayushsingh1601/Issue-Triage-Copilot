import pytest
from triage.evals.dataset import held_out_split
from triage.rag.parse import IssueRecord


def make_record(repo: str, number: int) -> IssueRecord:
    return IssueRecord(
        repo=repo,
        number=number,
        title="t",
        body="",
        state="closed",
        author="a",
        created_at="",
        closed_at=None,
        labels=[],
        linked_prs=[],
        comments=[],
    )


def test_held_out_split_disjoint_and_fraction():
    records = [make_record("x/y", i) for i in range(100)]
    split = held_out_split(records)
    split.assert_disjoint()
    assert len(split.corpus) + len(split.held_out) == 100
    assert 15 <= len(split.held_out) <= 20


def test_held_out_split_is_reproducible():
    records = [make_record("x/y", i) for i in range(20)]
    first = held_out_split(records, seed=7)
    second = held_out_split(records, seed=7)
    assert [r.number for r in first.held_out] == [r.number for r in second.held_out]
    assert [r.number for r in first.corpus] == [r.number for r in second.corpus]


def test_held_out_split_seed_changes_split():
    records = [make_record("x/y", i) for i in range(20)]
    first = held_out_split(records, seed=1)
    second = held_out_split(records, seed=2)
    assert [r.number for r in first.held_out] != [r.number for r in second.held_out]


def test_held_out_split_does_not_mutate_input():
    records = [make_record("x/y", i) for i in range(20)]
    original = [r.number for r in records]
    held_out_split(records)
    assert [r.number for r in records] == original


def test_held_out_split_rejects_invalid_fraction():
    records = [make_record("x/y", i) for i in range(10)]
    with pytest.raises(ValueError):
        held_out_split(records, held_out_fraction=0)
    with pytest.raises(ValueError):
        held_out_split(records, held_out_fraction=1.0)

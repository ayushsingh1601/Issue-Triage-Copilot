import pytest
from triage.evals.metrics import (
    action_overlap,
    label_accuracy,
    precision_at_k,
    recall_at_k,
    rouge_l_f1,
)


def test_label_accuracy_top1_and_top3():
    assert label_accuracy(["bug", "DataFrame"], ["bug"]) == {"top1": 1.0, "top3": 1.0}
    assert label_accuracy(["DataFrame", "bug"], ["bug"]) == {"top1": 0.0, "top3": 1.0}
    assert label_accuracy(["enhancement"], ["bug"]) == {"top1": 0.0, "top3": 0.0}
    assert label_accuracy([], ["bug"]) == {"top1": 0.0, "top3": 0.0}


def test_rouge_l_identical_and_disjoint():
    assert rouge_l_f1("a b c", "a b c") == 1.0
    assert rouge_l_f1("a b c", "x y z") == 0.0


def test_rouge_l_partial():
    assert rouge_l_f1("a b", "a") == pytest.approx(0.6667, abs=1e-3)


def test_action_overlap_matches_reference():
    result = action_overlap(["Add a regression test for empty frames"], "Add a regression test")
    assert result["rouge_l"] > 0.5


def test_action_overlap_entity_match():
    result = action_overlap(
        ["Merge PR 45 to fix the crash"],
        "fixed by PR 45",
        referenced_prs=[45],
    )
    assert result["entity_match"] == 1.0


def test_action_overlap_empty_actions():
    assert action_overlap([], "fixed") == {"rouge_l": 0.0, "entity_match": 0.0}


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}) == 1.0
    assert recall_at_k(["a", "b"], {"c"}) == 0.0
    assert recall_at_k(["a", "b", "c"], {"a", "d"}) == pytest.approx(0.5)
    assert recall_at_k(["a"], set()) == 0.0


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}) == pytest.approx(2 / 3)
    assert precision_at_k(["a", "b", "c"], {"d"}) == 0.0
    assert precision_at_k(["a"], {"a"}) == 1.0
    assert precision_at_k([], {"a"}) == 0.0

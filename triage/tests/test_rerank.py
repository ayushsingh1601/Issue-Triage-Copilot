import json

from triage.prompts.rerank import build_rerank_prompt
from triage.rag.rerank import Reranker
from triage.rag.store import Match


def make_match(index: int) -> Match:
    return Match(id=f"id{index}", text=f"text {index}", metadata={}, distance=float(index))


def fixed_order(order):
    return lambda prompt: json.dumps({"order": order, "reason": "ok"})


def test_rerank_reorders_by_indices():
    matches = [make_match(i) for i in range(3)]
    reranker = Reranker(respond=fixed_order([2, 0, 1]))
    ranked = reranker.rerank("q", matches, top_k=3)
    assert [m.id for m in ranked] == ["id2", "id0", "id1"]


def test_rerank_respects_top_k():
    matches = [make_match(i) for i in range(3)]
    reranker = Reranker(respond=fixed_order([1, 2, 0]))
    assert [m.id for m in reranker.rerank("q", matches, top_k=2)] == ["id1", "id2"]


def test_rerank_ignores_bad_indices_and_appends_missing():
    matches = [make_match(i) for i in range(3)]
    reranker = Reranker(respond=fixed_order([5, 0, 0, 2]))
    ranked = reranker.rerank("q", matches, top_k=3)
    assert [m.id for m in ranked] == ["id0", "id2", "id1"]


def test_rerank_single_match_passthrough():
    matches = [make_match(0)]
    reranker = Reranker(respond=fixed_order([0]))
    assert reranker.rerank("q", matches, top_k=3) == matches


def test_rerank_prompt_includes_candidates():
    prompt = build_rerank_prompt("crash", [make_match(0), make_match(1)])
    assert "0: text 0" in prompt
    assert "crash" in prompt
    assert '"order"' in prompt

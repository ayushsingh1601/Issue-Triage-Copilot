import json

from triage.evals.judge import Judge
from triage.prompts.judge import METRICS, build_judge_prompt

QUERY = "crash when reading an empty CSV into a DataFrame"
CONTEXT = 'x/y#1: "crash on empty frame", fixed by #45'
GOOD_DECISION = 'citations: ["x/y#1"]; labels: ["bug"]'
BAD_DECISION = 'citations: ["FAKE#999"]; labels: ["bug"]'


def fixed_respond(score: int, reason: str):
    return lambda prompt: json.dumps({"score": score, "reason": reason})


def test_known_good_sample_scores_high():
    judge = Judge(respond=fixed_respond(5, "decision is grounded"))
    score = judge.score("groundedness", QUERY, GOOD_DECISION, CONTEXT)
    assert score.score == 5
    assert score.metric == "groundedness"
    assert score.reason == "decision is grounded"


def test_known_bad_sample_scores_low():
    judge = Judge(respond=fixed_respond(1, "citation does not exist in context"))
    score = judge.score("groundedness", QUERY, BAD_DECISION, CONTEXT)
    assert score.score == 1


def test_evaluate_returns_all_metrics():
    judge = Judge(respond=fixed_respond(4, "ok"))
    results = judge.evaluate(QUERY, GOOD_DECISION, CONTEXT)
    assert set(results) == set(METRICS)
    assert all(score.score == 4 for score in results.values())


def test_prompt_includes_metric_and_context():
    prompt = build_judge_prompt("groundedness", QUERY, GOOD_DECISION, CONTEXT)
    assert "groundedness" in prompt
    assert QUERY in prompt
    assert "x/y#1" in prompt

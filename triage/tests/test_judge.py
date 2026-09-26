import json

from triage.evals.judge import Judge
from triage.prompts.judge import METRICS, build_judge_prompt

QUERY = "crash when reading an empty CSV into a DataFrame"
CONTEXT = 'x/y#1: "crash on empty frame", fixed by #45'
GOOD_DECISION = 'citations: ["x/y#1"]; labels: ["bug"]'
BAD_DECISION = 'citations: ["FAKE#999"]; labels: ["bug"]'


def verdict_respond(verdict: str, reason: str = "ok"):
    return lambda prompt: json.dumps({"verdict": verdict, "reason": reason})


def test_known_good_sample_scores_yes():
    judge = Judge(respond=verdict_respond("yes"))
    score = judge.score("groundedness", QUERY, GOOD_DECISION, CONTEXT)
    assert score.verdict == "yes"
    assert score.score == 1
    assert score.metric == "groundedness"


def test_known_bad_sample_scores_no():
    judge = Judge(respond=verdict_respond("no", "citation does not exist in context"))
    score = judge.score("groundedness", QUERY, BAD_DECISION, CONTEXT)
    assert score.verdict == "no"
    assert score.score == 0


def test_evaluate_returns_all_metrics_as_binary():
    judge = Judge(respond=verdict_respond("yes"))
    results = judge.evaluate(QUERY, GOOD_DECISION, CONTEXT)
    assert set(results) == set(METRICS)
    assert all(score.verdict == "yes" and score.score == 1 for score in results.values())


def test_each_metric_gets_its_own_llm_call():
    calls = {metric: 0 for metric in METRICS}

    def make_respond(metric):
        def respond(prompt):
            calls[metric] += 1
            return json.dumps({"verdict": "yes", "reason": "ok"})

        return respond

    judge = Judge(responds={metric: make_respond(metric) for metric in METRICS})
    judge.evaluate(QUERY, GOOD_DECISION, CONTEXT)
    assert calls == {metric: 1 for metric in METRICS}


def test_prompt_asks_for_binary_verdict():
    prompt = build_judge_prompt("groundedness", QUERY, GOOD_DECISION, CONTEXT)
    assert "verdict" in prompt
    assert QUERY in prompt
    assert "x/y#1" in prompt


def test_judge_threads_config_to_respond():
    seen = {}

    def respond(prompt, config=None):
        seen["config"] = config
        return json.dumps({"verdict": "yes", "reason": "ok"})

    judge = Judge(respond=respond)
    judge.score("groundedness", QUERY, GOOD_DECISION, CONTEXT, config={"callbacks": ["x"]})
    assert seen["config"] == {"callbacks": ["x"]}


def test_judge_callbacks_fire():
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage

    class Recorder(BaseCallbackHandler):
        def __init__(self) -> None:
            self.llm_ends = 0

        def on_llm_end(self, response, **kwargs) -> None:
            self.llm_ends += 1

    def respond(prompt, config=None):
        model = FakeMessagesListChatModel(
            responses=[AIMessage(content='{"verdict": "yes", "reason": "ok"}')]
        )
        return model.invoke([HumanMessage(content=prompt)], config=config).content

    recorder = Recorder()
    judge = Judge(respond=respond)
    judge.score("groundedness", QUERY, GOOD_DECISION, CONTEXT, config={"callbacks": [recorder]})
    assert recorder.llm_ends >= 1

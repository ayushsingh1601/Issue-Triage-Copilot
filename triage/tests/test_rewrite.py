import json

from triage.prompts.rewrite import build_rewrite_prompt
from triage.rag.rewrite import QueryRewriter


def fixed_respond(text: str):
    return lambda prompt: json.dumps({"query": text})


def test_rewrite_returns_clean_query():
    rewriter = QueryRewriter(respond=fixed_respond("dataframe crash on empty csv"))
    query = rewriter.rewrite("DataFrame crashes when reading empty CSV\n\nstack trace...")
    assert query == "dataframe crash on empty csv"


def test_rewrite_handles_fenced_json():
    rewriter = QueryRewriter(respond=lambda prompt: '```json\n{"query": "merge crash"}\n```')
    assert rewriter.rewrite("anything") == "merge crash"


def test_rewrite_prompt_includes_issue():
    prompt = build_rewrite_prompt("crash on empty frame")
    assert "crash on empty frame" in prompt
    assert '"query"' in prompt

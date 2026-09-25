"""Prompt for the LLM reranker."""
from __future__ import annotations

from triage.rag.store import Match


def build_rerank_prompt(query: str, matches: list[Match]) -> str:
    candidates = "\n".join(f"{index}: {_snip(match)}" for index, match in enumerate(matches))
    return f"""Rank the numbered candidate passages by relevance to the issue below, best first.

Respond with JSON only:
{{"order": [list of indices, best first], "reason": "<one sentence>"}}

ISSUE:
{query}

CANDIDATES:
{candidates}"""


def _snip(match: Match, limit: int = 200) -> str:
    title = match.metadata.get("title", "")
    text = match.text
    if title and not text.startswith(title):
        text = f"{title}: {text}"
    return text[:limit]

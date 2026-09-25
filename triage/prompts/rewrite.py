"""Prompt for the query rewriter."""
from __future__ import annotations


def build_rewrite_prompt(query: str) -> str:
    return f"""Rewrite the following GitHub issue text into a single concise search query for
finding similar previously-resolved issues and relevant process docs. Keep the core problem,
key terms, and any module names.

Respond with JSON only: {{"query": "<rewritten query>"}}

ISSUE:
{query}"""

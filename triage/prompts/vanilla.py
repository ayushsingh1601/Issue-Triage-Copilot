"""Prompt for the vanilla RAG baseline: dual-index context in a single prompt."""
from __future__ import annotations

from triage.rag.store import Match


def build_vanilla_prompt(
    query: str,
    issue_matches: list[Match],
    doc_matches: list[Match],
) -> str:
    return f"""You are an issue triage copilot for an open-source project. Recommend the next
actions for the new issue below.

NEW ISSUE:
{query}

SIMILAR RESOLVED ISSUES (citation = repo#number):
{_format_issues(issue_matches)}

PROCESS DOC EXCERPTS (citation = path (heading)):
{_format_docs(doc_matches)}

Respond with JSON only, exactly this schema:
{{
  "issue_id": "string",
  "suggested_labels": ["string"],
  "triage_route": "string",
  "next_steps": ["string"],
  "affected_modules": ["string"],
  "similar_issues": [
    {{"issue_id": "string", "repo": "string", "title": "string", "reason": "string"}}
  ],
  "citations": ["string"]
}}

Ground every recommendation in a citation from the provided sources
(repo#number or path (heading)).
similar_issues must come only from the SIMILAR RESOLVED ISSUES list above.
Give up to 4 similar issues. Keep citations to sources that actually appear above."""


def _format_issues(matches: list[Match]) -> str:
    return "\n".join(
        f"- [{match.id}] {match.metadata.get('title', '')}\n{match.text}" for match in matches
    )


def _format_docs(matches: list[Match]) -> str:
    return "\n".join(
        f"- [{match.metadata.get('path', '')} ({match.metadata.get('heading_path', '')})]\n"
        f"{match.text}"
        for match in matches
    )

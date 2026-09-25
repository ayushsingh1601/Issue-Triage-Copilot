"""Prompt for the classify_issue tool."""
from __future__ import annotations


def build_classify_prompt(issue: str, issue_types: list[str]) -> str:
    return (
        f"Classify the following issue into exactly one of these types: {issue_types}.\n"
        'Respond with JSON only: {"type": "<one type>", "confidence": <0-1 float>}.\n\n'
        f"ISSUE:\n{issue}"
    )

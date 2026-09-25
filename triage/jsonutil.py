"""Helpers for extracting JSON from LLM responses."""
from __future__ import annotations

import re


def extract_json(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return text
